# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short convolutions with their state kept in a sliding-window cache.

Each decoder layer owns one `InklingConvState` (shared with Inkling v1)
holding its four conv streams (K, V, attn-out, mlp-out) packed head-major. K/V
are TP-sharded with the KV heads; the attn-/mlp-output streams are replicated
full-width on every rank (`replicate_hidden=True`).
"""

import os
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.nvidia.sconv_swa_attn import (
    InklingConvState,
    InklingSconvBackend,
    InklingSconvMetadata,
    InklingSconvMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig

from .ops import use_native_conv_mix
from .ops.allreduce import Inkling2AllReduceWorkspace, all_reduce_conv_mix
from .ops.conv_mix import conv_mix as fused_conv_mix
from .ops.linear import linear_k_groups
from .ops.ll_gemm_sm120 import sm12x_bf16_ll_gemm, use_sm12x_bf16_ll_gemm
from .ops.sconv import SKIP_WRITE_SLOT, short_conv
from .ops.seq_parallel import SeqParallel

K_STREAM, V_STREAM, ATTN_STREAM, MLP_STREAM = range(4)


# Prefix-cache hits land on multiples of this many tokens; the conv inputs
# just before each multiple are kept so a hit can resume there.
CONV_RETAIN_STRIDE = int(os.environ.get("VLLM_INKLING2_CONV_RETAIN_STRIDE", "64"))


def conv_tail_only() -> bool:
    """Whether the conv-state cache keeps only the rows later steps read
    (`VLLM_INKLING2_CONV_TAIL=1`, Triton conv kernels only)."""
    return os.environ.get("VLLM_INKLING2_CONV_TAIL", "0") == "1" and (
        not use_native_conv_mix()
    )


class Inkling2ConvStateManager(SlidingWindowManager):
    """Conv-state blocks only where a later step reads: the last
    `sliding_window - 1` tokens of each step and, with prefix caching, of
    every `retain_stride` tokens (where a prefix-cache hit can resume).

    The conv kernels read the taps of tokens scheduled in the same step from
    their inputs, so the conv state of a prefill chunk's other tokens is
    never read and needs no block; their writes are skipped
    (`SKIP_WRITE_SLOT`). A chunk of `C` tokens then holds about
    `C / retain_stride` conv blocks instead of `C / block_size`.
    """

    @dataclass(frozen=True, slots=True)
    class Params(SlidingWindowManager.Params):
        retain_stride: int = 64

    @classmethod
    def max_admission_blocks(
        cls,
        config,
        block_size: int,
        max_in_flight_tokens: int,
        max_model_len: int,
    ) -> int | None:
        params = config.params
        assert isinstance(params, cls.Params)
        num_tokens = min(
            params.sliding_window - 1 + max_in_flight_tokens, max_model_len
        )
        # One block per stride, plus the previous and the current tails.
        return min(
            cdiv(num_tokens, params.retain_stride) + 4, cdiv(num_tokens, block_size) + 1
        )

    def __init__(self, kv_cache_group, **kwargs) -> None:
        super().__init__(kv_cache_group, **kwargs)
        params = self.manager_config.params
        assert isinstance(params, self.Params)
        self.retain_stride = params.retain_stride
        self.num_tail_tokens = params.sliding_window - 1
        assert self.retain_stride % self.block_size == 0
        assert self.block_size >= self.num_tail_tokens
        assert self.scheduler_block_size % self.retain_stride == 0, (
            "prefix-cache hits must land on retained conv rows"
        )
        # Per request: blocks below this index are already freed or null.
        self._freed_up_to: dict[str, int] = {}

    def _needed(self, block_idx: int, num_tokens: int) -> bool:
        end = (block_idx + 1) * self.block_size
        if end > num_tokens - self.num_tail_tokens:
            return True
        return self.enable_caching and end % self.retain_stride == 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
        num_evictable_blocks = 0
        if request_id in self.num_cached_block:
            assert len(new_computed_blocks) == 0
            start = num_req_blocks
        else:
            num_skipped_blocks = (
                self.get_num_skipped_tokens(total_computed_tokens) // self.block_size
            )
            start = max(num_skipped_blocks, len(new_computed_blocks) + num_req_blocks)
            skipped_new = max(0, num_skipped_blocks - num_req_blocks)
            num_evictable_blocks = self._get_num_evictable_blocks(
                new_computed_blocks[skipped_new:]
            )
        num_new = sum(
            self._needed(j, num_tokens) for j in range(start, num_required_blocks)
        )
        return num_new + num_evictable_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ):
        assert request_id not in self._partial_hit_reqs
        req_blocks = self.req_to_blocks[request_id]
        indices = range(len(req_blocks), cdiv(num_tokens, self.block_size))
        needed = [self._needed(j, num_tokens) for j in indices]
        new_blocks = iter(self.block_pool.get_new_blocks(sum(needed)))
        entries = [next(new_blocks) if n else self._null_block for n in needed]
        req_blocks.extend(entries)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in entries if not b.is_null)
        return entries

    def _remove_blocks_in_range(
        self, request_id: str, first_block: int, last_block: int
    ) -> None:
        # Blocks are interleaved with nulls, so scan the whole range (once).
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        last_block = min(last_block, len(blocks))
        first_block = max(first_block, self._freed_up_to.get(request_id, 0))
        freed = []
        for i in range(last_block - 1, first_block - 1, -1):
            if not blocks[i].is_null:
                freed.append(blocks[i])
                blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)
        self._freed_up_to[request_id] = max(first_block, last_block)

    def pop_blocks_for_free(self, request_id: str):
        self._freed_up_to.pop(request_id, None)
        return [b for b in super().pop_blocks_for_free(request_id) if not b.is_null]


class Inkling2SconvMetadataBuilder(InklingSconvMetadataBuilder):
    """With `conv_tail_only`, tokens whose conv state no later step reads
    get `SKIP_WRITE_SLOT` (see `Inkling2ConvStateManager`)."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.tail_only = conv_tail_only()
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.slot_buffer = torch.empty(max_num_tokens, dtype=torch.int64, device=device)
        self.token_index = torch.arange(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self.skip_slot = torch.tensor(SKIP_WRITE_SLOT, dtype=torch.int64, device=device)
        text_config = vllm_config.model_config.hf_text_config
        self.num_tail_tokens = text_config.sconv_kernel_size - 1
        self.retain_stride = (
            CONV_RETAIN_STRIDE if vllm_config.cache_config.enable_prefix_caching else 0
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> InklingSconvMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if not self.tail_only:
            return md
        n = md.slot_mapping.shape[0]
        last_req = max(common_attn_metadata.num_reqs - 1, 0)
        req = md.seq_idx.long().clamp_(0, last_req)
        end = common_attn_metadata.seq_lens[req].long()
        query_end = common_attn_metadata.query_start_loc[req + 1].long()
        pos = end - (query_end - self.token_index[:n])
        keep = end - pos <= self.num_tail_tokens
        if stride := self.retain_stride:
            keep |= pos % stride >= stride - self.num_tail_tokens
        slots = md.slot_mapping
        out = self.slot_buffer[:n]
        torch.where(keep | (slots < 0), slots, self.skip_slot, out=out)
        md.slot_mapping = out
        return md


class Inkling2SconvBackend(InklingSconvBackend):
    """The conv-state cache's metadata; it runs no kernel."""

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # Inkling2 is batch invariant where validated (A100); other GPUs run
        # other attention, MoE and GEMM kernels.
        return current_platform.is_device_capability(80)

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return Inkling2SconvMetadataBuilder


class Inkling2ConvState(InklingConvState):
    """`InklingConvState` with Inkling2's batch-invariance support and, with
    `conv_tail_only`, blocks only for the conv rows later steps read."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2SconvBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        config = super().get_kv_cache_config(vllm_config)
        if not conv_tail_only():
            return config
        return Inkling2ConvStateManager.layer_config(
            config.spec,
            sliding_window=self.sliding_window,
            retain_stride=CONV_RETAIN_STRIDE,
        )


def conv_metadata(owner: InklingConvState) -> InklingSconvMetadata | None:
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict) or owner.kv_cache.numel() == 0:
        return None
    return cast(InklingSconvMetadata | None, attn_metadata.get(owner.prefix))


class Inkling2ShortConv(nn.Module):
    """Causal depthwise conv (+ residual, scaled) over one conv-state stream.

    Static: `weight` of shape `(dim, 1, W)`. Dynamic: per-token taps
    `static_w (W, dim) + weight_proj(dynamic_input)` per head. The K/V convs
    only hold weights; the attention prologue kernel runs both.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        owner: InklingConvState,
        stream: int,
        *,
        dynamic_input_size: int | None = None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.stream = stream
        self.dynamic = dynamic_input_size is not None
        self.tp_rank = get_tensor_model_parallel_rank()
        if dynamic_input_size is not None:
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, dim), requires_grad=False
            )
            self.weight_proj = nn.Parameter(
                torch.empty(num_heads * kernel_size, dynamic_input_size),
                requires_grad=False,
            )
            set_weight_attrs(self.static_w, {"weight_loader": self._load_dim1})
            set_weight_attrs(self.weight_proj, {"weight_loader": self._load_dim0})
        else:
            self.weight = nn.Parameter(
                torch.empty(dim, 1, kernel_size), requires_grad=False
            )
            set_weight_attrs(self.weight, {"weight_loader": self._load_dim0})

    def _load(self, param: nn.Parameter, loaded: torch.Tensor, dim: int) -> None:
        size = param.shape[dim]
        if loaded.shape[dim] != size:
            loaded = loaded.narrow(dim, self.tp_rank * size, size)
        param.data.copy_(loaded)

    def _load_dim0(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 0)

    def _load_dim1(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        self._load(param, loaded, 1)

    @property
    def conv_weight(self) -> torch.Tensor:
        """Static `(dim, W)` taps, or `(W, dim)` static part of dynamic taps."""
        return self.static_w if self.dynamic else self.weight.squeeze(1)

    def dynamic_taps(self, x: torch.Tensor) -> torch.Tensor | None:
        """Per-token tap deltas `(T, num_heads * W)` projected from `x`."""
        if not self.dynamic:
            return None
        # Same dispatch as the reference ``linear``: fp32 1024-wide K groups.
        if self.weight_proj.shape[1] >= 2560 and self.weight_proj.shape[0] not in (
            1,
            2,
            4,
        ):
            return linear_k_groups(x.contiguous(), self.weight_proj, x.dtype)
        if use_sm12x_bf16_ll_gemm(x, self.weight_proj):
            return sm12x_bf16_ll_gemm(x, self.weight_proj)
        return nn.functional.linear(x, self.weight_proj)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        """Static conv of `x`; with `addend`, returns `(residual + addend)
        + conv(x)` accumulated in fp32 (written to `out` if given). `owner`
        overrides the conv-state cache (a loop pass after the first)."""
        assert not self.dynamic
        if owner is None:
            owner = self.owner
        m = conv_metadata(owner)
        if m is None:
            # No conv state (profile run): skip the conv, keep the residual sum.
            if addend is not None:
                acc = addend.float() if residual is None else residual + addend.float()
                x = (acc + x.to(addend.dtype).float()).to(addend.dtype)
            return x if out is None else out.copy_(x)
        offset, width = owner.stream_ranges[self.stream]
        return short_conv(
            x,
            self.conv_weight,
            owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            addend,
            residual,
            out,
        )

    def conv_mix(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
        stack: torch.Tensor,
        last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module | None,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        sp: SeqParallel | None = None,
        halo: int = 0,
        prefix_align: int = 0,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        """Convolve the reduced `x` into `out` like `forward` and return
        the AttnRes mixes of `[*stack, (last +) out]` with `queries`
        through `norm` (the fp32 mixes without it). With `sp`, the hidden
        stream is this rank's token shard, `x` its reduce-scatter (after
        `halo` rows) and the mixes are all-gathered. `owner` overrides the
        conv-state cache (a loop pass after the first)."""
        assert not self.dynamic
        if owner is None:
            owner = self.owner
        m = conv_metadata(owner)
        cache = owner.kv_cache
        if m is None:
            m, cache, align = self._profile_metadata(positions)
            prefix_align = align if sp is not None else 0
        offset, width = owner.stream_ranges[self.stream]
        return fused_conv_mix(
            x,
            self.conv_weight,
            cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            out,
            residual,
            addend,
            stack,
            last,
            queries,
            None if norm is None else norm.weight,
            0.0 if norm is None else norm.eps,
            sp=sp,
            halo=halo,
            prefix_align=prefix_align,
        )

    def all_reduce_mix(
        self,
        workspace: Inkling2AllReduceWorkspace,
        positions: torch.Tensor,
        out: torch.Tensor,
        stack: torch.Tensor,
        last: torch.Tensor | None,
        queries: torch.Tensor,
        norm: nn.Module | None,
        x: torch.Tensor | None = None,
        combine: tuple[torch.Tensor, torch.Tensor] | None = None,
        addend: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        two_shot: bool = False,
        owner: InklingConvState | None = None,
    ) -> torch.Tensor:
        """All-reduce this rank's partial sum (`x`, or the MoE `combine`)
        in one or `two_shot`s, convolve it into `out` like `forward`,
        and return the AttnRes mixes of `[*stack, (last +) out]` with
        `queries` through `norm` (the fp32 mixes without it). `owner`
        overrides the conv-state cache."""
        if owner is None:
            owner = self.owner
        m = conv_metadata(owner)
        assert m is not None and not self.dynamic
        offset, width = owner.stream_ranges[self.stream]
        return all_reduce_conv_mix(
            workspace,
            x,
            combine,
            self.conv_weight,
            owner.kv_cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            m.query_start,
            offset,
            width,
            out,
            residual,
            addend,
            stack,
            last,
            queries,
            None if norm is None else norm.weight,
            0.0 if norm is None else norm.eps,
            two_shot=two_shot,
        )

    def _profile_metadata(
        self, positions: torch.Tensor
    ) -> tuple[InklingSconvMetadata, torch.Tensor, int]:
        """Metadata of a batch without conv-state cache slots, a one-block
        cache and a prefix alignment for it."""
        owner = self.owner
        zeros = torch.zeros_like(positions, dtype=torch.int32)
        metadata = InklingSconvMetadata(
            block_table=zeros[:1].view(1, 1),
            slot_mapping=torch.full_like(positions, -1),
            seq_idx=zeros,
            query_start=zeros,
        )
        cache = torch.empty(
            (1, owner.num_kv_heads, owner.block_size, owner.head_size),
            dtype=torch.bfloat16,
            device=positions.device,
        )
        return metadata, cache, owner.block_size
