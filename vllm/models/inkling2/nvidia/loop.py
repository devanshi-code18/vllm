# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 checkpoints (`loop_config`).

The core layers `[core_start, core_end)` run `num_passes` times: a prelude,
then each pass from the AttnRes state after the prelude, then the coda from
the last pass's state. Each pass of a core layer has its own KV and
conv-state caches. In passes after the first, every core sublayer input
(attention q/k/v and MLP) is blended with a message: a normalized-value
AttnRes read of the previous pass's blocks, merged by a gated low-rank
controller. Each sublayer's short conv runs fused with the next AttnRes mix
(and small batches at TP > 1 with its all-reduce); without sequence parallel.

With `loop_max_passes`, each request picks its pass count
(`SamplingParams.extra_args["loops"]`). A batch runs its largest count, and
every row enters the coda from the state of its own last pass; the batch's
largest count selects the CUDA graphs (one set per count).
"""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .model import (
    Inkling2DecoderLayer,
    Inkling2ForConditionalGeneration,
    Inkling2Model,
)
from .ops.attn_res import attn_res
from .ops.linear import linear_k_groups
from .ops.ll_gemm_sm120 import sm12x_bf16_ll_gemm, use_sm12x_bf16_ll_gemm
from .ops.loop import blend_rms_norm, pair_rms_norm
from .sconv import conv_metadata

LOOP_STREAMS = ("attn", "k", "v", "mlp")


class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    The weights are views of the layer's stream-stacked weights
    (`Inkling2LoopDecoderLayer`), which run every stream at once; `forward`
    is the per-stream reference.
    """

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """fp32 `(T, D)` `dest` and `message` -> fp32 `(T, D)`."""
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        hidden = linear_k_groups(normalized, self.down_weight, torch.bfloat16)
        gain = torch.mm(hidden, self.gain_weight.t(), out_dtype=torch.float32)
        forget = torch.mm(hidden, self.forget_weight.t(), out_dtype=torch.float32)
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16.
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def process_weights_after_loading(self) -> None:
        # The controllers' weights must still alias the stacked ones.
        for i, controller in enumerate(self.controllers.values()):
            down, gain_forget = self.controller_down, self.controller_gain_forget
            assert controller.down_weight.data_ptr() == down[i].data_ptr()
            assert controller.gain_weight.data_ptr() == gain_forget[i].data_ptr()

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        pair = pair_rms_norm(dest, message)
        down = self.controller_down[streams]
        up = self.controller_gain_forget[streams]
        if use_sm12x_bf16_ll_gemm(pair, down):
            hidden = sm12x_bf16_ll_gemm(pair, down)
            gain_forget = sm12x_bf16_ll_gemm(hidden, up, out_dtype=torch.float32)
        else:
            hidden = torch.bmm(pair, down.transpose(1, 2))
            gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)


class Inkling2LoopModel(Inkling2Model):
    seq_parallel_paths = False

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        # Serving options set through --hf-overrides skip the constructor.
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        # Per core layer, the `LOOP_STREAMS` queries scaled by D**-0.5.
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks a loop forward holds at most: each pass
        restarts from the prelude's, and the coda continues the last pass."""
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: nn.Module | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        plan: list[tuple[bool, bool]],
        skip: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Run decoder layer `i` in loop pass `pass_idx` on its `inputs`
        (the attention inputs; for a visit with messages from `bank`, the
        fp32 mixes they blend into). Returns the number of completed AttnRes
        blocks after it and the mixes of its output with `next_query` through
        `next_norm` (fp32 without it), the next visit's inputs. Rows of the
        bool `skip` (past their pass count) skip the experts."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        workspace = self.allreduce_workspace
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        messages = None
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            # All four streams' messages: a mix of the bank's two blocks.
            messages = attn_res(
                bank[:1],
                bank[1],
                None,
                self.message_queries[i - self.loop_core.start],
                None,
                0.0,
                True,
            )
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            attn_inputs = layer.blend(
                slice(0, 3), dests, messages[:3], norm_weight, norm.eps
            )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)
        mix_args = (
            positions,
            attn_out,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if messages is not None else layer.mlp_norm,
        )
        if attn_fused:
            assert workspace is not None
            mlp_in = layer.attn_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                x=attn_partial,
                two_shot=attn_two_shot,
                owner=owner,
            )
        else:
            attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
            mlp_in = layer.attn_sconv.conv_mix(attn_reduced, *mix_args, owner=owner)
        if messages is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer)
            norm = layer.mlp_norm
            mlp_in = layer.blend(
                slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
            )

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        mix_args = (
            positions,
            stack[num_blocks],
            stack[:num_blocks],
            None,
            next_query,
            next_norm,
        )
        if moe_fused:
            assert workspace is not None
            outputs = layer.mlp_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                combine=layer.mlp(
                    mlp_in[0], reduce=False, pass_idx=pass_idx, skip=skip
                ),
                addend=attn_out,
                residual=residual,
                two_shot=moe_two_shot,
                owner=owner,
            )
        else:
            # The MoE output is all-reduced (in bf16) inside layer.mlp.
            outputs = layer.mlp_sconv.conv_mix(
                layer.mlp(mlp_in[0], pass_idx=pass_idx, skip=skip),
                *mix_args,
                addend=attn_out,
                residual=residual,
                owner=owner,
            )
        return num_blocks, outputs

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        loop_passes: torch.Tensor | None = None,
        num_loop_passes: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Prelude, `num_passes` core passes, then the coda. Each sublayer's
        short conv runs fused with the next AttnRes mix (and small batches
        with its all-reduce).

        With per-request pass counts, `loop_passes` holds each token's count
        and `num_loop_passes` their maximum: every row runs the maximum, and
        its last pass's blocks and coda inputs are kept for the coda."""
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = (
            self.config.default_loop_passes
            if num_loop_passes is None
            else num_loop_passes
        )
        per_row = loop_passes is not None and num_passes > 1
        kept: tuple[torch.Tensor, torch.Tensor] | None = None
        skip: torch.Tensor | None = None
        stack = inputs_embeds.new_empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden)
        )
        stack[0].copy_(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        workspace = self.allreduce_workspace
        plan = [(False, False)] * 2
        if (
            workspace is not None
            and conv_metadata(self.layers[0].conv_state) is not None
        ):
            plan = workspace.plan(num_tokens)
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        next_norm: nn.Module | None
        num_blocks = 0
        inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                if per_row:
                    assert loop_passes is not None
                    # Rows past their pass count skip the experts.
                    skip = loop_passes <= pass_idx
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                if per_row and i == core.stop - 1:
                    # Each pass ends with the coda's inputs, as if it were
                    # the last (a restart discards them).
                    next_i, next_pass = core.stop, 0
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
                plan,
                skip if looped else None,
            )
            if per_row and i == core.stop - 1:
                assert loop_passes is not None and prelude_blocks is not None
                # The pass's blocks (the next pass's message bank) and the
                # coda's inputs, kept for rows with at least `pass_idx + 1`
                # passes.
                blocks = stack[prelude_blocks + 1 : num_blocks + 1]
                if kept is None:
                    kept = blocks.clone(), inputs.clone()
                else:
                    keep = (loop_passes > pass_idx).view(1, -1, 1)
                    kept = (
                        torch.where(keep, blocks, kept[0]),
                        torch.where(keep, inputs, kept[1]),
                    )
                if pass_idx + 1 == num_passes:
                    blocks.copy_(kept[0])
                    inputs = kept[1]
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopModelState(DefaultModelState):
    """Per-request pass counts (`loop_max_passes`): each token's count for
    the forward, and the batch's largest as its CUDA graph variant."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_config = self.model_config.hf_text_config
        self.max_passes: int | None = self.text_config.loop_max_passes
        self.req_passes: dict[str, int] = {}
        if self.max_passes is None:
            return
        self.cudagraph_variant = self.max_passes
        self.loop_passes = torch.ones(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        )
        self.loop_passes_staging = UvaBufferPool(self.max_num_tokens, torch.int32)

    def cudagraph_variants(self) -> tuple[Hashable, ...]:
        if self.max_passes is None:
            return (None,)
        return tuple(range(1, self.max_passes + 1))

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self.max_passes is not None:
            params = new_req_data.sampling_params
            self.req_passes[new_req_data.req_id] = self.text_config.request_loop_passes(
                params and params.extra_args
            )

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.req_passes.pop(req_id, None)

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        if self.max_passes is None:
            return inputs
        num_reqs = input_batch.num_reqs
        # Dummy batches (profiling, warmup) run every pass.
        passes = np.array(
            [
                self.req_passes.get(req_id, self.max_passes)
                for req_id in input_batch.req_ids[:num_reqs]
            ],
            dtype=np.int32,
        )
        num_tokens = input_batch.num_tokens
        padded = input_batch.num_tokens_after_padding
        per_token = np.ones(padded, dtype=np.int32)
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        per_token[:num_tokens] = np.repeat(passes, query_lens)[:num_tokens]
        loop_passes = self.loop_passes_staging.copy_to_gpu(
            per_token, out=self.loop_passes[:padded]
        )
        self.cudagraph_variant = int(passes.max()) if num_reqs else self.max_passes
        return inputs | {
            "loop_passes": loop_passes,
            "num_loop_passes": self.cudagraph_variant,
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.max_passes is None:
            return inputs
        # CUDA graph capture: every row runs the variant's passes.
        passes = self.cudagraph_variant
        assert isinstance(passes, int)
        self.loop_passes[:num_tokens].fill_(passes)
        return inputs | {
            "loop_passes": self.loop_passes[:num_tokens],
            "num_loop_passes": passes,
        }


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` checkpoints' architecture name; the model is
    `Inkling2ForConditionalGeneration` with an `Inkling2LoopModel`."""

    @staticmethod
    def get_model_state_cls() -> type[Inkling2LoopModelState]:
        return Inkling2LoopModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """`loop_passes` and `num_loop_passes` (per-request pass counts) come
        from `Inkling2LoopModelState.prepare_inputs`."""
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        loop_passes = kwargs.get("loop_passes")
        num_loop_passes = kwargs.get("num_loop_passes")
        assert loop_passes is None or isinstance(loop_passes, torch.Tensor)
        assert num_loop_passes is None or isinstance(num_loop_passes, int)
        return self.model(positions, inputs_embeds, loop_passes, num_loop_passes)
