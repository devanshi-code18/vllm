# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Hopper: Humming expert GEMMs (bf16
activations, NVFP4 / MXFP8 weights) in place of the TRT-LLM-gen ones.

Rows stay in the flat (token * topk + slot) order: the gate/up GEMM
gathers each row's token, the down GEMM reads and writes row by row, and
`ops.moe.combine` sums them with the identity `src2dst` (-1: not local).
"""

import json
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.platform_utils import num_compute_units

from .activation import _situ
from .linear import _k_group_tile, k_group_partials
from .moe import (
    _load_bias,
    _local_experts,
    _select_experts,
)

if TYPE_CHECKING:
    from vllm.utils.humming import HummingInputSchema

# Row block size when no Humming tuning range covers the row count.
_DEFAULT_BLOCK_M = 64
# Up to this many tokens, `select_and_align` selects the experts and lays out
# the Humming rows in one launch.
ALIGN_MAX_TOKENS = 256
# Token mask words (32 tokens each) per local expert of `select_and_align`.
ALIGN_WORDS = ALIGN_MAX_TOKENS // 32


@triton.jit
def _select_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Local expert ids (-1: not local), routing weights and identity
    `src2dst` of the flat rows of one token."""
    token = tl.program_id(0) + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))


def select_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The router and top-k of `ops.moe.route`, without its layout.

    Returns the local expert id of each token's slots `[T, topk]` (-1: not
    one of the local experts `expert_bounds`, or a padding token), their
    routing weights `[T * topk]` and the identity `src2dst` `[T * topk]`
    of the local rows (-1 elsewhere).
    """
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    _select_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        num_warps=4 if tokens <= 256 else 1,
    )
    return ids, weights, src2dst


@triton.jit
def _align_rows(
    masks,
    local,
    Sorted,
    ExpertIds,
    Padded,
    numel,
    max_padded,
    TOPK: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """`moe_align_block_size(ids, BLOCK_M, NUM_LOCAL,
    ignore_invalid_experts=True)` of the flat local expert ids `local` (-1:
    none) from the token bit masks of each local expert `masks`
    [LOCAL_POW2, WORDS], by one program. The rows of an expert are in token
    order (the CUDA kernel places them with atomics, in any order)."""
    experts = tl.arange(0, LOCAL_POW2)
    local_mask = experts < NUM_LOCAL
    bits = tldevice.popc(masks)
    counts = tl.sum(bits, 1)
    padded = tl.cdiv(counts, BLOCK_M) * BLOCK_M
    ends = tl.cumsum(padded, 0)
    starts = ends - padded
    total = tl.sum(padded, 0)
    tl.store(Padded, total)
    # A routed row's place: the first row of its expert's tokens in its mask
    # word (the expert's first row plus its tokens of the words before) plus
    # the expert's tokens before it in the word.
    bases = starts[:, None] + tl.cumsum(bits, 1) - bits
    flat = tl.arange(0, local.shape[0])
    token = flat // TOPK
    offs = tl.maximum(local, 0) * WORDS + token // 32
    word = tl.gather(tl.reshape(masks, (LOCAL_POW2 * WORDS,)), offs, 0)
    below = (tl.full(flat.shape, 1, tl.int64) << (token % 32)) - 1
    rows = tl.gather(tl.reshape(bases, (LOCAL_POW2 * WORDS,)), offs, 0)
    rows += tldevice.popc(word.to(tl.int64) & below)
    tl.store(Sorted + rows, flat, local >= 0)
    # Each expert's padding rows and blocks, then those past the last expert.
    pad_rows = (starts + counts)[:, None] + tl.arange(0, BLOCK_M)[None, :]
    tl.store(Sorted + pad_rows, numel, local_mask[:, None] & (pad_rows < ends[:, None]))
    expert_blocks = (starts // BLOCK_M)[:, None] + tl.arange(0, MAX_BLOCKS)[None, :]
    tl.store(
        ExpertIds + expert_blocks,
        tl.broadcast_to(experts[:, None], (LOCAL_POW2, MAX_BLOCKS)),
        local_mask[:, None] & (expert_blocks < (ends // BLOCK_M)[:, None]),
    )
    for start in range(total, max_padded, 1024):
        tail = start + tl.arange(0, 1024)
        tl.store(Sorted + tail, numel, tail < max_padded)
    max_blocks = tl.cdiv(max_padded, BLOCK_M)
    for start in range(total // BLOCK_M, max_blocks, 1024):
        blocks = start + tl.arange(0, 1024)
        tl.store(ExpertIds + blocks, -1, blocks < max_blocks)


# Counters (`ops.moe.NUM_COUNTERS`) of the tokens `_select_align_kernel` has
# marked and of its router programs done, one 128-byte line apart (reset by
# the kernel).
_ALIGN_TOKENS_DONE = tl.constexpr(0)
_ALIGN_ROUTERS_DONE = tl.constexpr(32)
# Up to this many tokens on SM80 / SM90, the router GEMM runs in the same
# launch (its programs and the token programs fit on the GPU together).
ALIGN_ROUTER_MAX_TOKENS = 128


@triton.jit(do_not_specialize=["T", "route_scale", "max_padded", "max_padded_2"])
def _select_align_kernel(
    X,
    GateWeight,
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    Masks,
    Counters,
    Sorted,
    ExpertIds,
    Padded,
    Sorted2,
    ExpertIds2,
    Padded2,
    T,
    route_scale,
    max_padded,
    max_padded_2,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_M_2: tl.constexpr,
    MAX_BLOCKS_2: tl.constexpr,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    N_TILES: tl.constexpr,
    M_TILES: tl.constexpr,
    ROUTERS: tl.constexpr,
):
    # Programs [0, ROUTERS) compute the router's K-group partials
    # (`k_group_partials`; with no routers, they are given). Program
    # ROUTERS + t waits for them (all programs are co-resident) and runs
    # `_select_kernel` for token t, then marks the token in the mask of each
    # of its local experts (`Masks`, zeroed). The last token program to
    # finish lays out the rows of each block size (`_align_rows`) and resets
    # the masks and the counters.
    pid = tl.program_id(0)
    if pid < ROUTERS:
        _k_group_tile(
            X,
            GateWeight,
            Partials,
            T,
            E,
            H,
            BM,
            BN,
            True,
            pid % N_TILES,
            pid // N_TILES % M_TILES,
            pid // (N_TILES * M_TILES),
        )
        tl.debug_barrier()
        tl.atomic_add(Counters + _ALIGN_ROUTERS_DONE, 1, sem="release")
    else:
        _select_align_token(
            Partials,
            Bias,
            ExpertBounds,
            IsPadding,
            Ids,
            Weights,
            Src2Dst,
            Masks,
            Counters,
            Sorted,
            ExpertIds,
            Padded,
            Sorted2,
            ExpertIds2,
            Padded2,
            pid - ROUTERS,
            T,
            route_scale,
            max_padded,
            max_padded_2,
            E,
            E_POW2,
            GROUPS,
            TOPK,
            SORT,
            NUM_LOCAL,
            LOCAL_POW2,
            WORDS,
            BLOCK_M,
            MAX_BLOCKS,
            BLOCK_M_2,
            MAX_BLOCKS_2,
            ROWS,
            ROUTERS,
        )


@triton.jit
def _select_align_token(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    Masks,
    Counters,
    Sorted,
    ExpertIds,
    Padded,
    Sorted2,
    ExpertIds2,
    Padded2,
    pid,
    T,
    route_scale,
    max_padded,
    max_padded_2,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
    NUM_LOCAL: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_M_2: tl.constexpr,
    MAX_BLOCKS_2: tl.constexpr,
    ROWS: tl.constexpr,
    ROUTERS: tl.constexpr,
):
    token = pid + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    lo = tl.load(ExpertBounds)
    hi = tl.load(ExpertBounds + 1)
    if ROUTERS > 0:
        while tl.atomic_add(Counters + _ALIGN_ROUTERS_DONE, 0, sem="acquire") < ROUTERS:
            tl.inline_asm_elementwise(
                "nanosleep.u32 256; mov.b32 $0, 0;",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    local = _local_experts(ids, lo, hi, padding)
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))
    bit = (tl.full((1, 1), 1, tl.int64) << (token[:, None] % 32)).to(tl.int32)
    tl.atomic_or(
        Masks + local * WORDS + token[:, None] // 32,
        bit,
        mask=local >= 0,
        sem="relaxed",
    )
    tl.debug_barrier()
    if tl.atomic_add(Counters + _ALIGN_TOKENS_DONE, 1, sem="acq_rel") == T - 1:
        numel = T * TOPK
        flat = tl.arange(0, ROWS)
        flat_local = tl.load(Ids + flat, flat < numel, other=-1, cache_modifier=".cg")
        experts = tl.arange(0, LOCAL_POW2)[:, None]
        words = tl.arange(0, WORDS)[None, :]
        masks = tl.load(
            Masks + experts * WORDS + words,
            experts < NUM_LOCAL,
            other=0,
            cache_modifier=".cg",
        )
        _align_rows(
            masks,
            flat_local,
            Sorted,
            ExpertIds,
            Padded,
            numel,
            max_padded,
            TOPK,
            NUM_LOCAL,
            LOCAL_POW2,
            WORDS,
            BLOCK_M,
            MAX_BLOCKS,
        )
        if BLOCK_M_2 > 0:
            _align_rows(
                masks,
                flat_local,
                Sorted2,
                ExpertIds2,
                Padded2,
                numel,
                max_padded_2,
                TOPK,
                NUM_LOCAL,
                LOCAL_POW2,
                WORDS,
                BLOCK_M_2,
                MAX_BLOCKS_2,
            )
        # Every thread has read the masks.
        tl.debug_barrier()
        tl.store(Masks + experts * WORDS + words, 0, experts < NUM_LOCAL)
        tl.store(Counters + _ALIGN_TOKENS_DONE, 0)
        if ROUTERS > 0:
            tl.store(Counters + _ALIGN_ROUTERS_DONE, 0)


def _alignment_buffers(numel: int, num_experts: int, block_m: int, device):
    """The outputs of `moe_align_block_size` (same shapes)."""
    max_padded = numel + num_experts * (block_m - 1)
    if numel < num_experts:
        max_padded = min(numel * block_m, max_padded)
    return (
        torch.empty(max_padded, dtype=torch.int32, device=device),
        torch.empty(triton.cdiv(max_padded, block_m), dtype=torch.int32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
    )


def select_and_align(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    masks: torch.Tensor,
    counters: torch.Tensor,
    block_sizes: tuple[int, ...],
    is_padding: torch.Tensor | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """`select_experts` and, in the same launch, the Humming row layout
    `moe_align_block_size(ids, size, num_local, ignore_invalid_experts=True)`
    of each of `block_sizes` (at most two), for up to `ALIGN_MAX_TOKENS`
    tokens; up to `ALIGN_ROUTER_MAX_TOKENS` on SM80 / SM90, the router GEMM
    too. The rows of an expert are in token order. `masks` (int32
    `[num_local * ALIGN_WORDS]`) and `counters` (`ops.moe.NUM_COUNTERS`) are
    zeroed buffers that the kernel resets.
    """
    tokens = x.shape[0]
    assert 0 < tokens <= ALIGN_MAX_TOKENS and 1 <= len(block_sizes) <= 2
    num_local = masks.numel() // ALIGN_WORDS
    assert masks.dtype == torch.int32 and num_local * ALIGN_WORDS == masks.numel()
    assert is_padding is None or (
        is_padding.shape == (tokens,) and is_padding.dtype == torch.bool
    )
    experts, hidden = gate_weight.shape
    assert x.is_contiguous() and gate_weight.is_contiguous()
    device = x.device
    # The router tiles of `k_group_partials`. With them, all programs must be
    # co-resident: two per SM fit (up to 40 KiB of shared memory and 80
    # registers per thread measured on A100).
    groups = triton.cdiv(hidden, 1024)
    block_m, block_n = 16, 32 if tokens <= 16 else 64
    n_tiles, m_tiles = triton.cdiv(experts, block_n), triton.cdiv(tokens, block_m)
    routers = n_tiles * m_tiles * groups
    if not (
        tokens <= ALIGN_ROUTER_MAX_TOKENS
        and routers + tokens <= 2 * num_compute_units(device.index or 0)
        and (
            current_platform.is_device_capability(80)
            or current_platform.is_device_capability(90)
        )
    ):
        routers = 0
    if routers:
        partials = torch.empty(
            (groups, tokens, experts), dtype=torch.float32, device=device
        )
    else:
        partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=device)
    local_pow2 = triton.next_power_of_2(num_local)
    alignments = {
        size: _alignment_buffers(tokens * topk, num_local, size, device)
        for size in block_sizes
    }
    first = alignments[block_sizes[0]]
    second = alignments[block_sizes[-1]]
    sizes = (*block_sizes, 0)[:2]
    _select_align_kernel[(routers + tokens,)](
        x,
        gate_weight,
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        masks,
        counters,
        *first,
        *second,
        tokens,
        route_scale,
        first[0].numel(),
        second[0].numel(),
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        NUM_LOCAL=num_local,
        LOCAL_POW2=local_pow2,
        WORDS=ALIGN_WORDS,
        BLOCK_M=sizes[0],
        # Blocks of an expert.
        MAX_BLOCKS=triton.next_power_of_2(triton.cdiv(tokens, sizes[0])),
        BLOCK_M_2=sizes[1],
        MAX_BLOCKS_2=triton.next_power_of_2(triton.cdiv(tokens, max(sizes[1], 1))),
        ROWS=triton.next_power_of_2(tokens * topk),
        H=hidden,
        BM=block_m,
        BN=block_n,
        N_TILES=n_tiles,
        M_TILES=m_tiles,
        ROUTERS=routers,
        num_warps=4,
        # The stages of `k_group_partials`; token programs wait for the
        # router programs: launch all co-resident.
        num_stages=3,
        launch_cooperative_grid=routers > 0,
    )
    return (ids, weights, src2dst), alignments


@triton.jit
def _swiglu_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """bf16 softcapped SwiGLU of a local row times its routing weight."""
    row = tl.program_id(0)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    gamma = tl.load(Weights + row)
    values = _situ(GateUp, row, start, gamma, N, BLOCK, GATE_CAP, UP_CAP)
    tl.store(Out + row * N + start + tl.arange(0, BLOCK), values.to(tl.bfloat16))


def swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`activation.swiglu_quantize`'s SiTU of the interleaved (gate, up)
    columns times the routing weights, in bf16, for the local rows (the
    others are left unset)."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _swiglu_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=2,
    )
    return out


class HummingExperts:
    """Humming tensors and kernel configs of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`), the row-major checkpoint
            weights `[E, N, K / pack]`, block scales `[E, N, K / group]`
            and, for NVFP4, the per-expert global scales `[E]` (else None).
        shapes: `(N, K)` of each projection.
        nvfp4: NVFP4 (else MXFP8) weights.
        input_schema: Quantized input contract, or None for BF16 activations.

    """

    def __init__(
        self,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        input_schema: "HummingInputSchema | None" = None,
    ) -> None:
        from vllm.utils.humming import (
            GemmType,
            HummingInputSchema,
            ModeloptMxfp8WeightSchema,
            ModeloptNvfp4WeightSchema,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
        )

        self.configs = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list] = {}
        self.tuning_strs: dict[str, str] = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            num_experts = weight.shape[0]
            tensors = {"weight": weight, "weight_scale": scale}
            if nvfp4:
                source = ModeloptNvfp4WeightSchema()
                tensors["weight_scale_2"] = global_scale
            else:
                source = ModeloptMxfp8WeightSchema()
            schema, tensors = source.convert_humming(
                tensors=tensors,
                shape_n_stacks=[n],
                shape_k_stacks=[k],
                param_dtype=torch.bfloat16,
                num_experts=num_experts,
            )
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or HummingInputSchema(),
                weight_schema=schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                device=weight.device,
            )
            assert config.shape_n == n and config.shape_k == k, (n, k)
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gemm_type=GemmType.INDEXED,
            )
            # Stream-K adds the K slices into the output with bf16 atomics,
            # whose order (and so the result) changes from run to run.
            for _, _, kernel_config in tuning:
                kernel_config["use_stream_k"] = False
            self.configs[name] = config
            self.tensors[name] = tensors
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
        self.num_experts = num_experts
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
                "use_f16_accum": False,
                "gemm_type": GemmType.INDEXED.value,
            }
        )
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def _block_m(self, name: str, rows: int) -> int:
        for lower, upper, config in self.tunings[name]:
            if lower < rows <= upper:
                return config["block_shape"][0]
        return _DEFAULT_BLOCK_M

    def _gemm(
        self,
        name: str,
        inputs: torch.Tensor,
        rows: int,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        top_k: int,
        valid_rows: int,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        config = self.configs[name]
        tensors = self.tensors[name]
        out = torch.empty(
            (rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return humming_forward(
            config,
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            weight_scale_2=tensors.get("weight_scale_2"),
            outputs=out,
            locks=self.locks,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )

    def block_sizes(self, rows: int, num_global_experts: int) -> dict[str, int]:
        """The row block size of each projection's kernel for `rows` (token,
        slot) rows over `num_global_experts` experts."""
        # Expected local rows, which pick the kernel configs.
        valid_rows = -(-rows * self.num_experts // num_global_experts)
        return {name: self._block_m(name, valid_rows) for name in ("w13", "w2")}

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
        alignments: dict[int, tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset.
        `alignments`: the row layout of each `block_sizes` size
        (`select_and_align`), else computed here."""
        tokens, topk = ids.shape
        rows = tokens * topk
        valid_rows = -(-rows * self.num_experts // num_global_experts)
        block_m = self.block_sizes(rows, num_global_experts)
        if alignments is None:
            alignments = {
                size: moe_align_block_size(
                    ids, size, self.num_experts, ignore_invalid_experts=True
                )[:3]
                for size in set(block_m.values())
            }
        gate_up = self._gemm(
            "w13", x, rows, alignments[block_m["w13"]], topk, valid_rows
        )
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        return self._gemm(
            "w2", activated, rows, alignments[block_m["w2"]], 1, valid_rows
        )
