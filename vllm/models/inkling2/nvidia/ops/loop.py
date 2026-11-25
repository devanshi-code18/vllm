# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loop message controllers, batched over streams.

A controller blends a fp32 message into a fp32 sublayer input `dest`: from
`h = down(rms([dest, message]))` (bf16), `out = (1 - tanh(forget(h))) * dest +
tanh(gain(h)) * message` in fp32, unfused as the reference. The output feeds
the sublayer's RMSNorm, applied in the same kernel.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import current_stream

from .norm import rsqrt


@triton.jit
def _pair_rms_norm_kernel(
    Dest,
    Message,
    Out,
    stride_dest_s,
    T,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + (s * T + t) * D + cols, mask, other=0)
    dest_rstd = rsqrt(tl.div_rn(tl.sum(dest * dest), D) + eps)
    message_rstd = rsqrt(tl.div_rn(tl.sum(message * message), D) + eps)
    out = Out + (s * T + t) * (2 * D)
    tl.store(out + cols, (dest * dest_rstd).to(Out.dtype.element_ty), mask)
    tl.store(out + D + cols, (message * message_rstd).to(Out.dtype.element_ty), mask)


def pair_rms_norm(dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """bf16 `(S, T, 2D)` `[rms(dest), rms(message)]` of fp32 `(S, T, D)`
    inputs, weightless; `dest` may broadcast over streams (stride 0)."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    out = message.new_empty((num_streams, num_tokens, 2 * d), dtype=torch.bfloat16)
    if num_tokens:
        _pair_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            out,
            dest.stride(0),
            num_tokens,
            1e-6,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
    return out


@triton.jit
def _blend_rms_norm_kernel(
    Dest,
    Message,
    GainForget,
    NormW,
    Out,
    stride_dest_s,
    stride_w_s,
    T,
    norm_eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    row = s * T + t
    dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask, other=0)
    message = tl.load(Message + row * D + cols, mask, other=0)
    gain = tldevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask, other=0))
    forget = tldevice.tanh(
        tl.load(GainForget + row * (2 * D) + D + cols, mask, other=0)
    )
    x = (1.0 - forget) * dest + gain * message
    rstd = rsqrt(tl.div_rn(tl.sum(x * x), D) + norm_eps)
    y = x * rstd
    y = y * tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * D + cols, y.to(Out.dtype.element_ty), mask)


def blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """bf16 `(S, T, D)` RMSNorm (row `s` of `norm_weight` `(S, D)`) of
    `(1 - tanh(forget)) * dest + tanh(gain) * message`, with fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, and fp32 `(S, T, 2D)`
    `gain_forget` = `[gain, forget]`. `norm_weight` may broadcast too."""
    num_streams, num_tokens, d = message.shape
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert gain_forget.shape == (num_streams, num_tokens, 2 * d)
    assert gain_forget.is_contiguous() and gain_forget.dtype == torch.float32
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    if num_tokens:
        _blend_rms_norm_kernel[(num_streams, num_tokens)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            norm_weight.stride(0),
            num_tokens,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
            # The blend is unfused (no FMA), as the reference.
            enable_fp_fusion=False,
        )
    return out


# Small batches: the down projection as a split-K Triton GEMM (cuBLAS runs
# one CTA per stream at low M, and its split-K reduces in bf16 for some
# shapes), the up projection fused into the blend, then the RMSNorm.
# Reductions are summed in a fixed order (the down projection's 1024-wide K
# groups, as the reference `linear_k_groups`), so results are deterministic
# and do not depend on the other rows of the batch.
FUSED_MAX_TOKENS = 256
_BT = 16
_K_GROUP = 1024
_BK = 128
_MAX_TILES = 4096


@triton.jit
def _controller_down_kernel(
    Pair,
    Down,
    Partials,
    Hidden,
    Counter,
    T,
    stride_down_s,
    K: tl.constexpr,
    R: tl.constexpr,
    GROUP: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    RB: tl.constexpr,
):
    """Program `(s, g, b * R // RB + n)`: K group `g` (`GROUP` wide, fp32
    accumulation) of `pair[s] @ down[s].T` for row block `b` and output
    columns `n`. The last program of the tile to finish sums the groups in
    order into bf16 `Hidden` `(S, T, R)`, as the reference `linear_k_groups`."""
    GROUPS: tl.constexpr = (K + GROUP - 1) // GROUP
    SPLITS: tl.constexpr = R // RB
    s = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    b = tl.program_id(2) // SPLITS
    n = tl.program_id(2) % SPLITS
    rows = b * BT + tl.arange(0, BT)
    rmask = rows < T
    r = n * RB + tl.arange(0, RB)
    kk = tl.arange(0, BK)
    x_ptrs = Pair + (s * T + rows[:, None]) * K + kk[None, :]
    w_ptrs = Down + s * stride_down_s + r[:, None] * K + kk[None, :]
    acc = tl.zeros((BT, RB), tl.float32)
    k_end = tl.minimum(g * GROUP + GROUP, K)
    for k in tl.range(g * GROUP, k_end, BK, num_stages=3):
        w = tl.load(w_ptrs + k)
        x = tl.load(x_ptrs + k, rmask[:, None], other=0)
        acc = tl.dot(x, tl.trans(w), acc)
    offs = rows[:, None] * R + r[None, :]
    tl.store(Partials + ((s * GROUPS + g) * T) * R + offs, acc, rmask[:, None])
    tl.debug_barrier()
    tile = s * tl.num_programs(2) + tl.program_id(2)
    if tl.atomic_add(Counter + tile, 1, sem="acq_rel", scope="gpu") == GROUPS - 1:
        total = tl.load(
            Partials + (s * GROUPS * T) * R + offs,
            rmask[:, None],
            cache_modifier=".cg",
        )
        for i in tl.static_range(1, GROUPS):
            total += tl.load(
                Partials + ((s * GROUPS + i) * T) * R + offs,
                rmask[:, None],
                cache_modifier=".cg",
            )
        tl.store(Hidden + (s * T) * R + offs, total.to(tl.bfloat16), rmask[:, None])
        tl.atomic_xchg(Counter + tile, 0, sem="relaxed", scope="gpu")


@triton.jit
def _controller_blend_kernel(
    Hidden,
    GainForget,
    Dest,
    Message,
    X,
    SumSq,
    T,
    stride_gf_s,
    stride_dest_s,
    D: tl.constexpr,
    R: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Program `(s, b, c)`: column chunk `c` of the fp32 blend
    `(1 - tanh(forget)) * dest + tanh(gain) * message` (unfused, as the
    reference) for row block `b`, and the chunk's row sums of squares."""
    s = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    c = tl.program_id(2)
    rows = b * BT + tl.arange(0, BT)
    rmask = rows < T
    r = tl.arange(0, R)
    h_ptrs = Hidden + (s * T + rows[:, None]) * R + r[None, :]
    h = tl.load(h_ptrs, rmask[:, None], other=0)
    cols = c * BC + tl.arange(0, BC)
    cmask = cols < D
    gf = GainForget + s * stride_gf_s + cols[:, None] * R + r[None, :]
    wg = tl.load(gf, cmask[:, None], other=0)
    wf = tl.load(gf + D * R, cmask[:, None], other=0)
    gain = tldevice.tanh(tl.dot(h, tl.trans(wg)))
    forget = tldevice.tanh(tl.dot(h, tl.trans(wf)))
    m = rmask[:, None] & cmask[None, :]
    dest = tl.load(
        Dest + s * stride_dest_s + rows[:, None].to(tl.int64) * D + cols[None, :],
        m,
        other=0,
    )
    msg = tl.load(Message + (s * T + rows[:, None]) * D + cols[None, :], m, other=0)
    x = (1.0 - forget) * dest + gain * msg
    tl.store(X + (s * T + rows[:, None]) * D + cols[None, :], x, m)
    tl.store(
        SumSq + (s * tl.num_programs(2) + c) * T + rows, tl.sum(x * x, axis=1), rmask
    )


@triton.jit
def _controller_norm_kernel(
    X,
    SumSq,
    NormW,
    Out,
    T,
    stride_w_s,
    norm_eps,
    D: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row `(s, t)`: the RMSNorm of the blend, its chunk sums of squares
    added in order."""
    s = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    ss = tl.load(SumSq + s * CHUNKS * T + t)
    for i in tl.static_range(1, CHUNKS):
        ss += tl.load(SumSq + (s * CHUNKS + i) * T + t)
    rstd = rsqrt(tl.div_rn(ss, D) + norm_eps)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + (s * T + t) * D + cols, mask, other=0)
    w = tl.load(NormW + s * stride_w_s + cols, mask, other=0).to(tl.float32)
    y = x * rstd * w
    tl.store(Out + (s * T + t) * D + cols, y.to(Out.dtype.element_ty), mask)


@functools.cache
def _counter(stream: int) -> torch.Tensor:
    """Row-block arrival counters for one stream; each launch leaves them
    zeroed."""
    return torch.zeros(_MAX_TILES, dtype=torch.int32, device="cuda")


def use_fused_controller(message: torch.Tensor) -> bool:
    """Small batches off SM12x (which runs the controllers on its split-K
    `ll_gemm`)."""
    return (
        0 < message.shape[1] <= FUSED_MAX_TOKENS
        and message.shape[2] % _BK == 0
        and not current_platform.is_device_capability_family(120)
    )


def fused_controller(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """The stream-batched controllers and their RMSNorm: fp32 `(S, T, D)`
    `dest` (may broadcast over streams) and `message`, bf16 `down`
    `(S, R, 2D)` and `gain_forget` `(S, 2D, R)`, `norm_weight` `(S, D)` (may
    broadcast) -> bf16 `(S, T, D)`. The down projection sums 1024-wide K
    groups in order, as the reference `linear_k_groups`."""
    num_streams, num_tokens, d = message.shape
    rank = down.shape[1]
    assert dest.shape == message.shape and message.is_contiguous()
    assert dest.stride(1) == d and dest.stride(2) == 1
    assert dest.dtype == message.dtype == torch.float32
    assert down.shape == (num_streams, rank, 2 * d)
    assert down.stride(1) == 2 * d and down.stride(2) == 1
    assert gain_forget.shape == (num_streams, 2 * d, rank)
    assert gain_forget.stride(1) == rank and gain_forget.stride(2) == 1
    assert norm_weight.shape == (num_streams, d) and norm_weight.stride(1) == 1
    assert (2 * d) % _BK == 0 and _K_GROUP % _BK == 0
    # Smaller batches spread the down projection's columns and the blend's
    # chunks over more programs (tuned on A100).
    block_r = min(16 if num_tokens <= 16 else 32, rank)
    block_c = 64 if num_tokens <= 64 else 128
    groups = triton.cdiv(2 * d, _K_GROUP)
    splits = rank // block_r
    chunks = triton.cdiv(d, block_c)
    blocks = triton.cdiv(num_tokens, _BT)
    assert num_streams * blocks * splits <= _MAX_TILES
    device = message.device
    partials = torch.empty(
        (num_streams, groups, num_tokens, rank), dtype=torch.float32, device=device
    )
    hidden = torch.empty(
        (num_streams, num_tokens, rank), dtype=torch.bfloat16, device=device
    )
    _controller_down_kernel[(num_streams, groups, blocks * splits)](
        pair_rms_norm(dest, message),
        down,
        partials,
        hidden,
        _counter(current_stream().cuda_stream),
        num_tokens,
        down.stride(0),
        K=2 * d,
        R=rank,
        GROUP=_K_GROUP,
        BT=_BT,
        BK=_BK,
        RB=block_r,
        num_warps=4,
    )
    x = torch.empty_like(message)
    sumsq = torch.empty(
        (num_streams, chunks, num_tokens), dtype=torch.float32, device=device
    )
    _controller_blend_kernel[(num_streams, blocks, chunks)](
        hidden,
        gain_forget,
        dest,
        message,
        x,
        sumsq,
        num_tokens,
        gain_forget.stride(0),
        dest.stride(0),
        D=d,
        R=rank,
        BT=_BT,
        BC=block_c,
        num_warps=8,
        enable_fp_fusion=False,
    )
    out = message.new_empty((num_streams, num_tokens, d), dtype=torch.bfloat16)
    _controller_norm_kernel[(num_streams, num_tokens)](
        x,
        sumsq,
        norm_weight,
        out,
        num_tokens,
        norm_weight.stride(0),
        norm_eps,
        D=d,
        CHUNKS=chunks,
        BLOCK=triton.next_power_of_2(d),
        num_warps=8,
    )
    return out
