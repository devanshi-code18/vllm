# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a decoder's KV cache from per-layer inputs, skipping attention outputs
and MLPs. See `SupportsKVFromLayerInputs`."""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
from torch import nn


class KVWriteTarget(NamedTuple):
    """Where one attention layer writes: the layer inputs' tokens `rows`
    (`None`: all of them), at `positions`, into KV cache `slot_mapping`."""

    rows: torch.Tensor | None
    positions: torch.Tensor
    slot_mapping: torch.Tensor


def norm_layer_input(
    norm: nn.Module, layer_input: torch.Tensor, first_layer: bool
) -> torch.Tensor:
    """Apply a decoder layer's input norm the way its forward does, so the
    result matches a normal pass bit for bit: the first layer normalizes its
    input alone, the others through the fused residual add, here with a zero
    hidden state (`0 + x` is exact).

    A float32 `layer_input` is the unrounded residual sum, which fused norms
    that add in float32 (the native implementation, e.g. Gemma's) read; a
    model-dtype one is the rounded sum, which the CUDA kernel reads."""
    dtype = norm.weight.dtype
    if first_layer:
        return norm(layer_input.to(dtype))
    zeros = torch.zeros_like(layer_input, dtype=dtype)
    normed, _ = norm(zeros, layer_input.clone())
    return normed


def write_decoder_kv(
    layers: nn.ModuleList,
    start_layer: int,
    end_layer: int,
    layer_inputs: Sequence[torch.Tensor],
    targets: Mapping[str, KVWriteTarget],
) -> None:
    """For decoder layers with `input_layernorm` and a `self_attn` that has
    `kv_from_normed_input(positions, x) -> (k, v)` and an `attn` layer."""
    for idx in range(start_layer, end_layer):
        layer = layers[idx]
        attn = layer.self_attn
        target = targets.get(attn.attn.layer_name)
        if target is None or target.slot_mapping.numel() == 0:
            continue
        x = layer_inputs[idx]
        if target.rows is not None:
            x = x[target.rows]
        normed = norm_layer_input(layer.input_layernorm, x, idx == 0)
        k, v = attn.kv_from_normed_input(target.positions, normed)
        attn.attn.write_kv(k, v, target.slot_mapping)
