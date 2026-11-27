# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`SupportsKVFromLayerInputs` for the Transformers modeling backend.

Each decoder layer runs its own forward on its given input up to the attention
interface call, where `vllm_kv_only` writes K/V and stops the layer: whatever
the model does before attention (norms, Q/K/V projections, Q/K norms, RoPE)
runs as in a normal pass, and attention outputs, the output projection and the
MLP are skipped."""

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import nn

from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    override_forward_context,
    set_forward_context,
)
from vllm.model_executor.layers.kv_from_layer_inputs import KVWriteTarget
from vllm.model_executor.models.transformers.fusers.attention import VLLM_ATTN_IMPL
from vllm.model_executor.models.utils import PPMissingLayer

if TYPE_CHECKING:
    from vllm.model_executor.models.transformers.base import Base as _Base
else:
    _Base = object

VLLM_KV_ONLY_ATTN_IMPL = "vllm_kv_only"
# HF architectures whose decoder layers look up the attention interface at
# call time (so the K/V-only one takes effect) and were checked bit for bit.
VALIDATED_MODEL_TYPES = frozenset({"llama"})
KV_ONLY_SLOT_MAPPING_ATTR = "_vllm_kv_only_slot_mapping"


class _KVWritten(Exception):
    """Raised by `vllm_kv_only_attention_forward` to stop the decoder layer."""


def vllm_kv_only_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
):
    from vllm.model_executor.models.transformers.base import VLLM_ATTN_ATTR

    num_tokens = key.shape[-2]
    key, value = (x.transpose(1, 2).reshape(num_tokens, -1) for x in (key, value))
    attn = getattr(module, VLLM_ATTN_ATTR)
    attn.write_kv(key, value, getattr(module, KV_ONLY_SLOT_MAPPING_ATTR))
    raise _KVWritten


@contextmanager
def _eager(vllm_config) -> Iterator[None]:
    """Run the Transformers model eagerly (it is the compiled module)."""
    if is_forward_context_available():
        context = dataclasses.replace(get_forward_context(), skip_compiled=True)
        with override_forward_context(context):
            yield
    else:
        with set_forward_context(None, vllm_config, skip_compiled=True):
            yield


class KVFromLayerInputsMixin(_Base):
    """`SupportsKVFromLayerInputs` for decoder models of this backend."""

    supports_kv_from_layer_inputs: ClassVar[Literal[True]] = True
    kv_from_layer_inputs_dtype: ClassVar[torch.dtype | None] = None

    def _decoder_layer_args(
        self, layers: nn.ModuleList, x: torch.Tensor, positions: torch.Tensor
    ) -> dict[int, tuple[tuple, dict[str, Any]]]:
        """The arguments the model passes each decoder layer (rotary embeddings,
        masks, ...) for `positions`, captured by running the model with every
        decoder layer stubbed out."""
        captured: dict[int, tuple[tuple, dict[str, Any]]] = {}

        def stub(idx: int):
            def forward(*args, **kwargs):
                hidden_states = args[0] if args else kwargs.pop("hidden_states")
                captured[idx] = (args[1:], kwargs)
                return hidden_states

            return forward

        indices = [
            idx
            for idx in range(self.start_layer, self.end_layer)
            if not isinstance(layers[idx], PPMissingLayer)
        ]
        # Instance-level forwards (e.g. accelerate hooks) are put back after.
        saved = {idx: vars(layers[idx]).get("forward") for idx in indices}
        for idx in indices:
            layers[idx].forward = stub(idx)
        try:
            self.model(
                inputs_embeds=x[None].to(self.model.dtype),
                position_ids=positions[None],
                use_cache=False,
                return_dict=False,
            )
        finally:
            for idx in indices:
                if saved[idx] is None:
                    del layers[idx].forward
                else:
                    layers[idx].forward = saved[idx]
        return captured

    def kv_from_layer_inputs_unsupported_reason(self) -> str | None:
        model_type = self.text_config.model_type
        if model_type not in VALIDATED_MODEL_TYPES:
            return (
                f"the Transformers backend serving {model_type!r} (validated: "
                f"{', '.join(sorted(VALIDATED_MODEL_TYPES))})"
            )
        if (
            self._decoder_layers is None
            or self.text_config._attn_implementation != VLLM_ATTN_IMPL
        ):
            return "decoder layers that do not use vLLM's attention"
        return None

    def write_kv_from_layer_inputs(
        self,
        layer_inputs: Sequence[torch.Tensor],
        targets: Mapping[str, KVWriteTarget],
    ) -> None:
        from vllm.model_executor.models.transformers.base import VLLM_ATTN_ATTR

        layers = self._decoder_layers
        assert layers is not None
        text_config = self.text_config
        args_by_target: dict[int, dict[int, tuple[tuple, dict[str, Any]]]] = {}
        text_config._attn_implementation = VLLM_KV_ONLY_ATTN_IMPL
        try:
            with _eager(self.vllm_config):
                for idx in range(self.start_layer, self.end_layer):
                    attn_module = self.get_submodule(self.attention_fusers[idx][0])
                    target = targets.get(
                        getattr(attn_module, VLLM_ATTN_ATTR).layer_name
                    )
                    if target is None or target.slot_mapping.numel() == 0:
                        continue
                    x = layer_inputs[idx]
                    if target.rows is not None:
                        x = x[target.rows]
                    if id(target) not in args_by_target:
                        args_by_target[id(target)] = self._decoder_layer_args(
                            layers, x, target.positions
                        )
                    args, kwargs = args_by_target[id(target)][idx]
                    setattr(attn_module, KV_ONLY_SLOT_MAPPING_ATTR, target.slot_mapping)
                    hidden_states = x[None].to(self.model.dtype)
                    try:
                        layers[idx](hidden_states, *args, **kwargs)
                    except _KVWritten:
                        continue
                    finally:
                        delattr(attn_module, KV_ONLY_SLOT_MAPPING_ATTR)
                    raise RuntimeError(
                        f"Decoder layer {idx} returned without calling attention."
                    )
        finally:
            text_config._attn_implementation = VLLM_ATTN_IMPL
