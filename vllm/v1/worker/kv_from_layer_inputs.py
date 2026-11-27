# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side writer of a request's KV cache from per-layer inputs.

A KV connector that holds per-layer inputs for a request's prompt (e.g. hidden
states translated from another model) receives this writer through
`KVConnectorBase_V1.register_kv_from_layer_inputs_writer` and calls `write`
from `start_load_kv`. The served model turns the inputs into K/V with its own
weights (its tensor-parallel shard) and writes them with its own KV-cache
dtype and scales.
"""

from collections.abc import Sequence

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.kv_from_layer_inputs import KVWriteTarget
from vllm.model_executor.models.interfaces import (
    SupportsKVFromLayerInputs,
    supports_kv_from_layer_inputs,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MLAAttentionSpec,
)

logger = init_logger(__name__)


def create_kv_from_layer_inputs_writer(
    model: object,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    device: torch.device,
) -> "KVFromLayerInputsWriter | None":
    """The worker's writer, or None when the served model cannot write its KV
    cache from layer inputs in this configuration (the reason is logged)."""
    if not supports_kv_from_layer_inputs(model):
        return None
    reason = _unsupported_reason(model, vllm_config, kv_cache_config)
    if reason is not None:
        logger.info_once(
            "KV cache writes from per-layer inputs are unavailable: %s.", reason
        )
        return None
    return KVFromLayerInputsWriter(model, vllm_config, kv_cache_config, device)


class KVFromLayerInputsWriter:
    def __init__(
        self,
        model: SupportsKVFromLayerInputs,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.input_dtype = (
            model.kv_from_layer_inputs_dtype or vllm_config.model_config.dtype
        )
        self.block_sizes: list[int] = []
        self.layer_groups: dict[str, int] = {}
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            self.block_sizes.append(group.block_size)
            for name, spec in group.kv_cache_specs.items():
                if isinstance(spec, AttentionSpec):
                    self.layer_groups[name] = group_id

    @torch.inference_mode()
    def write(
        self,
        layer_inputs: Sequence[torch.Tensor],
        start: int,
        block_ids: Sequence[Sequence[int]],
    ) -> None:
        """Write K/V for prompt tokens `[start, start + n)`.

        Args:
            layer_inputs: `layer_inputs[i]` is the `[n, hidden_size]` input of
                decoder layer `i` for these tokens (layer 0: the scaled input
                embeddings), on this worker's device, in `self.input_dtype`.
            start: Index of the first token in the request.
            block_ids: The request's block ids per KV cache group, as the
                scheduler allocated them (connector metadata). Tokens in the
                null block (sliding-window layers past their window) are
                skipped.

        Raises:
            ValueError: `layer_inputs` are not in `self.input_dtype`.

        """
        dtypes = {x.dtype for x in layer_inputs}
        if dtypes != {self.input_dtype}:
            raise ValueError(
                f"{type(self.model).__name__} reproduces its forward only from "
                f"{self.input_dtype} layer inputs, got {sorted(map(str, dtypes))}."
            )
        num_tokens = layer_inputs[0].shape[0]
        positions = np.arange(start, start + num_tokens)
        # Token selection and slots for every group go to the GPU in one copy.
        host: list[np.ndarray] = []
        layout: dict[int, tuple[bool, int]] = {}
        for group_id in sorted(set(self.layer_groups.values())):
            block_size = self.block_sizes[group_id]
            blocks = np.asarray(block_ids[group_id], dtype=np.int64)
            blocks = blocks[positions // block_size]
            slots = blocks * block_size + positions % block_size
            rows = np.flatnonzero(blocks != 0)
            selected = len(rows) < num_tokens
            if selected:
                host.append(rows)
            host.append(positions[rows] if selected else positions)
            host.append(slots[rows] if selected else slots)
            layout[group_id] = (selected, len(rows))
        flat = torch.from_numpy(np.concatenate(host)).pin_memory()
        flat = flat.to(self.device, non_blocking=True)
        group_targets: dict[int, KVWriteTarget] = {}
        offset = 0
        for group_id, (selected, count) in layout.items():
            parts = []
            for _ in range(3 if selected else 2):
                parts.append(flat[offset : offset + count])
                offset += count
            rows_t = parts.pop(0) if selected else None
            group_targets[group_id] = KVWriteTarget(rows_t, parts[0], parts[1])
        targets = {
            name: group_targets[group_id]
            for name, group_id in self.layer_groups.items()
        }
        self.model.write_kv_from_layer_inputs(layer_inputs, targets)


def _unsupported_reason(
    model: SupportsKVFromLayerInputs,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> str | None:
    if vllm_config.lora_config is not None:
        # K/V would miss the adapters' deltas.
        return "LoRA"
    parallel = vllm_config.parallel_config
    if (
        parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
    ):
        return "context parallelism"
    specs = [
        spec
        for group in kv_cache_config.kv_cache_groups
        for spec in group.kv_cache_specs.values()
    ]
    if any(isinstance(spec, MLAAttentionSpec) for spec in specs):
        return "MLA"
    assert isinstance(model, torch.nn.Module)
    owned = {m.layer_name for m in model.modules() if isinstance(m, Attention)}
    foreign = [
        name
        for group in kv_cache_config.kv_cache_groups
        for name, spec in group.kv_cache_specs.items()
        if isinstance(spec, AttentionSpec) and name not in owned
    ]
    if foreign:
        return f"KV cache layers the model does not own ({foreign[0]}, ...)"
    return model.kv_from_layer_inputs_unsupported_reason()
