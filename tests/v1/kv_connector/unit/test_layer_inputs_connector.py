# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The served model writing a prefix's KV cache from its per-layer inputs
(`KVFromLayerInputsWriter`) gives the KV cache vLLM computes for that prefix.

The inputs are captured from a normal prefill of the same engine, so a request
whose prefix KV is written from them must give the same greedy tokens and
logprobs, bit for bit, as a prefix-cache hit on that prefill. Batch-invariant
mode makes the K/V GEMMs independent of how many tokens each pass holds.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

MODELS = ["Qwen/Qwen3-0.6B"]
PREFIX_TOKENS = 512


@dataclass
class _Load:
    path: str
    start: int
    end: int
    block_ids: tuple[list[int], ...]


@dataclass
class _Metadata(KVConnectorMetadata):
    loads: list[_Load] = field(default_factory=list)


class LayerInputsTestConnector(KVConnectorBase_V1):
    """Takes a request's first `num_tokens` prompt tokens from the per-layer
    inputs in `kv_transfer_params["layer_inputs_path"]` (safetensors,
    `layer_inputs` `[num_layers, num_tokens, hidden_size]`)."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._pending: dict[str, tuple[str, int, int]] = {}
        self._writer = None

    def _num_tokens(self, request) -> int:
        params = request.kv_transfer_params or {}
        if "layer_inputs_path" not in params:
            return 0
        return min(int(params["num_tokens"]), request.num_prompt_tokens - 1)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return max(0, self._num_tokens(request) - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            end = self._num_tokens(request)
            path = request.kv_transfer_params["layer_inputs_path"]
            self._pending[request.request_id] = (path, end - num_external_tokens, end)

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = _Metadata()
        for req in scheduler_output.scheduled_new_reqs:
            if (pending := self._pending.pop(req.req_id, None)) is not None:
                block_ids = tuple(list(ids) for ids in req.block_ids)
                meta.loads.append(_Load(*pending, block_ids))
        return meta

    def request_finished(self, request, block_ids) -> tuple[bool, Any]:
        self._pending.pop(request.request_id, None)
        return False, None

    def register_kv_from_layer_inputs_writer(self, writer) -> None:
        self._writer = writer

    def start_load_kv(self, forward_context, **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, _Metadata)
        for load in metadata.loads:
            assert self._writer is not None
            device = str(self._writer.device)
            inputs = load_file(load.path, device=device)["layer_inputs"]
            chunk = inputs[:, load.start : load.end]
            self._writer.write(list(chunk.unbind(0)), load.start, load.block_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self) -> None:
        return


def _capture_layer_inputs(model, captured: dict, num_tokens: int):
    """Record the input of every decoder layer for positions < num_tokens."""

    def hook(idx):
        def record(module, args):
            if not captured["active"]:
                return
            positions, hidden, residual = args[:3]
            x = hidden if residual is None else hidden + residual
            keep = positions < num_tokens
            buf = captured.setdefault(
                idx, torch.zeros(num_tokens, x.shape[-1], dtype=x.dtype)
            )
            buf[positions[keep].cpu()] = x[keep].cpu()

        return record

    for idx, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(hook(idx))


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="validated with batch invariance on NVIDIA SM80+ only",
)
@pytest.mark.parametrize("model", MODELS)
def test_written_prefix_matches_prefix_cache_hit(model, tmp_path, monkeypatch):
    # The capture hooks share Python state with this process.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    llm = LLM(
        model=model,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        kv_transfer_config=KVTransferConfig(
            kv_connector="LayerInputsTestConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
        ),
    )
    vocab = llm.get_tokenizer().vocab_size
    prefix = [(i * 7919) % (vocab - 1000) + 500 for i in range(PREFIX_TOKENS)]
    question = [(i * 104729) % (vocab - 1000) + 500 for i in range(24)]

    captured: dict = {"active": True}
    llm.apply_model(lambda m: _capture_layer_inputs(m, captured, PREFIX_TOKENS))
    llm.generate(
        [TokensPrompt(prompt_token_ids=prefix + question[:1])],
        SamplingParams(temperature=0.0, max_tokens=1),
        use_tqdm=False,
    )
    captured["active"] = False
    layers = sorted(k for k in captured if k != "active")
    path = tmp_path / "layer_inputs.safetensors"
    save_file({"layer_inputs": torch.stack([captured[i] for i in layers])}, str(path))

    greedy = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    hit = llm.generate(
        [TokensPrompt(prompt_token_ids=prefix + question)], greedy, use_tqdm=False
    )[0]
    assert hit.num_cached_tokens == PREFIX_TOKENS

    # A new cache salt: no prefix-cache hit, the connector supplies the prefix.
    written = llm.generate(
        [TokensPrompt(prompt_token_ids=prefix + question, cache_salt="written")],
        SamplingParams(
            temperature=0.0,
            max_tokens=16,
            logprobs=5,
            extra_args={
                "kv_transfer_params": {
                    "layer_inputs_path": str(path),
                    "num_tokens": PREFIX_TOKENS,
                }
            },
        ),
        use_tqdm=False,
    )[0]
    assert written.num_cached_tokens == PREFIX_TOKENS

    assert written.outputs[0].token_ids == hit.outputs[0].token_ids
    got, want = written.outputs[0].logprobs, hit.outputs[0].logprobs
    assert got is not None and want is not None
    assert [{k: v.logprob for k, v in step.items()} for step in got] == [
        {k: v.logprob for k, v in step.items()} for step in want
    ]
