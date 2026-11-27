# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A sync KV connector loads a hybrid model's prompt prefix: attention KV plus
the Mamba/GDN state at its end, with prefix caching on."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.inputs import TokensPrompt
from vllm.v1.kv_cache_interface import MambaSpec

MODEL = "Qwen/Qwen3.5-0.8B"
# (prefix_match_unit, prefix length): the load ends on a 576-token Mamba block
# boundary, or inside a block with 64-token prefix matching.
CASES = {"block_end": (None, 1152), "inside_block": (64, 1216)}
# Captured prompt caches by name, and the requests that loaded one.
STORE: dict[int, dict] = {}
LOADS: list[str] = []


@dataclass
class _Metadata(KVConnectorMetadata):
    loads: list = field(default_factory=list)
    captures: list = field(default_factory=list)


class StateLoadConnector(KVConnectorBase_V1, SupportsHMA):
    """`{"capture": name}` copies a request's prompt cache into `STORE` after
    a one-step prefill; `{"load": name}` loads it back as the first tokens."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.prefix_caching = vllm_config.cache_config.enable_prefix_caching
        self.pending: dict[str, tuple] = {}
        self.group_of: dict[str, int] = {}
        self.block_sizes = [g.block_size for g in kv_cache_config.kv_cache_groups]
        self.num_blocks = kv_cache_config.num_blocks
        self.mamba_layers: set[str] = set()
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for name, spec in group.kv_cache_specs.items():
                self.group_of[name] = group_id
                if isinstance(spec, MambaSpec):
                    self.mamba_layers.add(name)

    @property
    def loads_mamba_state_at_load_end(self) -> bool:
        return True

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if "load" not in params:
            return 0, False
        return max(0, len(STORE[params["load"]]["tokens"]) - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params or {}
        if num_external_tokens > 0:
            LOADS.append(request.request_id)
            self.pending[request.request_id] = ("load", params["load"])
        elif "capture" in params:
            self.pending[request.request_id] = ("capture", params["capture"])

    def build_connector_meta(self, scheduler_output):
        meta = _Metadata()
        for req in scheduler_output.scheduled_new_reqs:
            if (op := self.pending.pop(req.req_id, None)) is not None:
                kind, name = op
                getattr(meta, kind + "s").append(
                    (name, req.prompt_token_ids, [list(ids) for ids in req.block_ids])
                )
        return meta

    def request_finished_all_groups(self, request, block_ids):
        return False, None

    def register_kv_caches(self, kv_caches):
        # Attention: [kernel blocks, heads, kernel block size, K|V]; a backend
        # may split each block into smaller kernel blocks (e.g. on Blackwell).
        # Mamba: [blocks, 1, 1, page].
        self.kv_caches = kv_caches

    def _slots(self, name, block_ids, num_tokens):
        block_size = self.block_sizes[self.group_of[name]]
        kernel_block_size = self.kv_caches[name].shape[2]
        # Unpacked layout: each block's kernel blocks are adjacent.
        assert self.kv_caches[name].shape[0] == self.num_blocks * (
            block_size // kernel_block_size
        )
        positions = torch.arange(num_tokens)
        blocks = torch.tensor(block_ids[self.group_of[name]])[positions // block_size]
        kernel_blocks = (
            blocks * (block_size // kernel_block_size)
            + positions % block_size // kernel_block_size
        )
        return kernel_blocks, positions % kernel_block_size

    def _state_block(self, name, block_ids, num_tokens):
        column = (num_tokens - 1) // self.block_sizes[self.group_of[name]]
        return block_ids[self.group_of[name]][column if self.prefix_caching else 0]

    def _metadata(self) -> _Metadata:
        meta = self._get_connector_metadata()
        assert isinstance(meta, _Metadata)
        return meta

    def start_load_kv(self, forward_context, **kwargs):
        for name, _, block_ids in self._metadata().loads:
            stored = STORE[name]
            num_tokens = len(stored["tokens"])
            for layer, cache in self.kv_caches.items():
                if layer in self.mamba_layers:
                    block = self._state_block(layer, block_ids, num_tokens)
                    cache[block].copy_(stored[layer])
                else:
                    blocks, offsets = self._slots(layer, block_ids, num_tokens)
                    cache[blocks, :, offsets] = stored[layer].to(cache.device)

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        for name, tokens, block_ids in self._metadata().captures:
            stored: dict = {"tokens": list(tokens)}
            for layer, cache in self.kv_caches.items():
                if layer in self.mamba_layers:
                    block = self._state_block(layer, block_ids, len(tokens))
                    stored[layer] = cache[block].cpu()
                else:
                    blocks, offsets = self._slots(layer, block_ids, len(tokens))
                    stored[layer] = cache[blocks, :, offsets].cpu()
            STORE[name] = stored


def _llm(enable_prefix_caching: bool, **kwargs) -> LLM:
    return LLM(
        model=MODEL,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        **kwargs,
        kv_transfer_config=KVTransferConfig(
            kv_connector="StateLoadConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
        ),
    )


def _generate(llm, token_ids, salt, kv_transfer_params=None, max_tokens=16):
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=5,
        extra_args={"kv_transfer_params": kv_transfer_params},
    )
    [output] = llm.generate(
        TokensPrompt(prompt_token_ids=token_ids, cache_salt=salt), params
    )
    completion = output.outputs[0]
    return completion.token_ids, completion.text, completion.logprobs


@pytest.fixture(scope="module")
def reference():
    """Capture each prefix's cache the way an external producer would (one
    prefill, no prefix caching), and the outputs of requests that load it
    there."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        llm = _llm(enable_prefix_caching=False)
        text = (Path(__file__).parents[3] / "prompts" / "summary.txt").read_text()
        token_ids = llm.get_tokenizer()(text).input_ids
        outputs = {}
        for _, prefix_len in CASES.values():
            prefix = token_ids[:prefix_len]
            _generate(llm, prefix, None, {"capture": prefix_len}, max_tokens=1)
            outputs[prefix_len] = [
                _generate(llm, prefix + q, None, {"load": prefix_len})
                for q in _questions(token_ids, prefix_len)
            ]
        del llm
        cleanup_dist_env_and_memory()
    return token_ids, outputs


def _questions(token_ids: list[int], prefix_len: int) -> list[list[int]]:
    return [token_ids[prefix_len : prefix_len + 40], token_ids[-40:]]


@pytest.mark.parametrize(
    "prefix_match_unit,prefix_len", list(CASES.values()), ids=list(CASES)
)
def test_loaded_mamba_state_with_prefix_caching(
    monkeypatch: pytest.MonkeyPatch,
    reference: tuple[list[int], dict],
    prefix_match_unit: int | None,
    prefix_len: int,
):
    """With prefix caching, a request computes its prompt from the state the
    connector loaded, and a request sharing the loaded prefix hits it locally
    instead of loading it again."""
    token_ids, expected = reference
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    LOADS.clear()
    llm = _llm(
        enable_prefix_caching=True,
        prefix_match_unit=prefix_match_unit,
        num_gpu_blocks_override=32,
    )
    # Cycle every block through unrelated prompts first, so the block a load
    # writes holds some other state before the load lands.
    for i in range(3):
        _generate(llm, token_ids[::-1][i * 1000 : i * 1000 + 3000], "other")
    prefix = token_ids[:prefix_len]
    outputs = [
        _generate(llm, prefix + q, "loaded", {"load": prefix_len})
        for q in _questions(token_ids, prefix_len)
    ]
    del llm
    cleanup_dist_env_and_memory()

    check_logprobs_close(
        outputs_0_lst=expected[prefix_len],
        outputs_1_lst=outputs,
        name_0="no_prefix_caching",
        name_1="prefix_caching",
        always_check_logprobs=True,
    )
    # Only the first request loads; the second resumes from the cached state.
    assert len(LOADS) == 1
