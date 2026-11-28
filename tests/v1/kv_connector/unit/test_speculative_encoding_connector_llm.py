# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Through `LLM`: a request that names a document in the store reads that
document's KV from the store instead of prefilling it, with any TP size."""

import json
import os

import pybase64 as base64
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.models.utils import check_logprobs_close
from tests.utils import create_new_process_for_each_test, multi_gpu_marks
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding import (
    share_document,
    write_document,
)
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

MODEL = "Qwen/Qwen3-0.6B"
DOCUMENT = (
    "The Harbor Lease was signed on 3 March 2021 by Mira Okafor for the landlord "
    "and by Tomas Reyes for Bluewater Cargo. The rent is 4,200 euros a month, due "
    "on the first working day. Either side may end the lease with ninety days of "
    "written notice. The tenant keeps the north warehouse and two loading bays, "
    "and may sublet one bay with the landlord's written consent."
)
QUESTIONS = [
    " Question: who signed for Bluewater Cargo? Answer:",
    " Question: how much is the rent? Answer:",
    " Question: what may the tenant sublet? Answer:",
]


def _document_kv(ids: list[int]) -> torch.Tensor:
    """The document's post-RoPE K/V from transformers, `[layers, 2, N, kv_heads,
    head_size]`: what a translator would produce, here exact."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    device = current_platform.device_type
    with torch.no_grad():
        cache = model.to(device)(torch.tensor([ids], device=device)).past_key_values
    kv = torch.stack(
        [torch.stack((layer.keys[0], layer.values[0])) for layer in cache.layers]
    )
    del model, cache
    torch.accelerator.empty_cache()
    return kv.transpose(2, 3).cpu()


@pytest.mark.skipif(current_platform.is_rocm(), reason="not validated on ROCm")
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    "tp, kv_cache_dtype, load_async, shared",
    [
        (1, "auto", True, False),
        (1, "auto", False, False),
        (1, "fp8", True, False),
        pytest.param(2, "auto", True, False, marks=multi_gpu_marks(num_gpus=2)),
        (1, "auto", True, True),
        (1, "auto", False, True),
        pytest.param(2, "auto", True, True, marks=multi_gpu_marks(num_gpus=2)),
    ],
)
def test_requests_read_document_kv_from_store(
    tmp_path, tp, kv_cache_dtype, load_async, shared
):
    """`shared`: the K/V stays on this process's GPU (CUDA IPC) instead of a
    file; every worker maps it on its own device."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    document = tokenizer(DOCUMENT)["input_ids"]
    prompts = [document + tokenizer(q)["input_ids"] for q in QUESTIONS]
    kv = _document_kv(document)
    # Same tokens, wrong KV: answers change only if requests read the store.
    documents = {"lease": kv, "zeros": torch.zeros_like(kv)}
    published = []
    allocated = torch.accelerator.memory_allocated() if shared else 0
    for key, document_kv in documents.items():
        if shared:
            published.append(
                share_document(
                    tmp_path, key, document, document_kv.to("cuda:0").contiguous()
                )
            )
        else:
            write_document(tmp_path, key, document, document_kv)

    llm = LLM(
        MODEL,
        tensor_parallel_size=tp,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SpeculativeEncodingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "store": str(tmp_path),
                "load_async": load_async,
                "shared_documents": shared,
            },
        ),
    )

    def generate(key: str | None):
        extra = {}
        if key is not None:
            extra["kv_transfer_params"] = {"speculative_encoding": {"document": key}}
        params = SamplingParams(
            temperature=0.0, max_tokens=16, logprobs=5, extra_args=extra
        )
        outputs = llm.generate(
            [
                TokensPrompt(prompt_token_ids=p, cache_salt=f"{key}-{i}")
                for i, p in enumerate(prompts)
            ],
            params,
        )
        return outputs, [
            (o.outputs[0].token_ids, o.outputs[0].text, o.outputs[0].logprobs)
            for o in outputs
        ]

    native, native_results = generate(None)
    stored, stored_results = generate("lease")
    _, zeros_results = generate("zeros")

    assert [o.num_cached_tokens for o in native] == [0] * len(prompts)
    assert [o.num_cached_tokens for o in stored] == [len(document)] * len(prompts)
    check_logprobs_close(
        outputs_0_lst=native_results,
        outputs_1_lst=stored_results,
        name_0="prefill",
        name_1="store",
    )
    assert [r[0] for r in zeros_results] != [r[0] for r in native_results]
    if shared:
        # A handle that does not map (forged, or its producer gone) fails the
        # load (kv_load_failure_policy "fail"); the engine keeps serving.
        meta_path = tmp_path / "lease" / "meta.json"
        meta = json.loads(meta_path.read_text())
        handle = base64.b64decode(meta["ipc"]["handle"])
        meta["ipc"]["handle"] = base64.b64encode(
            bytes(b ^ 0xFF for b in handle)
        ).decode()
        meta_path.write_text(json.dumps(meta))
        failed, _ = generate("lease")
        assert [o.outputs[0].finish_reason for o in failed] == ["error"] * len(prompts)
        _, again_results = generate(None)
        check_logprobs_close(
            outputs_0_lst=native_results,
            outputs_1_lst=again_results,
            name_0="prefill",
            name_1="prefill after failed loads",
        )
    for document_ in published:
        document_.close(timeout=5)
        assert not document_.path.exists()
    # Every load mapped and dropped the shared K/V; closing still frees it.
    published.clear()
    if shared:
        assert torch.accelerator.memory_allocated() == allocated


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA IPC")
@create_new_process_for_each_test("spawn")
def test_share_document_refuses_memory_cuda_ipc_cannot_export(tmp_path):
    """Memory from PyTorch's expandable_segments allocator has no CUDA IPC
    handle: sharing it fails with the cause, before anything is published."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    kv = torch.zeros(2, 2, 16, 2, 8, dtype=torch.bfloat16, device="cuda:0")
    with pytest.raises(ValueError, match="expandable_segments"):
        share_document(tmp_path, "doc", list(range(16)), kv)
    assert not (tmp_path / "doc").exists()
