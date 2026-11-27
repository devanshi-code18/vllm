# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Through `LLM`: a request that names a document in the store reads that
document's KV from the store instead of prefilling it, with any TP size."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.models.utils import check_logprobs_close
from tests.utils import create_new_process_for_each_test, multi_gpu_marks
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding import (
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
    "tp, kv_cache_dtype",
    [
        (1, "auto"),
        (1, "fp8"),
        pytest.param(2, "auto", marks=multi_gpu_marks(num_gpus=2)),
    ],
)
def test_requests_read_document_kv_from_store(tmp_path, tp, kv_cache_dtype):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    document = tokenizer(DOCUMENT)["input_ids"]
    prompts = [document + tokenizer(q)["input_ids"] for q in QUESTIONS]
    kv = _document_kv(document)
    write_document(tmp_path, "lease", document, kv)
    # Same tokens, wrong KV: answers change only if requests read the store.
    write_document(tmp_path, "zeros", document, torch.zeros_like(kv))

    llm = LLM(
        MODEL,
        tensor_parallel_size=tp,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=1024,
        gpu_memory_utilization=0.4,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SpeculativeEncodingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"store": str(tmp_path)},
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
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            params,
        )
        return outputs, [
            (o.outputs[0].token_ids, o.outputs[0].text, o.outputs[0].logprobs)
            for o in outputs
        ]

    stored, stored_results = generate("lease")
    # Same tokens as the requests above: their translated blocks must not hit.
    native, native_results = generate(None)
    _, zeros_results = generate("zeros")

    # The first native request finds nothing cached: the translated blocks of
    # the same tokens are keyed under the document's provenance. (The others
    # share its native blocks.)
    assert native[0].num_cached_tokens == 0
    assert [o.num_cached_tokens for o in stored] == [len(document)] * len(prompts)
    check_logprobs_close(
        outputs_0_lst=native_results,
        outputs_1_lst=stored_results,
        name_0="prefill",
        name_1="store",
    )
    assert [r[0] for r in zeros_results] != [r[0] for r in native_results]
