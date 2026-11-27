# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpeculativeEncodingConnector without a GPU: which requests load a translated
document from the store, what a failed load reports, and the store and reader
helpers."""

import json

import pytest
import torch

from vllm import SamplingParams
from vllm.config import SpeculativeConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding import (
    write_document,
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.connector import (  # noqa: E501
    SpeculativeEncodingConnector,
    SpeculativeEncodingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.reader import (  # noqa: E501
    local_kv_heads,
    split_ranges,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_group,
    set_mtp_config,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
DOCUMENT = list(range(100, 150))  # 50 prompt tokens: 3 full blocks + 2
# The test model, facebook/opt-125m: 12 layers of 12 KV heads of size 64.
LAYERS, HEADS, HEAD_SIZE = 12, 12, 64


def _publish(store, key="doc0", tokens=DOCUMENT, heads=HEADS):
    kv = torch.zeros(LAYERS, 2, len(tokens), heads, HEAD_SIZE, dtype=torch.bfloat16)
    return write_document(store, key, tokens, kv)


def _config(store, **extra):
    return create_vllm_config(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=256,
        kv_connector="SpeculativeEncodingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"store": str(store), **extra},
    )


def _request(prompt, document="doc0", cache_salt="translated", request_id=None):
    init_none_hash(sha256)
    extra_args = {}
    if document is not None:
        extra_args["kv_transfer_params"] = {
            "speculative_encoding": {"document": document}
        }
    return Request(
        request_id=request_id or f"req-{len(prompt)}-{document}-{cache_salt}",
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=4, extra_args=extra_args),
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _schedule_one(scheduler, request):
    scheduler.add_request(request)
    output = scheduler.schedule()
    meta = output.kv_connector_metadata
    assert isinstance(meta, SpeculativeEncodingConnectorMetadata)
    return output, output.num_scheduled_tokens[request.request_id], meta.loads


def test_document_tokens_load_from_store(tmp_path):
    _publish(tmp_path)
    question = [7, 8, 9, 10]
    _, scheduled, loads = _schedule_one(
        create_scheduler(_config(tmp_path)), _request(DOCUMENT + question)
    )
    assert [(load.key, load.start, load.end) for load in loads] == [
        ("doc0", 0, len(DOCUMENT))
    ]
    assert scheduled == len(question)


def test_last_prompt_token_is_computed(tmp_path):
    # A prompt that is exactly the document still computes its last token,
    # whose logits sample the first output token.
    _publish(tmp_path)
    _, scheduled, loads = _schedule_one(
        create_scheduler(_config(tmp_path)), _request(DOCUMENT)
    )
    assert [(load.start, load.end) for load in loads] == [(0, len(DOCUMENT) - 1)]
    assert scheduled == 1


def _write_meta(store, key, text):
    (store / key).mkdir()
    (store / key / "meta.json").write_text(text)


@pytest.mark.parametrize(
    "case",
    [
        "other_prompt",
        "missing_document",
        "deleted_document",
        "no_cache_salt",
        "no_params",
        "path_key",
        "key_is_a_file",
        "unknown_format",
        "empty_meta",
        "missing_field",
        "other_model",
    ],
)
def test_prefills_when_document_does_not_apply(tmp_path, case):
    """Every client-chosen key or broken store entry prefills; none fails the
    engine."""
    _publish(tmp_path)
    scheduler = create_scheduler(_config(tmp_path))
    # A first request (own salt: no prefix hits) caches the document's metadata.
    _schedule_one(
        scheduler, _request(DOCUMENT + [1], cache_salt="first", request_id="first")
    )
    prompt = DOCUMENT + [7, 8]
    kwargs = {}
    if case == "other_prompt":
        prompt = DOCUMENT[:-1] + [1] + [7, 8]
    elif case == "missing_document":
        kwargs["document"] = "doc1"
    elif case == "deleted_document":
        (tmp_path / "doc0" / "kv.safetensors").unlink()
    elif case == "no_cache_salt":
        kwargs["cache_salt"] = None
    elif case == "no_params":
        kwargs["document"] = None
    elif case == "path_key":
        kwargs["document"] = f"../{tmp_path.name}/doc0"
    elif case == "key_is_a_file":
        (tmp_path / "notes.txt").write_text("hi")
        kwargs["document"] = "notes.txt"
    elif case == "unknown_format":
        _write_meta(tmp_path, "doc1", json.dumps({"format": "other"}))
        kwargs["document"] = "doc1"
    elif case == "empty_meta":
        _write_meta(tmp_path, "doc1", "")
        kwargs["document"] = "doc1"
    elif case == "missing_field":
        meta = json.loads((tmp_path / "doc0" / "meta.json").read_text())
        del meta["num_kv_heads"]
        _write_meta(tmp_path, "doc1", json.dumps(meta))
        (tmp_path / "doc1" / "kv.safetensors").write_bytes(b"")
        kwargs["document"] = "doc1"
    elif case == "other_model":
        _publish(tmp_path, key="doc1", heads=HEADS // 2)
        kwargs["document"] = "doc1"
    _, scheduled, loads = _schedule_one(scheduler, _request(prompt, **kwargs))
    assert loads == []
    assert scheduled == len(prompt)


def test_cache_salt_can_be_optional(tmp_path):
    _publish(tmp_path)
    scheduler = create_scheduler(_config(tmp_path, require_cache_salt=False))
    _, _, loads = _schedule_one(scheduler, _request(DOCUMENT + [7], cache_salt=None))
    assert len(loads) == 1


def _corrupt(store, case):
    kv = store / "doc0" / "kv.safetensors"
    if case == "deleted":
        kv.unlink()  # gone before the worker reads
        return
    # A file that disagrees with meta.json: other tokens, heads or dtype.
    tokens, heads, dtype = len(DOCUMENT), HEADS, torch.bfloat16
    if case == "other_tokens":
        tokens += 16
    elif case == "other_heads":
        heads //= 2
    else:
        dtype = torch.float16
    other = torch.zeros(LAYERS, 2, tokens, heads, HEAD_SIZE, dtype=dtype)
    write_document(store, "other", list(range(tokens)), other)
    (store / "other" / "kv.safetensors").replace(kv)


@pytest.mark.parametrize(
    "case", ["deleted", "other_tokens", "other_heads", "other_dtype"]
)
def test_failed_load_spares_shared_prefix_blocks(tmp_path, case):
    """A request whose load fails reports only the blocks the load writes, so
    a running request sharing its prefix-cache hits keeps running."""
    _publish(tmp_path)
    config = _config(tmp_path)
    scheduler = create_scheduler(config)
    shared = _request(DOCUMENT[:48] + [1, 2, 3], document=None, request_id="shared")
    output, _, _ = _schedule_one(scheduler, shared)
    scheduler.update_from_output(output, create_model_runner_output([shared]))

    loading = _request(DOCUMENT + [7, 8], request_id="loading")
    output, _, (load,) = _schedule_one(scheduler, loading)
    assert (load.start, load.end) == (48, len(DOCUMENT))
    _corrupt(tmp_path, case)
    worker = SpeculativeEncodingConnector(
        config, KVConnectorRole.WORKER, scheduler.kv_cache_config
    )
    worker.bind_connector_metadata(output.kv_connector_metadata)
    worker.start_load_kv(
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    )
    failed = worker.get_block_ids_with_load_errors()
    assert failed == {load.block_ids[3]}

    scheduler.update_from_output(
        output,
        create_model_runner_output([shared, loading], invalid_block_ids=failed),
    )
    assert shared.status == RequestStatus.RUNNING
    assert loading.status == RequestStatus.FINISHED_ERROR


def _kv_cache_config(num_groups=1):
    spec = AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            make_kv_cache_group([f"layer.{i}"], spec) for i in range(num_groups)
        ],
    )


def test_refuses_several_kv_cache_groups(tmp_path):
    with pytest.raises(ValueError, match="single KV cache group"):
        SpeculativeEncodingConnector(
            _config(tmp_path), KVConnectorRole.SCHEDULER, _kv_cache_config(2)
        )


def test_refuses_draft_model_speculation(tmp_path):
    """The store holds the target's K/V only; a drafter's attention layers
    would get none (EAGLE numbers them past the target's) or the target's."""
    ngram = _config(tmp_path)
    ngram.speculative_config = SpeculativeConfig(
        model="ngram", num_speculative_tokens=2
    )
    SpeculativeEncodingConnector(ngram, KVConnectorRole.SCHEDULER, _kv_cache_config())
    mtp = _config(tmp_path)
    set_mtp_config(mtp)
    with pytest.raises(ValueError, match=r"draft model \(mtp\)"):
        SpeculativeEncodingConnector(mtp, KVConnectorRole.SCHEDULER, _kv_cache_config())


def test_write_document_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError, match="invalid document key"):
        _publish(tmp_path, key="../escape")
    with pytest.raises(ValueError, match="KV holds 51 tokens"):
        write_document(
            tmp_path,
            "doc0",
            DOCUMENT,
            torch.zeros(2, 2, 51, 1, 4, dtype=torch.bfloat16),
        )
    _publish(tmp_path)
    with pytest.raises(FileExistsError):
        _publish(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc0"]


def test_split_ranges():
    # Adjacent ranges merge; pieces never exceed the buffer size.
    assert split_ranges([(0, 10), (10, 5), (30, 20)], 8) == [
        [(0, 8)],
        [(8, 7), (30, 1)],
        [(31, 8)],
        [(39, 8)],
        [(47, 3)],
    ]
    assert split_ranges([(0, 8), (8, 8)], 16) == [[(0, 16)]]


@pytest.mark.parametrize(
    "num_kv_heads, world, expected",
    [
        (8, 2, [range(0, 4), range(4, 8)]),  # split
        (2, 4, [range(0, 1), range(0, 1), range(1, 2), range(1, 2)]),  # replicated
        (1, 2, [range(0, 1), range(0, 1)]),
    ],
)
def test_local_kv_heads(num_kv_heads, world, expected):
    local = max(1, num_kv_heads // world)
    assert [local_kv_heads(num_kv_heads, local, r, world) for r in range(world)] == (
        expected
    )
