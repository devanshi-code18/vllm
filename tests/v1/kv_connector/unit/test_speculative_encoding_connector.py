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
    NULL_BLOCK_ID,
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
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
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
    # Most tests check what a request loads, so they load in the step that
    # schedules it; test_document_loads_in_background covers the default.
    extra.setdefault("load_async", False)
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
        _scheduler(_config(tmp_path)), _request(DOCUMENT + question)
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
        _scheduler(_config(tmp_path)), _request(DOCUMENT)
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
    scheduler = _scheduler(_config(tmp_path))
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
    scheduler = _scheduler(_config(tmp_path, require_cache_salt=False))
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
    scheduler = _scheduler(config)
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
    assert failed == {load.block_ids[0][3]}

    scheduler.update_from_output(
        output,
        create_model_runner_output([shared, loading], invalid_block_ids=failed),
    )
    assert shared.status == RequestStatus.RUNNING
    assert loading.status == RequestStatus.FINISHED_ERROR


def _kv_cache_config(num_groups=1, num_blocks=10000):
    """The test model's attention layers, in `num_groups` KV cache groups."""
    spec = AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )
    names = [f"model.layers.{i}.self_attn.attn" for i in range(LAYERS)]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            make_kv_cache_group(names[g::num_groups], spec) for g in range(num_groups)
        ],
    )


def _scheduler(config):
    return create_scheduler(config, kv_cache_config=_kv_cache_config())


def _worker(config, kv_cache_config):
    worker = SpeculativeEncodingConnector(
        config, KVConnectorRole.WORKER, kv_cache_config
    )
    worker.register_kv_caches({"layer.0": torch.zeros(1)})
    return worker


def _forward_context():
    return ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})


@pytest.mark.parametrize("delete", [False, True])
def test_document_loads_in_background(tmp_path, delete):
    """By default a request waits for its document while other requests keep
    running, and is scheduled once every worker reports the load done; a
    failed load fails that request only."""
    _publish(tmp_path)
    config = _config(tmp_path, load_async=True)
    scheduler = _scheduler(config)
    running = _request([1, 2, 3], document=None, request_id="running")
    output, _, _ = _schedule_one(scheduler, running)
    scheduler.update_from_output(output, create_model_runner_output([running]))

    loading = _request(DOCUMENT + [7, 8], request_id="loading")
    scheduler.add_request(loading)
    output = scheduler.schedule()
    meta = output.kv_connector_metadata
    assert isinstance(meta, SpeculativeEncodingConnectorMetadata)
    (load,) = meta.loads
    assert loading.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert "loading" not in output.num_scheduled_tokens
    assert output.num_scheduled_tokens["running"] == 1
    assert (load.start, load.end) == (0, len(DOCUMENT))

    if delete:
        (tmp_path / "doc0" / "kv.safetensors").unlink()
    worker = _worker(config, scheduler.kv_cache_config)
    worker.bind_connector_metadata(output.kv_connector_metadata)
    worker.start_load_kv(_forward_context())
    worker.shutdown()
    results = worker.get_transfer_results(set())
    assert results.finished_recving == {"loading"}
    assert results.failed_recving == ({"loading"} if delete else set())
    assert worker.get_block_ids_with_load_errors() == set()

    scheduler.update_from_output(
        output,
        create_model_runner_output(
            [running],
            finished_recving=results.finished_recving,
            failed_recving=results.failed_recving,
        ),
    )
    output = scheduler.schedule()
    if delete:
        assert loading.status == RequestStatus.FINISHED_ERROR
    else:
        # Only the last prompt tokens are computed.
        assert output.num_scheduled_tokens["loading"] == 2


def test_several_kv_cache_groups_need_background_loads(tmp_path):
    # A failed step load is reported by block id, which can't name a group.
    with pytest.raises(ValueError, match="needs load_async"):
        SpeculativeEncodingConnector(
            _config(tmp_path), KVConnectorRole.SCHEDULER, _kv_cache_config(2)
        )
    SpeculativeEncodingConnector(
        _config(tmp_path, load_async=True),
        KVConnectorRole.SCHEDULER,
        _kv_cache_config(2),
    )


def _sliding_window_kv_cache_config(window=BLOCK_SIZE):
    # Layers 0..5 full attention, 6..11 a sliding window.
    names = [f"model.layers.{i}.self_attn.attn" for i in range(LAYERS)]
    spec = AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            make_kv_cache_group(names[:6], spec),
            make_kv_cache_group(
                names[6:],
                spec,
                SlidingWindowManager,
                SlidingWindowManager.config(sliding_window=window),
            ),
        ],
    )


def test_sliding_window_group_loads_its_window(tmp_path):
    """With the hybrid KV cache manager, a load covers every group: a sliding
    window group's blocks before the window stay null and are not loaded."""
    _publish(tmp_path)
    kv_cache_config = _sliding_window_kv_cache_config()
    config = _config(tmp_path, load_async=True)
    scheduler = create_scheduler(config, kv_cache_config=kv_cache_config)
    scheduler.add_request(_request(DOCUMENT + [7, 8], request_id="loading"))
    meta = scheduler.schedule().kv_connector_metadata
    assert isinstance(meta, SpeculativeEncodingConnectorMetadata)
    (load,) = meta.loads
    full, window = load.block_ids
    assert NULL_BLOCK_ID not in full
    # Tokens 0..49: the window group holds only the blocks of its last tokens.
    assert window[0] == NULL_BLOCK_ID and window[-1] != NULL_BLOCK_ID
    # vLLM zeroes the blocks first; the load writes after that.
    assert (
        SpeculativeEncodingConnector(
            config, KVConnectorRole.SCHEDULER, kv_cache_config
        ).get_loaded_kv_cache_group_ids(scheduler.requests["loading"])
        == ()
    )


def test_sliding_window_layer_must_hold_the_first_computed_tokens_window(tmp_path):
    """A document whose sliding window layers keep only the last window: a
    prompt that is the document loads one token short, so its first computed
    token attends to a token those layers don't hold, and it prefills."""
    first = len(DOCUMENT) - BLOCK_SIZE + 1
    kv = {
        i: torch.zeros(
            2,
            len(DOCUMENT) - (first if i >= 6 else 0),
            HEADS,
            HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        for i in range(LAYERS)
    }
    write_document(
        tmp_path, "doc0", DOCUMENT, kv, kv_first={i: first for i in range(6, LAYERS)}
    )
    connector = SpeculativeEncodingConnector(
        _config(tmp_path, load_async=True),
        KVConnectorRole.SCHEDULER,
        _sliding_window_kv_cache_config(),
    )
    assert connector.get_num_new_matched_tokens(_request(DOCUMENT), 0) == (0, False)
    assert connector.get_num_new_matched_tokens(_request(DOCUMENT + [7]), 0) == (
        len(DOCUMENT),
        True,
    )
    # A served window wider than the stored one.
    connector = SpeculativeEncodingConnector(
        _config(tmp_path, load_async=True),
        KVConnectorRole.SCHEDULER,
        _sliding_window_kv_cache_config(window=2 * BLOCK_SIZE),
    )
    assert connector.get_num_new_matched_tokens(_request(DOCUMENT + [7]), 0) == (
        0,
        False,
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
