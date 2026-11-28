# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve documents whose KV cache was translated from another model.

Speculative encoding (https://github.com/b10-tv/speculative-encoding) lets one
model read a document once and gives another model its KV cache: a small
translator maps the source model's per-layer hidden states to this model's, and
this model's own frozen K/V projections turn them into K/V. A producer
publishes each translated document into a store (`store.write_document`); a
request whose prompt starts with the document's tokens, and that names it in
`kv_transfer_params={"speculative_encoding": {"document": "<key>"}}`, loads
those tokens' KV from the store instead of prefilling them.

The scheduler and every worker read the store themselves (no state is shared
through Python objects), so this works with multiprocessing, the OpenAI server,
streaming and tensor or pipeline parallelism. Each tensor-parallel rank reads
only its own KV heads.
"""

import contextlib
import json
import math
import os
import queue
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.reader import (  # noqa: E501
    SAFETENSORS_DTYPES,
    StagedReader,
    check_layer,
    layer_ranges,
    local_kv_heads,
    split_ranges,
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.store import (  # noqa: E501
    StoredDocument,
    read_document,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

PARAMS_KEY = "speculative_encoding"
# BlockPool's null block (its first block), which sliding window groups use
# for blocks outside the window.
NULL_BLOCK_ID = 0


@dataclass
class DocumentLoad:
    request_id: str
    key: str
    num_tokens: int  # of the whole document
    dtype: torch.dtype
    start: int
    end: int
    block_ids: tuple[list[int], ...]  # per KV cache group
    kv_first: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Group:
    block_size: int
    gdn: bool  # gated delta net state, else attention K/V
    layers: tuple[int, ...]
    sliding_window: int | None = None


@dataclass
class SpeculativeEncodingConnectorMetadata(KVConnectorMetadata):
    loads: list[DocumentLoad] = field(default_factory=list)


class SpeculativeEncodingConnector(KVConnectorBase_V1, SupportsHMA):
    """Loads a request's document tokens from a store of translated KV.

    `kv_connector_extra_config`:
        store: The store directory (required), readable by the scheduler and
            every worker.
        require_cache_salt: Load only for requests with a `cache_salt`
            (default True). Use a salt only for translated requests, so their
            blocks never serve requests that prefill the same tokens.
        load_async: Load documents in the background (default True): a request
            waits for its document while the other requests keep running.
            With false, the step that schedules the request loads it, and the
            other requests in that step wait for the load.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        store = self._kv_transfer_config.get_from_extra_config("store", None)
        if not store:
            raise ValueError(
                "SpeculativeEncodingConnector needs kv_connector_extra_config"
                '["store"], the translated document store directory'
            )
        self._store = Path(store)
        self._require_salt = bool(
            self._kv_transfer_config.get_from_extra_config("require_cache_salt", True)
        )
        self._load_async = bool(
            self._kv_transfer_config.get_from_extra_config("load_async", True)
        )
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.method not in (
            "ngram",
            "ngram_gpu",
            "suffix",
        ):
            raise ValueError(
                "SpeculativeEncodingConnector stores the target model's K/V only; "
                f"speculative decoding with a draft model ({speculative.method}) "
                "is not supported (ngram and suffix are)"
            )
        self._groups = tuple(
            self._check_group(group) for group in kv_cache_config.kv_cache_groups
        )
        self._layer_group = {
            name: group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }
        self._gdn = any(group.gdn for group in self._groups)
        if len(self._groups) > 1 and not self._load_async:
            # A failed step load is reported by block id, which the scheduler
            # can map back to requests only with a single KV cache group.
            raise ValueError(
                "SpeculativeEncodingConnector needs load_async for models with "
                f"several KV cache groups ({len(self._groups)})"
            )
        if self._gdn and speculative is not None:
            raise ValueError(
                "SpeculativeEncodingConnector writes gated delta net state for "
                "one token, so it does not support speculative decoding"
            )
        world = vllm_config.parallel_config.tensor_parallel_size
        text_config = vllm_config.model_config.hf_text_config
        for attr in ("linear_num_key_heads", "linear_num_value_heads"):
            heads = getattr(text_config, attr, None)
            if self._gdn and heads is not None and heads % world:
                raise ValueError(
                    "SpeculativeEncodingConnector splits gated delta net state "
                    f"by head: {attr} = {heads} is not divisible by tensor "
                    f"parallel size {world}"
                )
        self._block_size = self._groups[0].block_size
        model_config = vllm_config.model_config
        self._geometry = (
            model_config.get_total_num_kv_heads(),
            model_config.get_head_size(),
        )
        # Scheduler side.
        self._documents: OrderedDict[str, StoredDocument] = OrderedDict()
        self._matches: dict[str, StoredDocument | None] = {}
        self._pending: dict[str, DocumentLoad] = {}
        self._warned: set[str] = set()
        # Worker side.
        self._dtype = model_config.dtype
        self._reader = StagedReader()
        self._failed_block_ids: set[int] = set()
        self._layers: dict[str, Any] | None = None
        # Background loads run one at a time on their own thread and stream.
        self._loader: ThreadPoolExecutor | None = None
        self._stream: torch.Stream | None = None
        self._done: queue.SimpleQueue[tuple[DocumentLoad, bool]] = queue.SimpleQueue()

    @staticmethod
    def _check_group(group: Any) -> _Group:
        gdn = set()
        for name, spec in group.kv_cache_specs.items():
            if isinstance(spec, AttentionSpec) and not isinstance(
                spec, MLAAttentionSpec
            ):
                gdn.add(False)
            elif (
                isinstance(spec, MambaSpec)
                and spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
            ):
                gdn.add(True)
            else:
                raise ValueError(
                    "SpeculativeEncodingConnector loads attention K/V and gated "
                    f"delta net state only; {name} is {type(spec).__name__}"
                )
        (is_gdn,) = gdn
        params = group.manager_config.params
        return _Group(
            group.block_size,
            is_gdn,
            tuple(sorted(extract_layer_index(name) for name in group.layer_names)),
            params.sliding_window
            if isinstance(params, SlidingWindowManager.Params)
            else None,
        )

    @property
    def loads_mamba_state_at_load_end(self) -> bool:
        # For use with sync-load-mamba-precopy: the state of a load of n
        # tokens goes into the block holding token n - 1, where the model
        # computes from it. Hybrid loads here are background loads, which that
        # ordering doesn't affect; connector-hybrid-state-cache's caching of
        # background loads is untested with this connector.
        return self._gdn

    # ==============================
    # Scheduler-side methods
    # ==============================

    def _skip(self, kind: str, message: str, *args: Any) -> None:
        # Clients choose document keys: warn once per kind, never per key.
        log = logger.debug if kind in self._warned else logger.warning
        self._warned.add(kind)
        log(message + "; prefilling the request", *args)

    def _document(self, key: str) -> StoredDocument | None:
        document = self._documents.get(key)
        if document is not None and (self._store / key / "kv.safetensors").is_file():
            self._documents.move_to_end(key)
            return document
        self._documents.pop(key, None)
        try:
            document = read_document(self._store, key)
        except FileNotFoundError:
            self._skip("missing", "document %r is not in %s", key, self._store)
            return None
        except (OSError, ValueError) as e:
            self._skip("invalid", "document %r is not readable: %s", key, e)
            return None
        attention = {i for g in self._groups if not g.gdn for i in g.layers}
        gdn = {i for g in self._groups if g.gdn for i in g.layers}
        if not attention <= set(document.kv_layers) or not gdn <= set(
            document.state_layers
        ):
            self._skip(
                "layers",
                "document %r lacks K/V or state of some of this model's layers",
                key,
            )
            return None
        if (document.num_kv_heads, document.head_size) != self._geometry:
            self._skip(
                "geometry",
                "document %r has %d KV heads of size %d; this model has %d of %d",
                key,
                document.num_kv_heads,
                document.head_size,
                *self._geometry,
            )
            return None
        self._documents[key] = document
        if len(self._documents) > 1024:
            self._documents.popitem(last=False)
        return document

    def _match(self, request: "Request") -> StoredDocument | None:
        if request.request_id in self._matches:
            return self._matches[request.request_id]
        document = None
        spec = (request.kv_transfer_params or {}).get(PARAMS_KEY)
        if isinstance(spec, dict) and isinstance(spec.get("document"), str):
            found = self._document(spec["document"])
            prompt = request.prompt_token_ids or []
            if found is None:
                pass
            elif tuple(prompt[: found.num_tokens]) != found.prompt_token_ids:
                self._skip(
                    "prefix",
                    "request %s names document %r but its prompt does not start "
                    "with the document's %d tokens",
                    request.request_id,
                    found.key,
                    found.num_tokens,
                )
            elif self._require_salt and request.cache_salt is None:
                self._skip(
                    "salt",
                    "request %s names a translated document but has no "
                    "cache_salt (set require_cache_salt=false to load anyway)",
                    request.request_id,
                )
            else:
                document = found
        self._matches[request.request_id] = document
        return document

    def _num_tokens(self, request: "Request", document: StoredDocument) -> int:
        # The last prompt token is computed to sample the first output token.
        if not self._gdn:
            return min(document.num_tokens, request.num_prompt_tokens - 1)
        # Gated delta net state is stored after the whole document only.
        return document.num_tokens * (request.num_prompt_tokens > document.num_tokens)

    def get_loaded_kv_cache_group_ids(self, request: "Request") -> tuple[int, ...]:
        """No groups, so vLLM zeroes a loading request's new blocks: the load
        waits for an event recorded after that step's zeroing, and the tail of
        a document's last block, which it does not write, must read as zeros
        (FlashInfer's TRTLLM kernels read whole pages; a stale hybrid page can
        hold NaN)."""
        return ()

    def _holds_attended_kv(
        self, document: StoredDocument, start: int, end: int
    ) -> bool:
        """Whether the document holds the K/V the request reads from the
        loaded tokens `[start, end)`: a full attention layer all of them, a
        sliding window layer the window of the first computed token."""
        for group in self._groups:
            if group.gdn:
                continue
            first = start
            if group.sliding_window is not None:
                first = max(start, end - group.sliding_window + 1)
            if any(document.kv_first.get(i, 0) > first for i in group.layers):
                return False
        return True

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        document = self._match(request)
        if document is None:
            return 0, False
        end = self._num_tokens(request, document)
        num_tokens = max(0, end - num_computed_tokens)
        if num_tokens and not self._holds_attended_kv(
            document, num_computed_tokens, end
        ):
            self._skip(
                "window",
                "document %r lacks K/V that request %s attends to (a layer's "
                "stored tokens start after its window)",
                document.key,
                request.request_id,
            )
            return 0, False
        return num_tokens, self._load_async and num_tokens > 0

    def may_load_async(self, request: "Request") -> bool:
        spec = (request.kv_transfer_params or {}).get(PARAMS_KEY)
        return self._load_async and isinstance(spec, dict)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens <= 0:
            return
        document = self._matches[request.request_id]
        assert document is not None
        end = self._num_tokens(request, document)
        # A background load starts now; a step load waits for the step that
        # schedules the request.
        block_ids = (
            tuple(list(ids) for ids in blocks.get_block_ids())
            if self._load_async
            else ()
        )
        self._pending[request.request_id] = DocumentLoad(
            request.request_id,
            document.key,
            document.num_tokens,
            document.dtype,
            end - num_external_tokens,
            end,
            block_ids,
            dict(document.kv_first),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = SpeculativeEncodingConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            load = self._pending.get(new_req.req_id)
            if load is not None and not load.block_ids:
                load.block_ids = tuple(list(ids) for ids in new_req.block_ids)
        for request_id, load in list(self._pending.items()):
            if load.block_ids:
                meta.loads.append(self._pending.pop(request_id))
        return meta

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._matches.pop(request.request_id, None)
        self._pending.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, [])

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._reader.allocate()
        if self._load_async:
            device = next(iter(kv_caches.values())).device
            if device.type == "cpu":
                self._loader = ThreadPoolExecutor(1, thread_name_prefix="se-load")
            else:
                self._loader = ThreadPoolExecutor(
                    1,
                    thread_name_prefix="se-load",
                    initializer=torch.accelerator.set_device_index,
                    initargs=(device.index,),
                )
                self._stream = torch.Stream(device=device)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SpeculativeEncodingConnectorMetadata)
        if not metadata.loads:
            return
        layers = self._cache_layers(forward_context)
        for load in metadata.loads:
            if self._loader is None:
                # A deliberate bulk transfer: the step waits for its loads.
                with gpu_sync_allowed():
                    if not self._try_load(load, layers):
                        self._fail(load)
                continue
            # The load writes blocks this step may still be zeroing.
            ready = None
            if self._stream is not None:
                ready = torch.Event()
                ready.record()
            self._loader.submit(self._load_in_background, load, layers, ready)

    def _cache_layers(self, forward_context: "ForwardContext") -> dict[str, Any]:
        """The layers whose cache a load writes, by name: those of the KV cache
        groups that own their cache (not sharing another layer's)."""
        if self._layers is None:
            self._layers = {
                name: layer
                for name, layer in forward_context.no_compile_layers.items()
                if name in self._layer_group
                and getattr(layer, "kv_sharing_target_layer_name", None) is None
            }
        return self._layers

    def _load_in_background(
        self, load: DocumentLoad, layers: dict[str, Any], ready: torch.Event | None
    ) -> None:
        ok = False
        try:
            stream = self._stream
            with stream if stream is not None else contextlib.nullcontext():
                if stream is not None and ready is not None:
                    stream.wait_event(ready)
                with gpu_sync_allowed():
                    ok = self._try_load(load, layers, stream)
        finally:
            self._done.put((load, ok))

    def _try_load(
        self,
        load: DocumentLoad,
        layers: dict[str, Any],
        stream: torch.Stream | None = None,
    ) -> bool:
        started = time.perf_counter()
        try:
            self._load(load, layers)
            if stream is not None:
                stream.synchronize()
        except Exception:
            logger.exception(
                "loading document %r for request %s failed", load.key, load.request_id
            )
            return False
        logger.debug(
            "loaded %d tokens of document %r for request %s in %.3f s",
            load.end - load.start,
            load.key,
            load.request_id,
            time.perf_counter() - started,
        )
        return True

    def _fail(self, load: DocumentLoad) -> None:
        # Only the blocks this load writes: earlier ones are prefix cache hits
        # that other requests may share. (Step loads have one KV cache group.)
        first = load.start // self._block_size
        last = -(-load.end // self._block_size)
        self._failed_block_ids.update(load.block_ids[0][first:last])

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        # A failed background load fails its request alone (the scheduler
        # recomputes or rejects it per kv_load_failure_policy).
        results = KVConnectorTransferResults()
        while True:
            try:
                load, ok = self._done.get_nowait()
            except queue.Empty:
                return results
            results.finished_recving.add(load.request_id)
            if not ok:
                results.failed_recving.add(load.request_id)

    def _load(self, load: DocumentLoad, layers: dict[str, Any]) -> None:
        fd = os.open(self._store / load.key / "kv.safetensors", os.O_RDONLY)
        try:
            size = int.from_bytes(os.pread(fd, 8, 0), "little")
            header = json.loads(os.pread(fd, size, 8))
            self._write_layers(fd, 8 + size, header, load, layers)
        finally:
            os.close(fd)

    def _write_layers(
        self,
        fd: int,
        data_start: int,
        header: dict[str, Any],
        load: DocumentLoad,
        layers: dict[str, Any],
    ) -> None:
        num_kv_heads, head_size = self._geometry
        # The scheduler checked meta.json; every layer in the file must agree.
        for name, entry in header.items():
            if name.startswith("layers."):
                first = load.kv_first.get(int(name.split(".")[1]), 0)
                shape = [2, num_kv_heads, load.num_tokens - first, head_size]
                check_layer(entry, shape, load.dtype)
        if not layers:
            return
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
        plans, pieces = [], []
        slots: dict[tuple[int, int], torch.Tensor] = {}
        for name, layer in layers.items():
            group_id = self._layer_group[name]
            group = self._groups[group_id]
            index = extract_layer_index(name)
            if group.gdn:
                self._write_gdn_state(
                    fd, data_start, header, load, layer, group_id, index
                )
                continue
            start = self._first_written(load, group_id, index)
            if start >= load.end:
                continue
            entry = header[f"layers.{index}"]
            heads = local_kv_heads(num_kv_heads, layer.num_kv_heads, rank, world)
            first = load.kv_first.get(index, 0)
            ranges = layer_ranges(
                entry, data_start, heads, start - first, load.end - first
            )
            layer_pieces = split_ranges(ranges, self._reader.chunk_bytes)
            nbytes = sum(length for _, length in ranges)
            if (group_id, start) not in slots:
                slots[(group_id, start)] = self._slots(load, group_id, start)
            plans.append(
                (layer, slots[(group_id, start)], len(heads), nbytes, len(layer_pieces))
            )
            pieces.extend(layer_pieces)
        staged = self._reader.read(fd, pieces)
        try:
            for layer, layer_slots, heads, nbytes, num_pieces in plans:
                kv_cache = layer.kv_cache
                kv = torch.empty(nbytes, dtype=torch.uint8, device=kv_cache.device)
                position = 0
                for _ in range(num_pieces):
                    buffer, piece = next(staged)
                    kv[position : position + piece.numel()].copy_(
                        piece, non_blocking=True
                    )
                    self._reader.release(buffer)
                    position += piece.numel()
                # K, V of `[num_tokens, heads, head_size]`; the backend's own
                # cache writer applies the KV cache layout, dtype and scales.
                key, value = (
                    kv.view(load.dtype)
                    .view(2, heads, layer_slots.numel(), -1)
                    .to(self._dtype)
                ).transpose(1, 2)
                layer.impl.do_kv_cache_update(
                    layer,
                    key.contiguous(),
                    value.contiguous(),
                    kv_cache,
                    layer_slots.to(kv_cache.device, non_blocking=True),
                )
        finally:
            staged.close()

    def _first_written(self, load: DocumentLoad, group_id: int, index: int) -> int:
        """The first token a load writes for an attention layer: past the
        request's local hits, the layer's stored tokens, and the blocks a
        sliding window group left null."""
        block_size = self._groups[group_id].block_size
        block_ids = load.block_ids[group_id]
        start = max(load.start, load.kv_first.get(index, 0))
        block = start // block_size
        while block < len(block_ids) and block_ids[block] == NULL_BLOCK_ID:
            block += 1
        return max(start, block * block_size)

    def _slots(self, load: DocumentLoad, group_id: int, start: int) -> torch.Tensor:
        block_size = self._groups[group_id].block_size
        positions = torch.arange(start, load.end, dtype=torch.int64)
        blocks = torch.tensor(load.block_ids[group_id], dtype=torch.int64)
        return blocks[positions // block_size] * block_size + positions % block_size

    def _write_gdn_state(
        self,
        fd: int,
        data_start: int,
        header: dict[str, Any],
        load: DocumentLoad,
        layer: Any,
        group_id: int,
        index: int,
    ) -> None:
        """Writes a gated delta net layer's state after the document into the
        block holding its last token, where the model computes from it."""
        world, rank = layer.tp_size, layer.tp_rank
        k_heads = local_kv_heads(
            layer.num_k_heads, layer.num_k_heads // world, rank, world
        )
        v_heads = local_kv_heads(
            layer.num_v_heads, layer.num_v_heads // world, rank, world
        )
        kernel = layer.conv_kernel_size - 1
        conv_cache, recurrent_cache = layer.kv_cache[0], layer.kv_cache[1]
        conv_dtype, recurrent_dtype = conv_cache.dtype, recurrent_cache.dtype
        shapes = {
            "conv_q": ([layer.num_k_heads, layer.head_k_dim, kernel], k_heads),
            "conv_k": ([layer.num_k_heads, layer.head_k_dim, kernel], k_heads),
            "conv_v": ([layer.num_v_heads, layer.head_v_dim, kernel], v_heads),
            "recurrent": (
                [layer.num_v_heads, layer.head_v_dim, layer.head_k_dim],
                v_heads,
            ),
        }
        parts = {}
        for name, (shape, heads) in shapes.items():
            entry = header[f"{name}.{index}"]
            dtype = SAFETENSORS_DTYPES.get(str(entry.get("dtype")))
            if dtype is None or entry.get("shape") != shape:
                raise ValueError(
                    f"{name}.{index} is {entry.get('dtype')} {entry.get('shape')}, "
                    f"expected {shape}"
                )
            row = math.prod(shape[1:]) * dtype.itemsize
            begin = data_start + entry["data_offsets"][0] + heads.start * row
            data = os.pread(fd, len(heads) * row, begin)
            parts[name] = torch.frombuffer(bytearray(data), dtype=dtype).view(
                len(heads), *shape[1:]
            )
        conv = torch.cat(
            [parts[name].reshape(-1, kernel) for name in ("conv_q", "conv_k", "conv_v")]
        )
        block_size = self._groups[group_id].block_size
        block = load.block_ids[group_id][(load.end - 1) // block_size]
        conv_view = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        conv_view[block].copy_(conv.to(conv_dtype), non_blocking=True)
        recurrent_cache[block].copy_(
            parts["recurrent"].to(recurrent_dtype), non_blocking=True
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed, self._failed_block_ids = self._failed_block_ids, set()
        return failed

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self):
        return

    def shutdown(self):
        if self._loader is not None:
            self._loader.shutdown(wait=True)
