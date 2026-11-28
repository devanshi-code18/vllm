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
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.reader import (  # noqa: E501
    StagedReader,
    check_layer,
    layer_ranges,
    local_kv_heads,
    split_ranges,
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.shared import (  # noqa: E501
    host_id,
    open_shared,
    producer_gone,
    published_meta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.store import (  # noqa: E501
    DTYPES,
    StoredDocument,
    read_document,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

PARAMS_KEY = "speculative_encoding"


@dataclass
class DocumentLoad:
    request_id: str
    key: str
    num_tokens: int  # of the whole document
    dtype: torch.dtype
    start: int
    end: int
    block_ids: list[int]
    nonce: str = ""


@dataclass
class SpeculativeEncodingConnectorMetadata(KVConnectorMetadata):
    loads: list[DocumentLoad] = field(default_factory=list)


class SpeculativeEncodingConnector(KVConnectorBase_V1):
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
        shared_documents: Load documents shared from a producer's GPU
            (`share_document`; default False). Mapping one maps memory its
            meta.json names, so enable it only for a store that no other user
            can write; documents shared by another user are refused.
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
        self._shared = bool(
            self._kv_transfer_config.get_from_extra_config("shared_documents", False)
        )
        # A failed load is reported by block id, which the scheduler can map
        # back to requests only with a single KV cache group.
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
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError(
                "SpeculativeEncodingConnector supports models with a single KV "
                f"cache group, not {len(groups)}"
            )
        group = groups[0]
        for name, spec in group.kv_cache_specs.items():
            if isinstance(spec, MLAAttentionSpec) or not isinstance(
                spec, AttentionSpec
            ):
                raise ValueError(
                    "SpeculativeEncodingConnector loads K/V of attention layers "
                    f"only; {name} is {type(spec).__name__}"
                )
        self._block_size = group.block_size
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
        if (
            document is not None
            and (
                self._store / key / ("meta.json" if document.ipc else "kv.safetensors")
            ).is_file()
        ):
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
        if document.ipc and (reason := self._refuse_shared(key, document.ipc)):
            self._skip("shared", "document %r %s", key, reason)
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

    def _refuse_shared(self, key: str, ipc: dict[str, Any]) -> str | None:
        if not self._shared:
            return "is shared from a producer's GPU (set shared_documents=true)"
        host = host_id()
        if not host or ipc.get("host") != host:
            return "is shared from another host"
        if producer_gone(ipc):
            return "was shared by a process that has exited"
        if (self._store / key / "meta.json").stat().st_uid != os.getuid():
            return "is shared by another user"
        return None

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
        return min(document.num_tokens, request.num_prompt_tokens - 1)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        document = self._match(request)
        if document is None:
            return 0, False
        num_tokens = max(0, self._num_tokens(request, document) - num_computed_tokens)
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
        block_ids = list(blocks.get_block_ids()[0]) if self._load_async else []
        self._pending[request.request_id] = DocumentLoad(
            request.request_id,
            document.key,
            document.num_tokens,
            document.dtype,
            end - num_external_tokens,
            end,
            block_ids,
            document.nonce,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = SpeculativeEncodingConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            load = self._pending.get(new_req.req_id)
            if load is not None and not load.block_ids:
                load.block_ids = list(new_req.block_ids[0])
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
        layers = self._attention_layers(forward_context)
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

    def _attention_layers(self, forward_context: "ForwardContext") -> dict[str, Any]:
        if self._layers is None:
            from vllm.model_executor.layers.attention.attention import Attention

            self._layers = {
                name: layer
                for name, layer in forward_context.no_compile_layers.items()
                if isinstance(layer, Attention)
                and layer.kv_sharing_target_layer_name is None
                and layer.kv_cache.numel() > 0
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
        # that other requests may share.
        first = load.start // self._block_size
        last = -(-load.end // self._block_size)
        self._failed_block_ids.update(load.block_ids[first:last])

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
        folder = self._store / load.key
        with published_meta(folder) as (meta, folder_fd):
            if (
                meta.get("nonce", "") != load.nonce
                or DTYPES.get(str(meta.get("dtype"))) != load.dtype
            ):
                raise FileNotFoundError(f"{folder} was replaced since it was scheduled")
            if isinstance(meta.get("ipc"), dict):
                self._write_shared(meta["ipc"], load, layers)
                return
            fd = os.open("kv.safetensors", os.O_RDONLY, dir_fd=folder_fd)
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
        shape = [2, num_kv_heads, load.num_tokens, head_size]
        for name, entry in header.items():
            if name.startswith("layers."):
                check_layer(entry, shape, load.dtype)
        if not layers:
            return
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
        num_tokens = load.end - load.start
        plans, pieces = [], []
        for name, layer in layers.items():
            entry = header[f"layers.{extract_layer_index(name)}"]
            heads = local_kv_heads(num_kv_heads, layer.num_kv_heads, rank, world)
            ranges = layer_ranges(entry, data_start, heads, load.start, load.end)
            layer_pieces = split_ranges(ranges, self._reader.chunk_bytes)
            nbytes = sum(length for _, length in ranges)
            plans.append((layer, len(heads), nbytes, len(layer_pieces)))
            pieces.extend(layer_pieces)
        slots = self._slots(load)
        slot_mapping = None
        staged = self._reader.read(fd, pieces)
        try:
            for layer, heads, nbytes, num_pieces in plans:
                kv_cache = layer.kv_cache
                if slot_mapping is None:
                    slot_mapping = slots.to(kv_cache.device, non_blocking=True)
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
                    kv.view(load.dtype).view(2, heads, num_tokens, -1).to(self._dtype)
                ).transpose(1, 2)
                layer.impl.do_kv_cache_update(
                    layer, key.contiguous(), value.contiguous(), kv_cache, slot_mapping
                )
        finally:
            staged.close()

    def _write_shared(
        self, ipc: dict[str, Any], load: DocumentLoad, layers: dict[str, Any]
    ) -> None:
        num_kv_heads, head_size = self._geometry
        shape = [2, load.num_tokens, num_kv_heads, head_size]
        if ipc.get("shape", [])[1:] != shape:
            raise ValueError(
                f"shared K/V is {ipc.get('shape')}, expected [L, *{shape}]"
            )
        if not layers:
            return
        device = next(iter(layers.values())).kv_cache.device
        # Mapped on this rank's own device: reads cross NVLink/PCIe peer access,
        # and this process makes no context on the producer's GPU.
        num_layers = 1 + max(extract_layer_index(name) for name in layers)
        shared = open_shared(ipc, load.dtype, device, num_layers)
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
        slots = self._slots(load).to(device, non_blocking=True)
        for name, layer in layers.items():
            heads = local_kv_heads(num_kv_heads, layer.num_kv_heads, rank, world)
            key, value = shared[
                extract_layer_index(name),
                :,
                load.start : load.end,
                heads.start : heads.stop,
            ].to(self._dtype)
            layer.impl.do_kv_cache_update(
                layer, key.contiguous(), value.contiguous(), layer.kv_cache, slots
            )
        # The producer may free the memory once this returns.
        torch.accelerator.current_stream(device).synchronize()

    def _slots(self, load: DocumentLoad) -> torch.Tensor:
        positions = torch.arange(load.start, load.end, dtype=torch.int64)
        blocks = torch.tensor(load.block_ids, dtype=torch.int64)
        slots = blocks[positions // self._block_size] * self._block_size
        return slots + positions % self._block_size

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
