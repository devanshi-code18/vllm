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

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVProvenance,
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
    published_meta,
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
    layer_inputs: bool = False


@dataclass
class SpeculativeEncodingConnectorMetadata(KVConnectorMetadata):
    loads: list[DocumentLoad] = field(default_factory=list)


class SpeculativeEncodingConnector(KVConnectorBase_V1):
    """Loads a request's document tokens from a store of translated KV.

    `kv_connector_extra_config`:
        store: The store directory (required), readable by the scheduler and
            every worker.
        layer_inputs_chunk_tokens: Tokens per write of a layer-inputs
            document (default 4096): bounds the GPU memory a load holds.

    A document holds either the served model's K/V or its per-layer inputs;
    the latter are written by the model itself (its own, possibly quantized,
    K/V projections) and need `register_kv_from_layer_inputs_writer`.

    Translated KV is approximate, so requests that load a document declare its
    provenance: their blocks are cached apart from natively prefilled ones.
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
        self._input_geometry = (
            model_config.get_total_num_hidden_layers(),
            model_config.get_hidden_size(),
        )
        self._chunk_tokens = int(
            self._kv_transfer_config.get_from_extra_config(
                "layer_inputs_chunk_tokens", 4096
            )
        )
        self._writer: Any = None
        # Scheduler side.
        self._documents: OrderedDict[str, StoredDocument] = OrderedDict()
        self._matches: dict[str, StoredDocument | None] = {}
        self._pending: dict[str, DocumentLoad] = {}
        self._warned: set[str] = set()
        # Worker side.
        self._dtype = model_config.dtype
        self._reader = StagedReader()
        self._failed_block_ids: set[int] = set()

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
                self._store
                / key
                / ("meta.json" if document.shared_host else document.file)
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
        if document.shared_host and document.shared_host != host_id():
            self._skip("host", "document %r is shared from another host", key)
            return None
        if document.layer_inputs:
            if (document.num_layers, document.hidden_size) != self._input_geometry:
                self._skip(
                    "geometry",
                    "document %r has inputs of %d layers of width %d; this model "
                    "has %d of %d",
                    key,
                    document.num_layers,
                    document.hidden_size,
                    *self._input_geometry,
                )
                return None
        elif (document.num_kv_heads, document.head_size) != self._geometry:
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
            else:
                document = found
        self._matches[request.request_id] = document
        return document

    def get_kv_provenance(self, request: "Request") -> KVProvenance | None:
        document = self._match(request)
        if document is None:
            return None
        return KVProvenance(
            f"speculative_encoding:{document.provenance}:{document.key}"
        )

    def _num_tokens(self, request: "Request", document: StoredDocument) -> int:
        # The last prompt token is computed to sample the first output token.
        return min(document.num_tokens, request.num_prompt_tokens - 1)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        document = self._match(request)
        if document is None:
            return 0, False
        return max(0, self._num_tokens(request, document) - num_computed_tokens), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        if num_external_tokens <= 0:
            return
        document = self._matches[request.request_id]
        assert document is not None
        end = self._num_tokens(request, document)
        self._pending[request.request_id] = DocumentLoad(
            request.request_id,
            document.key,
            document.num_tokens,
            document.dtype,
            end - num_external_tokens,
            end,
            [],
            document.layer_inputs,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = SpeculativeEncodingConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            load = self._pending.pop(new_req.req_id, None)
            if load is not None:
                load.block_ids = list(new_req.block_ids[0])
                meta.loads.append(load)
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

    def register_kv_from_layer_inputs_writer(self, writer: Any) -> None:
        self._writer = writer

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SpeculativeEncodingConnectorMetadata)
        if not metadata.loads:
            return
        from vllm.model_executor.layers.attention.attention import Attention

        layers = {
            name: layer
            for name, layer in forward_context.no_compile_layers.items()
            if isinstance(layer, Attention)
            and layer.kv_sharing_target_layer_name is None
            and layer.kv_cache.numel() > 0
        }
        for load in metadata.loads:
            started = time.perf_counter()
            try:
                # A deliberate bulk transfer: the step waits for its loads.
                with gpu_sync_allowed():
                    self._load(load, layers)
            except Exception:
                logger.exception(
                    "loading document %r for request %s failed",
                    load.key,
                    load.request_id,
                )
                # Only the blocks this load writes: earlier ones are prefix
                # cache hits that other requests may share.
                first = load.start // self._block_size
                last = -(-load.end // self._block_size)
                self._failed_block_ids.update(load.block_ids[first:last])
                continue
            logger.debug(
                "loaded %d tokens of document %r for request %s in %.3f s",
                load.end - load.start,
                load.key,
                load.request_id,
                time.perf_counter() - started,
            )

    def _load(self, load: DocumentLoad, layers: dict[str, Any]) -> None:
        folder = self._store / load.key
        with published_meta(folder) as meta:
            # Read again: a shared document may have been replaced since.
            if isinstance(meta.get("ipc"), dict):
                self._write_shared(meta["ipc"], load, layers)
                return
        file = "layer_inputs.safetensors" if load.layer_inputs else "kv.safetensors"
        fd = os.open(folder / file, os.O_RDONLY)
        try:
            size = int.from_bytes(os.pread(fd, 8, 0), "little")
            header = json.loads(os.pread(fd, size, 8))
            if load.layer_inputs:
                self._write_from_layer_inputs(fd, 8 + size, header, load)
            else:
                self._write_layers(fd, 8 + size, header, load, layers)
        finally:
            os.close(fd)

    def _write_from_layer_inputs(
        self, fd: int, data_start: int, header: dict[str, Any], load: DocumentLoad
    ) -> None:
        if self._writer is None:
            raise RuntimeError(
                "document holds per-layer inputs, but the served model cannot "
                "write its KV cache from them in this configuration"
            )
        num_layers, hidden = self._input_geometry
        shape = [load.num_tokens, hidden]
        entries = [header[f"layers.{i}"] for i in range(num_layers)]
        for entry in entries:
            check_layer(entry, shape, load.dtype)
        row = hidden * load.dtype.itemsize
        device = self._writer.device
        for first in range(load.start, load.end, self._chunk_tokens):
            last = min(load.end, first + self._chunk_tokens)
            ranges = [
                (
                    data_start + entry["data_offsets"][0] + first * row,
                    (last - first) * row,
                )
                for entry in entries
            ]
            pieces = split_ranges(ranges, self._reader.chunk_bytes)
            flat = torch.empty(
                num_layers * (last - first) * row, dtype=torch.uint8, device=device
            )
            position = 0
            staged = self._reader.read(fd, pieces)
            try:
                for buffer, piece in staged:
                    flat[position : position + piece.numel()].copy_(
                        piece, non_blocking=True
                    )
                    self._reader.release(buffer)
                    position += piece.numel()
            finally:
                staged.close()
            inputs = flat.view(load.dtype).view(num_layers, last - first, hidden)
            inputs = inputs.to(self._writer.input_dtype)
            self._writer.write(list(inputs.unbind(0)), first, [load.block_ids])

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
        shape = [2, num_kv_heads, load.num_tokens, head_size]
        if ipc.get("shape", [])[1:] != shape:
            raise ValueError(
                f"shared K/V is {ipc.get('shape')}, expected [L, *{shape}]"
            )
        if not layers:
            return
        device = next(iter(layers.values())).kv_cache.device
        # Mapped on this rank's own device: reads cross NVLink/PCIe peer access,
        # and this process makes no context on the producer's GPU.
        shared = open_shared(ipc, load.dtype, device)
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
        slots = self._slots(load).to(device, non_blocking=True)
        for name, layer in layers.items():
            heads = local_kv_heads(num_kv_heads, layer.num_kv_heads, rank, world)
            kv = shared[
                extract_layer_index(name),
                :,
                heads.start : heads.stop,
                load.start : load.end,
            ].to(self._dtype)
            key, value = kv.transpose(1, 2)
            layer.impl.do_kv_cache_update(
                layer, key.contiguous(), value.contiguous(), layer.kv_cache, slots
            )
        # The producer may free the memory once the lease is gone.
        torch.accelerator.synchronize(device)

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
