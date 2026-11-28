# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Documents shared from a producer's GPU by CUDA IPC (same host only).

`share_document` keeps a document's K/V on the producer's GPU and publishes
only `meta.json`, which carries the IPC handle of one head-major
`[num_layers, 2, num_kv_heads, N, head_size]` tensor. Each worker maps it on
its own device (peer access, no context on the producer's GPU), copies its
heads device to device and unmaps it.

A worker holds a shared lock on the document's `meta.json` while it reads.
`close()` renames the document away first, so no new reader can start, then
takes the lock exclusively before it frees the memory.
"""

import contextlib
import fcntl
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pybase64 as base64
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding.store import (  # noqa: E501
    check_document,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Tensors whose readers did not finish before `close()` gave up: never freed.
_abandoned: list[torch.Tensor] = []


def host_id() -> str:
    """This machine's boot id: equal only for processes on the same host."""
    with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
        return f.read().strip()


def _b64(data: bytes | None) -> str | None:
    return None if data is None else base64.b64encode(data).decode("ascii")


def _unb64(data: str | None) -> bytes | None:
    return None if data is None else base64.b64decode(data)


class SharedDocument:
    """A published document whose K/V lives in this process's GPU memory.

    Keep it until `close()`: the memory must outlive every reader.
    """

    def __init__(self, path: Path, kv: torch.Tensor, meta_fd: int) -> None:
        self.path = path
        self._kv: torch.Tensor | None = kv
        self._meta_fd = meta_fd

    def close(self, timeout: float = 60.0) -> None:
        """Unpublish the document, wait for readers, then free its memory.

        Args:
            timeout: Seconds to wait for readers; after that the memory is
                kept (and a warning logged) rather than freed under a reader.

        """
        if self._kv is None:
            return
        gone = Path(tempfile.mkdtemp(dir=self.path.parent, prefix=".closed."))
        os.rename(self.path, gone / "doc")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self._meta_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    logger.warning("%s: readers did not finish; keeping it", self.path)
                    _abandoned.append(self._kv)
                    break
                time.sleep(0.001)
        self._kv = None
        os.close(self._meta_fd)
        shutil.rmtree(gone, ignore_errors=True)


def share_document(
    store: str | os.PathLike,
    key: str,
    prompt_token_ids: Sequence[int],
    kv: torch.Tensor | Sequence[torch.Tensor],
    provenance: str = "",
) -> SharedDocument:
    """Publish a document whose K/V stays on this process's GPU.

    Servers on this host map it by CUDA IPC instead of reading a file; others
    find no K/V and prefill. Arguments as for `write_document`; `kv` must be
    on a CUDA device.

    Returns:
        The document; `close()` it to unpublish it and free its memory.

    """
    layers = list(kv.unbind(0)) if isinstance(kv, torch.Tensor) else list(kv)
    meta = check_document(key, prompt_token_ids, layers, provenance)
    if not layers[0].is_cuda:
        raise ValueError("share_document needs K/V on a CUDA device")
    # [layers, 2, heads, N, head_size]: each rank's heads are contiguous.
    shared = torch.stack([layer.permute(0, 2, 1, 3) for layer in layers])
    storage = shared.untyped_storage()
    (
        _,
        handle,
        size,
        offset,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    meta["ipc"] = {
        "host": host_id(),
        "shape": list(shared.shape),
        "stride": list(shared.stride()),
        "offset": shared.storage_offset(),
        "handle": _b64(handle),
        "size": size,
        "storage_offset": offset,
        "ref_counter_handle": _b64(ref_counter_handle),
        "ref_counter_offset": ref_counter_offset,
        "event_handle": _b64(event_handle),
        "event_sync_required": event_sync_required,
    }
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    final = store / key
    if final.exists():
        raise FileExistsError(
            f"{final} exists: publish a changed document under a new key"
        )
    tmp = Path(tempfile.mkdtemp(dir=store, prefix=f".{key}."))
    try:
        (tmp / "meta.json").write_text(json.dumps(meta))
        meta_fd = os.open(tmp / "meta.json", os.O_RDONLY)
        try:
            os.rename(tmp, final)
        except BaseException:
            os.close(meta_fd)
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return SharedDocument(final, shared, meta_fd)


@contextlib.contextmanager
def published_meta(folder: Path) -> Iterator[dict[str, Any]]:
    """The meta.json of the document in `folder`, held so that a shared
    document's memory stays allocated until the caller is done with it.

    Raises:
        FileNotFoundError: The document is not (or no longer) published.

    """
    fd = os.open(folder / "meta.json", os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        # Still the published one: `close()` unpublishes before it waits.
        if os.stat(folder / "meta.json").st_ino != os.fstat(fd).st_ino:
            raise FileNotFoundError(f"{folder} was unpublished")
        with os.fdopen(os.dup(fd), encoding="utf-8") as f:
            yield json.load(f)
    finally:
        os.close(fd)


def open_shared(ipc: dict[str, Any], dtype: torch.dtype, device: torch.device):
    """Map a shared document's K/V on `device` (no copy).

    Raises:
        ValueError: The document was shared from another host.

    """
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    if ipc.get("host") != host_id():
        raise ValueError("the document was shared from another host")
    return rebuild_cuda_tensor(
        torch.Tensor,
        torch.Size(ipc["shape"]),
        tuple(ipc["stride"]),
        int(ipc["offset"]),
        torch.storage.TypedStorage,
        dtype,
        device.index,
        _unb64(ipc["handle"]),
        int(ipc["size"]),
        int(ipc["storage_offset"]),
        False,
        _unb64(ipc["ref_counter_handle"]),
        int(ipc["ref_counter_offset"]),
        _unb64(ipc["event_handle"]),
        bool(ipc["event_sync_required"]),
    )
