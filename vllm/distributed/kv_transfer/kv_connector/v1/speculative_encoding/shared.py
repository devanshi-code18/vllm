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
import ctypes
import fcntl
import json
import math
import mmap
import os
import shutil
import tempfile
import time
import weakref
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
    """This machine's boot id: equal only for processes on the same host
    (empty where there is none; CUDA IPC is Linux-only)."""
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return ""


def pid_namespace() -> int:
    """This process's pid namespace (0 where there is none)."""
    try:
        return os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        return 0


def producer_gone(ipc: dict[str, Any]) -> bool:
    """Whether the process that shared a document has exited (known only
    when it runs in this process's pid namespace)."""
    if not ipc.get("pid_namespace") or ipc["pid_namespace"] != pid_namespace():
        return False
    return not os.path.exists(f"/proc/{int(ipc['pid'])}")


def _b64(data: bytes | None) -> str | None:
    return None if data is None else base64.b64encode(data).decode("ascii")


def _unb64(data: str | None) -> bytes | None:
    return None if data is None else base64.b64decode(data)


def _unpublish(path: Path) -> Path:
    """Rename a published document away (no new reader can start) and return
    where it went."""
    gone = Path(tempfile.mkdtemp(dir=path.parent, prefix=".closed."))
    os.rename(path, gone / "doc")
    return gone


def _abandon(path: Path, kv: torch.Tensor, meta_fd: int) -> None:
    """A SharedDocument dropped without `close()`: unpublish it and keep its
    memory for good, since a reader may still be copying from it."""
    with contextlib.suppress(OSError):
        _unpublish(path)
    _abandoned.append(kv)
    os.close(meta_fd)


class SharedDocument:
    """A published document whose K/V lives in this process's GPU memory.

    Keep it until `close()`: the memory must outlive every reader. Dropping
    it without `close()` unpublishes the document and leaks the memory.
    """

    def __init__(
        self,
        path: Path,
        kv: torch.Tensor,
        meta_fd: int,
        ref_counter: tuple[bytes, int] | None = None,
    ) -> None:
        self.path = path
        self._kv: torch.Tensor | None = kv
        self._meta_fd = meta_fd
        self._ref_counter = ref_counter
        self._finalizer = weakref.finalize(self, _abandon, path, kv, meta_fd)

    def close(self, timeout: float = 60.0) -> None:
        """Unpublish the document, wait for readers, then free its memory.

        Args:
            timeout: Seconds to wait for readers; after that the memory is
                kept (and a warning logged) rather than freed under a reader.

        """
        if self._kv is None:
            return
        self._finalizer.detach()
        gone = _unpublish(self.path)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self._meta_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() < deadline:
                    time.sleep(0.001)
                    continue
                logger.warning("%s: readers did not finish; keeping it", self.path)
                _abandoned.append(self._kv)
            else:
                if self._ref_counter is not None:
                    _clear_ref_counter(*self._ref_counter)
            break
        self._kv = None
        os.close(self._meta_fd)
        shutil.rmtree(gone, ignore_errors=True)


def _clear_ref_counter(handle: bytes, offset: int) -> None:
    """Zero torch's IPC reference counter of a shared allocation.

    torch frees a shared CUDA block only once its (unsigned) counter, set to
    1 when shared, is back to 0; every reader that maps and drops it counts
    down once, so with several readers or loads it wraps and the block would
    never be freed. `close()` holds the lock no reader holds, so it can.
    """
    path = Path("/dev/shm") / handle.decode().lstrip("/")
    try:
        with open(path, "r+b") as f, mmap.mmap(f.fileno(), 0) as counters:
            counters[offset * 8 : offset * 8 + 8] = bytes(8)
    except OSError:
        logger.warning("could not reset the IPC counter %s; memory may leak", path)


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
    on a CUDA device. One contiguous `[num_layers, 2, N, num_kv_heads,
    head_size]` tensor is shared as is (do not change it while published);
    per-layer tensors are stacked into one copy.

    Returns:
        The document; `close()` it to unpublish it and free its memory.

    """
    layers = list(kv.unbind(0)) if isinstance(kv, torch.Tensor) else list(kv)
    meta = check_document(key, prompt_token_ids, layers, provenance)
    if not layers[0].is_cuda:
        raise ValueError("share_document needs K/V on a CUDA device")
    if (
        isinstance(kv, torch.Tensor)
        and kv.is_contiguous()
        and kv.storage_offset() == 0
        and kv.untyped_storage().nbytes() == kv.numel() * kv.element_size()
    ):
        shared = kv
    else:
        shared = torch.stack(layers)
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
        "pid": os.getpid(),
        "pid_namespace": pid_namespace(),
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
    return SharedDocument(
        final, shared, meta_fd, (ref_counter_handle, ref_counter_offset)
    )


@contextlib.contextmanager
def published_meta(folder: Path) -> Iterator[tuple[dict[str, Any], int]]:
    """The meta.json of the document in `folder` and a descriptor of its
    directory (for its other files), held so that a shared document's memory
    stays allocated until the caller is done with it.

    Raises:
        FileNotFoundError: The document is not (or no longer) published.

    """
    folder_fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd = os.open("meta.json", os.O_RDONLY, dir_fd=folder_fd)
    except BaseException:
        os.close(folder_fd)
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        # Still the published one: `close()` unpublishes before it waits.
        if os.stat(folder / "meta.json").st_ino != os.fstat(fd).st_ino:
            raise FileNotFoundError(f"{folder} was unpublished")
        with os.fdopen(os.dup(fd), encoding="utf-8") as f:
            meta = json.load(f)
        if "ipc" in meta and os.fstat(fd).st_uid != os.getuid():
            raise PermissionError(f"{folder} is shared by another user")
        yield meta, folder_fd
    finally:
        os.close(fd)
        os.close(folder_fd)


def _allocation_range(ptr: int) -> tuple[int, int]:
    """The device allocation that contains `ptr`: (base, size)."""
    libcuda = ctypes.CDLL("libcuda.so.1")
    base, size = ctypes.c_uint64(), ctypes.c_size_t()
    result = libcuda.cuMemGetAddressRange_v2(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(ptr)
    )
    if result != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed (CUresult {result})")
    return base.value, size.value


def open_shared(
    ipc: dict[str, Any], dtype: torch.dtype, device: torch.device, num_layers: int
) -> torch.Tensor:
    """Map a shared document's K/V `[L, 2, N, num_kv_heads, head_size]` on
    `device` (no copy), after checking that the handle describes exactly one
    contiguous tensor of at least `num_layers` layers inside its allocation.

    Raises:
        ValueError: The document was shared from another host, or its
            handle does not describe such a tensor.

    """
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    if ipc.get("host") != host_id():
        raise ValueError("the document was shared from another host")
    shape = [int(n) for n in ipc["shape"]]
    contiguous = list(torch.empty(shape, device="meta").stride())
    nbytes = math.prod(shape) * dtype.itemsize
    if (
        len(shape) != 5
        or shape[0] < num_layers
        or list(ipc["stride"]) != contiguous
        or int(ipc["offset"]) != 0
        or int(ipc["size"]) != nbytes
    ):
        raise ValueError(f"shared K/V handle describes {ipc['shape']} {ipc['stride']}")
    kv = rebuild_cuda_tensor(
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
    base, size = _allocation_range(kv.data_ptr())
    if kv.data_ptr() + nbytes > base + size:
        raise ValueError("shared K/V handle runs past its allocation")
    return kv
