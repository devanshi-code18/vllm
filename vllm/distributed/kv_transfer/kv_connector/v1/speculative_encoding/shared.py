# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Documents shared from a producer's GPU by CUDA IPC (same host only).

`share_document` keeps a document's K/V on the producer's GPU and publishes
only `meta.json`, which carries the CUDA IPC handle of the allocation that
holds one contiguous `[num_layers, 2, N, num_kv_heads, head_size]` tensor.
Each worker maps it on its own device (peer access, no context on the
producer's GPU), copies its heads device to device and unmaps it.

A worker holds a shared lock on the document's `meta.json` while it reads.
`close()` renames the document away first, so no new reader can start, then
takes the lock exclusively before it frees the memory. torch's own CUDA
tensor sharing is not used: it frees a shared block only when a reference
count that each mapping decrements is back to zero, which never happens when
several ranks or loads map one document.
"""

import contextlib
import ctypes
import fcntl
import json
import math
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

from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
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


def _start_time(pid: int | str) -> int:
    """A process's start time (field 22 of /proc/<pid>/stat); with the pid it
    names one process even if the pid is reused. 0 if it is not running."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            # The command name (field 2) may hold spaces; it ends at the last ")".
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return 0


def producer_gone(ipc: dict[str, Any]) -> bool:
    """Whether the process that shared a document has exited (known only
    when it runs in this process's pid namespace)."""
    if not ipc.get("pid_namespace") or ipc["pid_namespace"] != pid_namespace():
        return False
    return _start_time(int(ipc["pid"])) != ipc.get("start_time")


def shared_by_this_process(ipc: dict[str, Any]) -> bool:
    """CUDA IPC cannot map a handle in the process that exported it."""
    return ipc.get("pid") == os.getpid() and ipc.get("pid_namespace") == pid_namespace()


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

    def __init__(self, path: Path, kv: torch.Tensor, meta_fd: int) -> None:
        self.path = path
        self._kv: torch.Tensor | None = kv
        self._meta_fd = meta_fd
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
    if isinstance(kv, torch.Tensor) and kv.is_contiguous():
        shared = kv
    else:
        shared = torch.stack(layers)
    previous = torch.accelerator.current_device_index()
    torch.accelerator.set_device_index(shared.device.index)
    try:
        torch.accelerator.synchronize(shared.device)  # readers see its values
        base, _ = _allocation_range(shared.data_ptr())
        cudart = CudaRTLibrary()
        try:
            handle = cudart.cudaIpcGetMemHandle(ctypes.c_void_p(base))
        except RuntimeError as e:
            cudart.cudaGetLastError()  # else the next CUDA call reports it
            raise ValueError(
                "share_document: CUDA IPC cannot export this tensor's memory "
                "(PyTorch's expandable_segments allocator does not support it; "
                f"use the default allocator or write_document): {e}"
            ) from e
    finally:
        torch.accelerator.set_device_index(previous)
    meta["ipc"] = {
        "host": host_id(),
        "pid": os.getpid(),
        "pid_namespace": pid_namespace(),
        "start_time": _start_time("self"),
        "handle": base64.b64encode(bytes(handle.internal)).decode("ascii"),
        "offset": shared.data_ptr() - base,
        "shape": list(shared.shape),
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


@contextlib.contextmanager
def map_shared(
    ipc: dict[str, Any], dtype: torch.dtype, device: torch.device, num_layers: int
) -> Iterator[torch.Tensor]:
    """A shared document's K/V `[L, 2, N, num_kv_heads, head_size]`, mapped
    on `device` (the current device) until the block exits; work queued on
    the current stream finishes before it is unmapped.

    Raises:
        ValueError: The document was shared from another host, or its
            handle does not name at least `num_layers` layers inside the
            allocation it maps.

    """
    if ipc.get("host") != host_id():
        raise ValueError("the document was shared from another host")
    if shared_by_this_process(ipc):
        raise ValueError("the document was shared by this process")
    shape = [int(n) for n in ipc["shape"]]
    offset = int(ipc["offset"])
    raw = base64.b64decode(ipc["handle"])
    if len(shape) != 5 or shape[0] < num_layers or offset < 0 or len(raw) != 128:
        raise ValueError(f"shared K/V handle describes {shape} at {offset}")
    nbytes = math.prod(shape) * dtype.itemsize
    cudart = CudaRTLibrary()
    try:
        ptr = cudart.cudaIpcOpenMemHandle(cudaIpcMemHandle_t.from_buffer_copy(raw))
    except RuntimeError:
        # A handle that does not map leaves the error for this thread's next
        # CUDA call (a kernel launch of the step) to report: clear it.
        cudart.cudaGetLastError()
        raise
    try:
        mapped = int(ptr.value or 0)
        base, size = _allocation_range(mapped)
        if mapped + offset + nbytes > base + size:
            raise ValueError("shared K/V handle runs past its allocation")
        storage = torch._C._construct_storage_from_data_pointer(
            mapped + offset, device, nbytes
        )
        kv = torch.empty(0, dtype=dtype, device=device)
        yield kv.set_(storage, 0, shape, torch.empty(shape, device="meta").stride())
    finally:
        try:
            torch.accelerator.current_stream(device).synchronize()
        finally:
            cudart.cudaIpcCloseMemHandle(ptr)
