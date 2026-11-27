# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reads one tensor-parallel rank's K/V of a stored document into pinned host
buffers, ahead of the host-to-device copies that consume them."""

import math
import os
from collections import deque
from collections.abc import Generator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from vllm.platforms import current_platform

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


def local_kv_heads(num_kv_heads: int, local_heads: int, rank: int, world: int):
    """The KV heads tensor-parallel rank `rank` of `world` holds: split across
    ranks, or replicated when there are fewer heads than ranks."""
    first = rank * num_kv_heads // world
    return range(first, first + local_heads)


def check_layer(entry: dict[str, Any], shape: list[int], dtype: torch.dtype) -> None:
    """Check that a safetensors layer entry holds `shape` (`[2, num_kv_heads,
    N, head_size]`) of `dtype` in exactly that many bytes.

    Raises:
        ValueError: The entry disagrees.

    """
    if (
        SAFETENSORS_DTYPES.get(str(entry.get("dtype"))) != dtype
        or entry.get("shape") != shape
    ):
        raise ValueError(
            f"layer is {entry.get('dtype')} {entry.get('shape')}, "
            f"expected {dtype} {shape}"
        )
    begin, end = entry["data_offsets"]
    if end - begin != math.prod(shape) * dtype.itemsize:
        raise ValueError(f"layer holds {end - begin} bytes, expected {shape}")


def layer_ranges(
    entry: dict[str, Any], data_start: int, heads: range, start: int, end: int
) -> list[tuple[int, int]]:
    """Byte ranges (offset, length) of `heads` of K and V for tokens
    `[start, end)` of one checked safetensors layer `[2, num_kv_heads, N,
    head_size]`."""
    _, num_kv_heads, num_tokens, head_size = entry["shape"]
    row = head_size * SAFETENSORS_DTYPES[entry["dtype"]].itemsize
    base = data_start + entry["data_offsets"][0]
    return [
        (
            base + ((kv * num_kv_heads + h) * num_tokens + start) * row,
            (end - start) * row,
        )
        for kv in (0, 1)
        for h in heads
    ]


def split_ranges(
    ranges: Sequence[tuple[int, int]], chunk_bytes: int
) -> list[list[tuple[int, int]]]:
    """`ranges`, merged where adjacent, cut into pieces of at most
    `chunk_bytes`."""
    merged: list[tuple[int, int]] = []
    for offset, length in ranges:
        if merged and merged[-1][0] + merged[-1][1] == offset:
            merged[-1] = (merged[-1][0], merged[-1][1] + length)
        else:
            merged.append((offset, length))
    pieces: list[list[tuple[int, int]]] = [[]]
    room = chunk_bytes
    for offset, length in merged:
        while length:
            if room == 0:
                pieces.append([])
                room = chunk_bytes
            take = min(length, room)
            pieces[-1].append((offset, take))
            offset, length, room = offset + take, length - take, room - take
    return pieces


class StagedReader:
    """Fills `depth` reusable (pinned) host buffers from a file on threads."""

    def __init__(self, chunk_bytes: int = 32 << 20, depth: int = 4):
        self.chunk_bytes = chunk_bytes
        self._pool = ThreadPoolExecutor(depth, thread_name_prefix="se-read")
        self._depth = depth
        self._pin = current_platform.is_pin_memory_available()
        # Free buffers, each with the event of the copy last queued from it.
        self._free: list[tuple[torch.Tensor, torch.Event | None]] = []

    def allocate(self) -> None:
        # Also replaces buffers a failed read left in flight.
        while len(self._free) < self._depth:
            buffer = torch.empty(
                self.chunk_bytes, dtype=torch.uint8, pin_memory=self._pin
            )
            self._free.append((buffer, None))

    @staticmethod
    def _fill(fd: int, ranges, buffer: torch.Tensor, copied) -> torch.Tensor:
        if copied is not None:
            copied.synchronize()
        view, position = buffer.numpy(), 0
        for offset, length in ranges:
            done = 0
            while done < length:
                chunk = memoryview(view[position + done : position + length])
                read = os.preadv(fd, [chunk], offset + done)
                if read == 0:
                    raise EOFError(f"short read at offset {offset + done}")
                done += read
            position += length
        return buffer[:position]

    def read(
        self, fd: int, pieces: Sequence[list[tuple[int, int]]]
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yields `(buffer, bytes)` per piece of at most `chunk_bytes`, in
        order. Pass each buffer to `release` right after queueing the copy out
        of it."""
        self.allocate()
        remaining = iter(pieces)
        pending: deque[tuple[torch.Tensor, Future]] = deque()

        def submit() -> None:
            ranges = next(remaining, None)
            if ranges is not None:
                buffer, copied = self._free.pop()
                future = self._pool.submit(self._fill, fd, ranges, buffer, copied)
                pending.append((buffer, future))

        for _ in range(self._depth):
            submit()
        while pending:
            buffer, future = pending.popleft()
            yield buffer, future.result()
            submit()

    def release(self, buffer: torch.Tensor) -> None:
        copied = None
        if self._pin:
            copied = torch.Event()
            copied.record()
        self._free.append((buffer, copied))
