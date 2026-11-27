# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PoC: worker half of the pinned-CPU state image tier (see
vllm/v1/core/state_image_tier.py). Copies whole KV blocks (every layer's page
of a block id) between the GPU KV allocation and pinned host slots."""

import time

import torch

from vllm.logger import init_logger
from vllm.v1.core.state_image_tier import sit_cpu_bytes

logger = init_logger(__name__)

_KV_BUF: torch.Tensor | None = None


def record_kv_buffer(buf: torch.Tensor) -> None:
    global _KV_BUF
    _KV_BUF = buf


class StateTierWorker:
    def __init__(self, kv_cache_config) -> None:
        from vllm.v1.worker.utils import layer_spec_for_tensor

        assert _KV_BUF is not None
        self.gpu = _KV_BUF.view(torch.int8).reshape(-1)
        num_blocks = kv_cache_config.num_blocks
        regions: dict[tuple[int, int], int] = {}
        for t in kv_cache_config.kv_cache_tensors:
            _, spec = layer_spec_for_tensor(kv_cache_config, t)
            for li in range(len(t.layers)):
                key = (t.offset + li * t.layer_stride, t.block_stride)
                regions[key] = max(regions.get(key, 0), spec.page_size_bytes)
        regs = sorted((start, bs, page) for (start, bs), page in regions.items())
        bytes_per_block = kv_cache_config.kv_cache_tensors[0].size // num_blocks
        total = sum(p for _, _, p in regs)
        strides = {bs for _, bs, _ in regs}
        contiguous = all(
            regs[i][0] + regs[i][2] == regs[i + 1][0] for i in range(len(regs) - 1)
        )
        if len(strides) == 1 and contiguous and total == next(iter(strides)):
            regs = [(regs[0][0], total, total)]
        assert total <= bytes_per_block, (total, bytes_per_block)
        self.regions = []
        off = 0
        for start, bs, page in regs:
            self.regions.append((start, bs, page, off))
            off += page
        self.slot_bytes = off
        num_slots = sit_cpu_bytes() // bytes_per_block
        t0 = time.monotonic()
        self.cpu = torch.empty(
            (num_slots, self.slot_bytes), dtype=torch.int8, pin_memory=True
        )
        logger.info(
            "StateTierWorker: %d slots x %d B pinned in %.1f s, %d copy region(s) "
            "per block (block-major=%s)",
            num_slots,
            self.slot_bytes,
            time.monotonic() - t0,
            len(self.regions),
            len(self.regions) == 1,
        )
        self.bytes_stored = 0
        self.bytes_loaded = 0
        # (start, end, nbytes, is_store) CUDA events not yet read back
        self._pending: list = []
        self.ms = {True: 0.0, False: 0.0}
        self.nbytes = {True: 0, False: 0}
        self._last_log = time.monotonic()

    def _poll(self) -> None:
        keep = []
        for st, en, n, is_store in self._pending:
            if en.query():
                self.ms[is_store] += st.elapsed_time(en)
                self.nbytes[is_store] += n
            else:
                keep.append((st, en, n, is_store))
        self._pending = keep
        now = time.monotonic()
        if now - self._last_log > 30:
            self._last_log = now

            def bw(k):
                return self.nbytes[k] / max(self.ms[k], 1e-9) / 1e6

            logger.info(
                "StateTierWorker: stored %.1f GB in %.2f s (%.1f GB/s), "
                "loaded %.1f GB in %.2f s (%.1f GB/s)",
                self.nbytes[True] / 1e9,
                self.ms[True] / 1e3,
                bw(True),
                self.nbytes[False] / 1e9,
                self.ms[False] / 1e3,
                bw(False),
            )

    def _copy(self, pairs, store: bool) -> None:
        self._poll()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for a, b in pairs:
            blk, slot = (a, b) if store else (b, a)
            for start, bs, page, off in self.regions:
                g = self.gpu[start + blk * bs : start + blk * bs + page]
                c = self.cpu[slot, off : off + page]
                if store:
                    c.copy_(g, non_blocking=True)
                else:
                    g.copy_(c, non_blocking=True)
        n = len(pairs) * self.slot_bytes
        en.record()
        self._pending.append((st, en, n, store))
        if store:
            self.bytes_stored += n
        else:
            self.bytes_loaded += n

    def run_stores(self, ops) -> None:
        if ops is not None and ops.stores:
            self._copy(ops.stores, store=True)

    def run_loads(self, ops) -> None:
        if ops is not None and ops.loads:
            self._copy(ops.loads, store=False)
