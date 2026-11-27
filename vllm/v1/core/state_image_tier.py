# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PoC: a pinned-CPU tier for evicted prefix-cache blocks ("state image tier").

Research prototype, not for upstream. Enabled with VLLM_SIT_CPU_GB=<GiB>.

- Write-back on GPU eviction: when the block pool hands out a cached block,
  its bytes are copied to a CPU slot keyed by the block's prefix hash
  (BlockHashWithGroupId). Nothing is copied while blocks stay on the GPU.
- Recurrent / window state blocks are state images: a Mamba block holds the
  whole state at one boundary, a sliding-window block one window piece. With
  VLLM_SIT_POLICY=image (default) a Mamba state the manager deprioritized as
  superseded (an older turn of the same conversation) is dropped, not
  stored, so a conversation keeps O(1) state images plus its O(n) KV blocks.
- Restore on admission: before the prefix-cache lookup of a new request,
  every CPU-resident block on the request's hash chain is copied back into a
  fresh GPU block and registered as cached; the normal lookup then hits it.
  Copies are ordered on the compute stream before the step's forward pass
  (stores, then the new-block zeroing, then loads).
- Inclusive: a restored block keeps its CPU slot, so evicting it again copies
  nothing.
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    get_group_id,
    make_block_hash_with_group_id,
)

logger = init_logger(__name__)

KIND_FULL, KIND_SWA, KIND_STATE = "full", "swa", "state"


def sit_cpu_bytes() -> int:
    gb = float(os.environ.get("VLLM_SIT_CPU_GB", "0") or 0)
    return int(gb * (1 << 30))


@dataclass
class StateTierOps:
    """Copies for one step: (gpu_block, cpu_slot) stores, (cpu_slot, gpu_block)
    loads. Stores run first."""

    stores: list[tuple[int, int]] = field(default_factory=list)
    loads: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class _Slot:
    keys: list[BlockHashWithGroupId]
    num_tokens: int | None


class StateImageTier:
    def __init__(self, num_slots: int, bytes_per_block: int, group_kinds: list[str]):
        self.num_slots = num_slots
        self.bytes_per_block = bytes_per_block
        self.group_kinds = group_kinds
        self.policy = os.environ.get("VLLM_SIT_POLICY", "image")
        # Window blocks restored before a sliding-window hit boundary.
        self.swa_blocks = int(os.environ.get("VLLM_SIT_SWA_BLOCKS", "16"))
        self.debug = os.environ.get("VLLM_SIT_DEBUG", "0") == "1"
        # Drops from the Mamba manager's own superseded-state chain act per
        # group and can split an image; promote() drops whole images instead.
        self.hook_drop = os.environ.get("VLLM_SIT_HOOK_DROP", "0") == "1"
        self.free_slots = list(range(num_slots - 1, -1, -1))
        # key -> slot, LRU order (oldest first)
        self.slot_of: OrderedDict[BlockHashWithGroupId, int] = OrderedDict()
        self.slots: dict[int, _Slot] = {}
        # gpu block id -> (slot, key) for blocks whose bytes equal a CPU slot
        self.clean: dict[int, tuple[int, BlockHashWithGroupId]] = {}
        self.superseded: set[int] = set()
        self.ops = StateTierOps()
        self.stats = dict(
            stores=0,
            loads=0,
            skipped_superseded=0,
            dropped_superseded=0,
            skipped_clean=0,
            skipped_no_room=0,
            cpu_evicted=0,
        )
        self.stats_by_kind = {k: [0, 0] for k in (KIND_FULL, KIND_SWA, KIND_STATE)}
        self._last_log = time.monotonic()
        logger.info(
            "StateImageTier: %d CPU slots x %d B = %.1f GiB, policy=%s, groups=%s",
            num_slots,
            bytes_per_block,
            num_slots * bytes_per_block / (1 << 30),
            self.policy,
            group_kinds,
        )

    # ---- slots -------------------------------------------------------------
    def _drop_slot(self, slot: int) -> None:
        s = self.slots.pop(slot)
        for k in s.keys:
            if self.slot_of.get(k) == slot:
                del self.slot_of[k]
        self.free_slots.append(slot)

    def _alloc_slot(self) -> int:
        if not self.free_slots:
            # Stores run before loads in a step: never reuse a slot that a
            # pending load of this step still reads.
            pending = {slot for slot, _ in self.ops.loads}
            victim = next(s for s in self.slot_of.values() if s not in pending)
            self._drop_slot(victim)
            self.stats["cpu_evicted"] += 1
        return self.free_slots.pop()

    # ---- block pool hooks --------------------------------------------------
    def on_deprioritize(self, block: KVCacheBlock) -> None:
        self.superseded.add(block.block_id)

    def on_touch(self, block: KVCacheBlock) -> None:
        self.superseded.discard(block.block_id)

    def on_evict(self, block: KVCacheBlock, extra_keys) -> None:
        """Called before the pool drops `block`'s hashes to reuse it."""
        key = block.block_hash
        if key is None:
            return
        bid = block.block_id
        was_superseded = bid in self.superseded
        self.superseded.discard(bid)
        clean = self.clean.pop(bid, None)
        if (
            clean is not None
            and clean[1] == key
            and self.slot_of.get(clean[1]) == clean[0]
        ):
            self.slot_of.move_to_end(clean[1])
            self.stats["skipped_clean"] += 1
            return
        kind = self.group_kinds[get_group_id(key)]
        if kind == KIND_STATE and was_superseded and self.policy == "image":
            self.stats["skipped_superseded"] += 1
            return
        if key in self.slot_of:
            self.slot_of.move_to_end(key)
            return
        slot = self._alloc_slot()
        keys = [key]
        self.slots[slot] = _Slot(keys=keys, num_tokens=block.block_hash_num_tokens)
        self.slot_of[key] = slot
        self.ops.stores.append((bid, slot))
        self.stats["stores"] += 1
        self.stats_by_kind[kind][0] += 1

    def drop(self, block_hash, group_ids) -> None:
        """Forget a superseded state image (an older turn's state)."""
        if self.policy != "image" or self.ops.loads or not self.hook_drop:
            return
        for gid in group_ids:
            slot = self.slot_of.get(make_block_hash_with_group_id(block_hash, gid))
            if slot is not None:
                self._drop_slot(slot)
                self.stats["dropped_superseded"] += 1

    # ---- admission ---------------------------------------------------------
    def promote(self, request, block_pool, group_block_sizes: list[int]) -> int:
        """Restore CPU-resident blocks on `request`'s hash chain. Returns the
        number of blocks restored."""
        if not self.slot_of or not request.block_hashes:
            return 0
        hashes = request.block_hashes
        hbs = block_pool.hash_block_size
        cached = block_pool.cached_block_hash_to_block

        # GPU blocks the planned hit relies on; pinned while restoring so the
        # allocation below cannot evict them.
        keep: dict[int, KVCacheBlock] = {}

        def where(key, pin=True):
            blk = cached.get_one_block(key)
            if blk is not None:
                if pin:
                    keep[blk.block_id] = blk
                return "gpu"
            return self.slot_of.get(key)

        todo: list[tuple[BlockHashWithGroupId, int]] = []
        superseded: list[BlockHashWithGroupId] = []
        # Full attention: the prefix covered by GPU or CPU blocks bounds the hit.
        reach = len(hashes) * hbs
        for gid, kind in enumerate(self.group_kinds):
            if kind != KIND_FULL:
                continue
            bs = group_block_sizes[gid]
            scale = bs // hbs
            n = 0
            for j in range(len(hashes) // scale):
                key = make_block_hash_with_group_id(hashes[(j + 1) * scale - 1], gid)
                w = where(key)
                if w is None:
                    break
                if w != "gpu":
                    todo.append((key, w))
                n = j + 1
            reach = min(reach, n * bs)
        # Window groups: the window blocks before the latest boundary.
        for gid, kind in enumerate(self.group_kinds):
            if kind != KIND_SWA:
                continue
            bs = group_block_sizes[gid]
            scale = bs // hbs
            last = min(reach, len(hashes) * hbs) // bs - 1
            for j in range(last, max(0, last - self.swa_blocks) - 1, -1):
                key = make_block_hash_with_group_id(hashes[(j + 1) * scale - 1], gid)
                w = where(key)
                if w is not None and w != "gpu":
                    todo.append((key, w))
        # State groups: an image is the blocks of every state group at one
        # boundary. Restore the latest complete image at or below the reach.
        state_gids = [g for g, k in enumerate(self.group_kinds) if k == KIND_STATE]
        if state_gids:
            last = min(reach, len(hashes) * hbs) // hbs - 1
            complete = []
            for j in range(last, -1, -1):
                ws = []
                for gid in state_gids:
                    key = make_block_hash_with_group_id(hashes[j], gid)
                    w = where(key, pin=False)
                    if w is None:
                        break
                    ws.append((key, w))
                else:
                    complete.append(ws)
            if complete:
                todo.extend((key, w) for key, w in complete[0] if w != "gpu")
                for key, w in complete[0]:
                    if w == "gpu":
                        where(key)
                # Images between the conversation's first one (usually the
                # shared system prompt) and its latest belong to older turns.
                if self.policy == "image":
                    for ws in complete[1:-1]:
                        superseded.extend(key for key, w in ws if w != "gpu")
        pending = {slot for slot, _ in self.ops.loads}
        for key in superseded:
            slot = self.slot_of.get(key)
            if slot is not None and slot not in pending:
                self._drop_slot(slot)
                self.stats["dropped_superseded"] += 1
        free = block_pool.get_num_free_blocks()
        if self.debug:
            nk: dict[str, int] = {}
            for key, _ in todo:
                k = self.group_kinds[get_group_id(key)]
                nk[k] = nk.get(k, 0) + 1
            logger.info(
                "SIT promote %s: %d hashes, reach %d tok, restore %s, free %d",
                request.request_id[-12:],
                len(hashes),
                reach,
                nk,
                free,
            )
        if not todo:
            return 0
        # A request that cannot get these blocks could not recompute them
        # either, so restore whenever they fit.
        if len(todo) > free:
            self.stats["skipped_no_room"] += 1
            return 0
        for key, _ in todo:
            self.slot_of.move_to_end(key)
        pinned = list(keep.values())
        block_pool.touch(pinned)
        if len(todo) > block_pool.get_num_free_blocks():
            block_pool.free_blocks(pinned)
            self.stats["skipped_no_room"] += 1
            return 0
        new_blocks = block_pool.get_new_blocks(len(todo))
        for blk, (key, slot) in zip(new_blocks, todo):
            # get_new_blocks may have evicted CPU slots of other keys; ours were
            # looked up before, so re-check.
            if self.slot_of.get(key) != slot:
                continue
            block_pool._insert_block_hash(key, blk, self.slots[slot].num_tokens)
            self.slot_of.move_to_end(key)
            self.clean[blk.block_id] = (slot, key)
            self.ops.loads.append((slot, blk.block_id))
            self.stats["loads"] += 1
            self.stats_by_kind[self.group_kinds[get_group_id(key)]][1] += 1
        block_pool.free_blocks(new_blocks)
        block_pool.free_blocks(pinned)
        return len(todo)

    def take_ops(self) -> StateTierOps | None:
        now = time.monotonic()
        if now - self._last_log > 30:
            self._last_log = now
            logger.info(
                "StateImageTier: used %d/%d slots (%.1f GiB), %s, by kind "
                "[stores, loads] %s",
                len(self.slots),
                self.num_slots,
                len(self.slots) * self.bytes_per_block / (1 << 30),
                self.stats,
                self.stats_by_kind,
            )
        if not self.ops.stores and not self.ops.loads:
            return None
        ops, self.ops = self.ops, StateTierOps()
        return ops


def maybe_create_tier(kv_cache_config, block_pool) -> StateImageTier | None:
    cpu_bytes = sit_cpu_bytes()
    if cpu_bytes <= 0 or not kv_cache_config.kv_cache_tensors:
        return None
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
        SlidingWindowManager,
    )

    kinds = []
    for g in kv_cache_config.kv_cache_groups:
        cls = g.manager_cls
        if issubclass(cls, MambaManager):
            kinds.append(KIND_STATE)
        elif issubclass(cls, SlidingWindowManager):
            kinds.append(KIND_SWA)
        elif issubclass(cls, FullAttentionManager):
            kinds.append(KIND_FULL)
        else:
            raise NotImplementedError(f"StateImageTier: {cls.__name__}")
    bytes_per_block = (
        kv_cache_config.kv_cache_tensors[0].size // kv_cache_config.num_blocks
    )
    tier = StateImageTier(cpu_bytes // bytes_per_block, bytes_per_block, kinds)
    block_pool.sit = tier
    return tier
