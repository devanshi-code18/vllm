# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`Inkling2ConvStateManager`: a prefill chunk backs only the conv-state rows a
later step reads (its tail, and the tail of every retain stride)."""

import torch

from tests.v1.core.utils import new_kv_cache_group
from vllm.models.inkling2.nvidia.sconv import Inkling2ConvStateManager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import AttentionSpec

BLOCK, STRIDE, WINDOW = 16, 64, 4


def new_manager(
    num_gpu_blocks: int = 128, enable_caching: bool = True
) -> tuple[Inkling2ConvStateManager, BlockPool]:
    layer = Inkling2ConvStateManager.layer_config(
        AttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
        ),
        sliding_window=WINDOW,
        retain_stride=STRIDE,
    )
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=BLOCK,
    )
    manager = layer.manager_cls(
        new_kv_cache_group(["layer"], layer),
        block_pool=pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=STRIDE,
        max_admission_blocks_per_request=10**9,
    )
    return manager, pool


def held(manager: Inkling2ConvStateManager, req: str) -> list[int]:
    return [i for i, b in enumerate(manager.req_to_blocks[req]) if not b.is_null]


def test_prefill_chunk_backs_stride_tails_and_its_tail():
    manager, pool = new_manager()
    free = pool.get_num_free_blocks()
    need = manager.get_num_blocks_to_allocate("r", 1000, [], 0, 0, 1000)
    manager.allocate_new_blocks("r", 1000, 1000)
    # Blocks ending on a multiple of 64 hold tokens 61..63 of each stride;
    # block 62 holds the chunk's last 3 tokens (997..999).
    assert held(manager, "r") == [*range(3, 62, 4), 62]
    assert free - pool.get_num_free_blocks() == need == 16

    # The next step frees everything the window has passed, nulls included.
    manager.remove_skipped_blocks("r", 1000)
    assert held(manager, "r") == [62]
    assert pool.get_num_free_blocks() == free - 1


def test_decode_and_second_chunk_keep_only_tails():
    manager, pool = new_manager()
    manager.allocate_new_blocks("r", 100, 100)
    assert held(manager, "r") == [3, 6]  # tokens 61..63 and 97..99
    manager.remove_skipped_blocks("r", 100)
    assert held(manager, "r") == [6]
    # A 300-token chunk from token 100 reads tokens 97..99 (block 6).
    assert manager.get_num_blocks_to_allocate("r", 400, [], 100, 0, 400) == 6
    manager.allocate_new_blocks("r", 400, 400)
    assert held(manager, "r") == [6, 7, 11, 15, 19, 23, 24]
    manager.remove_skipped_blocks("r", 400)
    assert held(manager, "r") == [24]
    manager.free("r")
    assert pool.get_num_free_blocks() == 127


def test_without_prefix_caching_only_the_tail_is_backed():
    manager, _ = new_manager(enable_caching=False)
    assert manager.get_num_blocks_to_allocate("r", 1000, [], 0, 0, 1000) == 1
    manager.allocate_new_blocks("r", 1000, 1000)
    assert held(manager, "r") == [62]


def test_admission_cap_counts_strides_not_blocks():
    config = Inkling2ConvStateManager.config(
        sliding_window=WINDOW, retain_stride=STRIDE
    )
    cap = Inkling2ConvStateManager.max_admission_blocks(config, BLOCK, 16384, 10**6)
    assert cap == 16387 // STRIDE + 1 + 4
