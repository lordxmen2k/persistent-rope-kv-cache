# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the vLLM-compatible block manager interface."""
import sys
sys.path.insert(0, "src")

import pytest
from rope import RopeBlockManager, RopeBlock, Tier, NULL_NODE


def test_create_and_delete_sequence():
    bm = RopeBlockManager(block_size=16)
    bm.create_sequence(1)
    assert bm.get_seq_root(1) == NULL_NODE
    bm.delete_sequence(1)
    assert bm.get_seq_root(1) is None


def test_allocate_returns_blocks():
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    blocks = bm.allocate(1, n_tokens=10, kv_data=b"x" * 10)
    assert len(blocks) > 1  # at least 3 blocks for 10 tokens at size 4
    for b in blocks:
        assert b.num_tokens > 0
        assert b.tier == Tier.HOT


def test_free_truncates():
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    bm.allocate(1, n_tokens=10, kv_data=b"x" * 10)
    bm.free(1, n_tokens=4)
    snap = bm.get_snapshot(1)
    assert sum(s[2] for s in snap) == 6


def test_fork_shares_then_diverges():
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    bm.allocate(1, n_tokens=8, kv_data=b"x" * 8)

    bm.fork(parent_seq_id=1, child_seq_id=2)
    # Both should have the same snapshot
    s1 = bm.get_snapshot(1)
    s2 = bm.get_snapshot(2)
    assert sum(s[2] for s in s1) == sum(s[2] for s in s2) == 8

    # Diverge: child appends different data
    bm.allocate(1, n_tokens=4, kv_data=b"a" * 4)
    bm.allocate(2, n_tokens=4, kv_data=b"b" * 4)

    s1_after = bm.get_snapshot(1)
    s2_after = bm.get_snapshot(2)
    assert sum(s[2] for s in s1_after) == 12
    assert sum(s[2] for s in s2_after) == 12
    # The data is different
    assert s1_after != s2_after


def test_can_append_always_true_for_existing_seq():
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    assert bm.can_append(1, 100) is True
    assert bm.can_append(99, 1) is False  # non-existent seq


def test_swap_out_demotes_tier():
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    bm.allocate(1, n_tokens=8, kv_data=b"x" * 8)
    bm.swap_out(1, 4)
    # First 4 tokens should be WARM
    # Check via the rope
    rope = bm.rope
    leaves = list(rope._iter_leaves(bm.get_seq_root(1)))
    assert leaves[0].tier == Tier.WARM


def test_swap_in_promotes_tier():
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    bm.allocate(1, n_tokens=8, kv_data=b"x" * 8)
    bm.swap_out(1, 8)  # all to warm
    bm.swap_in(1, 4)   # last 4 to hot
    rope = bm.rope
    leaves = list(rope._iter_leaves(bm.get_seq_root(1)))
    # Last leaf should be HOT
    assert leaves[-1].tier == Tier.HOT


def test_append_slot_singleton():
    """vLLM's append_slot is called per-token."""
    bm = RopeBlockManager(block_size=4)
    bm.create_sequence(1)
    for i in range(5):
        bm.append_slot(1, kv_data=bytes([i + 1]))
    snap = bm.get_snapshot(1)
    assert sum(s[2] for s in snap) == 5


def test_multi_sequence_independence():
    bm = RopeBlockManager(block_size=4)
    for sid in range(5):
        bm.create_sequence(sid)
        bm.allocate(sid, n_tokens=8, kv_data=bytes([sid + 1]) * 8)

    for sid in range(5):
        snap = bm.get_snapshot(sid)
        assert sum(s[2] for s in snap) == 8
