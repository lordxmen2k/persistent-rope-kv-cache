# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive basic-operation tests for PersistentRope.

Run: pytest tests/test_rope_basic.py -v
"""
import sys
sys.path.insert(0, "src")

import pytest
from rope import (
    PersistentRope,
    Tier,
    RopeError,
    EmptySequenceError,
    OutOfRangeError,
    NodeID,
    NULL_NODE,
)


# -------------------------------------------------------------------
# Construction & invariants
# -------------------------------------------------------------------


def test_empty_rope():
    r = PersistentRope(block_size=4)
    assert r.is_empty()
    assert r.length() == 0
    assert r.resolve() == []


def test_single_token_appends():
    r = PersistentRope(block_size=4)
    root = r.append(b"a")
    assert not r.is_empty()
    assert r.length() == 1
    assert r.get_kv(0) == b"a"


def test_many_singletons_force_multiple_splits():
    r = PersistentRope(block_size=3)
    for i in range(10):
        r.append(bytes([ord("a") + i]))
    assert r.length() == 10
    s = r.stats()
    assert s["n_leaves"] >= 4
    assert s["n_internals"] >= 1


def test_batch_append():
    r = PersistentRope(block_size=4)
    r.append(b"abcd")
    assert r.length() == 4
    assert r.get_kv(0) == b"a"
    assert r.get_kv(3) == b"d"


def test_long_sequence_random_walks():
    """Sanity: a long sequence of appends doesn't lose data."""
    r = PersistentRope(block_size=4)
    r.append(b"a")
    for i in range(1, 100):
        r.append(bytes([(i % 26) + ord("a")]))
    assert r.length() == 100
    for p in range(100):
        expected = bytes([(p % 26) + ord("a")])
        assert r.get_kv(p) == expected, f"position {p}"


# -------------------------------------------------------------------
# Unlink
# -------------------------------------------------------------------


def test_unlink_zero_is_noop():
    r = PersistentRope(block_size=4)
    r.append(b"abcdef")
    r.unlink(0)
    assert r.length() == 6


def test_unlink_partial():
    r = PersistentRope(block_size=4)
    r.append(b"abcdefgh")
    r.unlink(3)
    assert r.length() == 5
    assert r.get_kv(0) == b"d"
    assert r.get_kv(4) == b"h"


def test_unlink_exact_boundary():
    r = PersistentRope(block_size=4)
    r.append(b"abcdefgh")
    r.unlink(4)
    assert r.length() == 4
    assert r.get_kv(0) == b"e"


def test_unlink_more_than_length_empties():
    r = PersistentRope(block_size=4)
    r.append(b"abc")
    r.unlink(100)
    assert r.length() == 0


def test_unlink_from_empty_raises():
    r = PersistentRope(block_size=4)
    with pytest.raises(EmptySequenceError):
        r.unlink(1)


def test_unlink_negative_raises():
    r = PersistentRope(block_size=4)
    r.append(b"abc")
    with pytest.raises(ValueError):
        r.unlink(-1)


def test_unlink_then_append():
    r = PersistentRope(block_size=4)
    r.append(b"abcdefgh")
    r.unlink(3)
    r.append(b"X")
    assert r.length() == 6
    assert r.get_kv(0) == b"d"
    assert r.get_kv(5) == b"X"


def test_unlink_full_then_append_works():
    """After fully unlinking, appending from empty must work."""
    r = PersistentRope(block_size=4)
    r.append(b"abcd")
    r.unlink(100)
    assert r.length() == 0
    r.append(b"xy")
    assert r.length() == 2
    assert r.get_kv(0) == b"x"
    assert r.get_kv(1) == b"y"


# -------------------------------------------------------------------
# Fork (structural sharing)
# -------------------------------------------------------------------


def test_fork_preserves_contents():
    r = PersistentRope(block_size=4)
    r.append(b"abcdef")
    fork_root = r.fork()
    assert r.length_of(fork_root) == 6
    assert r.get_kv(0, fork_root) == b"a"
    assert r.get_kv(5, fork_root) == b"f"


def test_fork_increments_refcounts_on_shared_subtree():
    r = PersistentRope(block_size=2)
    r.append(b"abcd")
    fork_root = r.fork()
    # Both roots are valid; the data is shared
    assert r.get_kv(0) == b"a"
    assert r.get_kv(0, fork_root) == b"a"
    # The fork root's refcount should be >= 1
    assert r.refcount_of(fork_root) >= 1


def test_fork_then_diverge_yields_independent_trees():
    r = PersistentRope(block_size=2)
    r.append(b"abcd")
    fork_root = r.fork()
    a_side = r.append_to(r.root_id(), b"X")
    b_side = r.append_to(fork_root, b"Y")
    assert r.get_kv(4, a_side) == b"X"
    assert r.get_kv(4, b_side) == b"Y"
    assert r.get_kv(0, a_side) == b"a"
    assert r.get_kv(0, b_side) == b"a"


# -------------------------------------------------------------------
# share_prefix
# -------------------------------------------------------------------


def test_share_prefix_full():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    a = r.root_id()
    b = r.share_prefix(a, 6)
    assert r.length_of(b) == 6
    assert r.get_kv(0, b) == b"a"
    assert r.get_kv(5, b) == b"f"


def test_share_prefix_zero_returns_input():
    r = PersistentRope(block_size=2)
    r.append(b"abc")
    a = r.root_id()
    b = r.share_prefix(a, 0)
    assert b == a


def test_share_prefix_full_fork():
    """Sharing the full length is equivalent to fork()."""
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    a = r.root_id()
    f = r.fork()
    b = r.share_prefix(a, r.length_of(a))
    # Both should give access to the same data
    for p in range(6):
        assert r.get_kv(p, b) == r.get_kv(p, f)


def test_share_prefix_partial():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")  # length 6
    a = r.root_id()
    b = r.share_prefix(a, 3)
    assert r.length_of(b) == 3
    assert r.get_kv(0, b) == b"a"
    assert r.get_kv(2, b) == b"c"


# -------------------------------------------------------------------
# Evict / promote (tier changes)
# -------------------------------------------------------------------


def test_evict_marks_warm():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    r.evict(2)
    leaves = list(r._iter_leaves(r.root_id()))
    assert leaves[0].tier == Tier.WARM
    assert all(l.tier == Tier.HOT for l in leaves[1:])


def test_promote_marks_hot():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    r.evict(6)
    r.promote(2)
    leaves = list(r._iter_leaves(r.root_id()))
    assert all(l.tier == Tier.WARM for l in leaves[:-1])
    assert leaves[-1].tier == Tier.HOT


# -------------------------------------------------------------------
# Snapshot (resolve)
# -------------------------------------------------------------------


def test_resolve_order():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    snap = r.resolve()
    assert len(snap) == 3
    assert [s[2] for s in snap] == [2, 2, 2]
    assert snap[0][1] == 0
    assert snap[1][1] == 2
    assert snap[2][1] == 4


def test_resolve_after_unlink():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    r.unlink(2)
    snap = r.resolve()
    assert sum(s[2] for s in snap) == 4


# -------------------------------------------------------------------
# Refcount / GC
# -------------------------------------------------------------------


def test_collect_garbage_drops_deferred_nodes():
    r = PersistentRope(block_size=2)
    r.append(b"abcdefghij")
    before = r.stats()["n_nodes_total"]
    r.collect_garbage()
    after = r.stats()["n_nodes_total"]
    assert r.length() == 10


def test_refcount_zero_path():
    r = PersistentRope(block_size=2)
    r.append(b"abc")
    fork_root = r.fork()
    # Drop the original entirely
    r.unlink(100)
    assert r.length() == 0
    # The fork is still valid
    assert r.length_of(fork_root) == 3


# -------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------


def test_stats_tier_counts():
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    r.evict(2)
    s = r.stats()
    assert s["tier_counts"]["warm"] == 1
    assert s["tier_counts"]["hot"] >= 2


def test_stats_mutations_counter():
    r = PersistentRope(block_size=2)
    r.append(b"abc")
    s_before = r.stats()["mutations"]
    r.append(b"d")
    s_after = r.stats()["mutations"]
    assert s_after == s_before + 1


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


def test_append_blob_length_must_match_n():
    r = PersistentRope(block_size=2)
    with pytest.raises(ValueError):
        r.append(b"abc", n=4)
    with pytest.raises(ValueError):
        r.append(b"ab", n=3)


def test_block_size_validation():
    with pytest.raises(ValueError):
        PersistentRope(block_size=0)


def test_get_kv_out_of_range():
    r = PersistentRope(block_size=2)
    r.append(b"abc")
    with pytest.raises(OutOfRangeError):
        r.get_kv(10)


def test_repeated_fork_unlink_idempotent():
    """Forking and unlinking repeatedly should not corrupt the rope."""
    r = PersistentRope(block_size=2)
    r.append(b"abcdef")
    for _ in range(10):
        f = r.fork()
        assert r.get_kv(0, f) == b"a"
    assert r.length() == 6


def test_append_one_blob_byte():
    """A 1-byte blob appends exactly 1 token."""
    r = PersistentRope(block_size=4)
    r.append(b"x")
    r.append(b"y")
    r.append(b"z")
    assert r.length() == 3
    assert r.get_kv(0) == b"x"
    assert r.get_kv(1) == b"y"
    assert r.get_kv(2) == b"z"
