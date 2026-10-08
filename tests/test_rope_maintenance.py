"""Tests for rebalance() and consolidate() (P0-02 from GITHUB_ISSUES.md)."""
import sys
import time
sys.path.insert(0, "src")

import pytest
from rope import PersistentRope, NULL_NODE


# -------------------------------------------------------------------
# Regression: unlink + append cycle
# -------------------------------------------------------------------


def test_unlink_then_append_does_not_crash():
    """After many appends (which right-skew the tree), unlink then append
    should not crash. Regression test for the deferred_free / GC bug."""
    r = PersistentRope(block_size=16)
    r.append(b"x" * 32)
    r.collect_garbage()
    for _ in range(5):
        r.unlink(1)
        r.append(b"y")
    assert r.length() == 32


def test_unlink_append_cycle_preserves_data():
    r = PersistentRope(block_size=16)
    r.append(b"x" * 1000)
    r.collect_garbage()
    for i in range(50):
        r.unlink(1)
        r.append(bytes([i % 256]))
    # Last 50 tokens should be from the loop
    for i in range(50):
        assert r.get_kv(r.length() - 50 + i) == bytes([i % 256])


# -------------------------------------------------------------------
# rebalance
# -------------------------------------------------------------------


def test_rebalance_empty():
    r = PersistentRope(block_size=16)
    r.rebalance()
    assert r.length() == 0


def test_rebalance_single_leaf():
    r = PersistentRope(block_size=16)
    r.append(b"abc")
    new_root = r.rebalance()
    assert r.length() == 3
    assert r.get_kv(0) == b"a"


def test_rebalance_balances_tree():
    """A right-skewed tree from sequential appends should become balanced."""
    r = PersistentRope(block_size=16)
    r.append(b"x" * 10000)
    r.collect_garbage()
    internals_before = r.stats()["n_internals"]
    # Right-skewed: internals form a chain, much less than the leaves-1 you'd get balanced
    assert internals_before < 1000  # confirm right-skewed
    r.rebalance_to_root()
    internals_after = r.stats()["n_internals"]
    leaves = r.stats()["n_leaves"]
    # A balanced binary tree has internals = leaves - 1
    assert abs(internals_after - (leaves - 1)) < leaves // 2

    # Depth is O(log n) — measured by the get_kv perf test below
    # (no easy depth-walk without exposing internals)


def test_rebalance_preserves_data():
    r = PersistentRope(block_size=16)
    payload = bytes([i % 256 for i in range(1000)])
    r.append(payload)
    r.rebalance_to_root()
    for i in range(1000):
        assert r.get_kv(i) == bytes([i % 256])


def test_rebalance_improves_get_kv_perf():
    """After rebalance, get_kv on middle positions should be much faster."""
    r = PersistentRope(block_size=16)
    r.append(b"x" * 10000)
    r.collect_garbage()
    N = 1000
    t = time.perf_counter()
    for _ in range(N):
        r.get_kv(5000)
    before = time.perf_counter() - t

    r.rebalance_to_root()
    t = time.perf_counter()
    for _ in range(N):
        r.get_kv(5000)
    after = time.perf_counter() - t

    # At least 5x faster (right-skewed is ~45us, balanced is ~1.5us)
    assert after * 5 < before, f"rebalance did not improve get_kv: before={before:.3f}s, after={after:.3f}s"


def test_rebalance_then_unlink_preserves_suffix():
    r = PersistentRope(block_size=16)
    r.append(b"x" * 100)
    r.rebalance_to_root()
    r.unlink(30)
    assert r.length() == 70
    for i in range(70):
        assert r.get_kv(i) == b"x"


def test_rebalance_then_fork_shares():
    r = PersistentRope(block_size=16)
    r.append(b"x" * 100)
    r.rebalance_to_root()
    a = r.root_id()
    b = r.fork(a)
    a2 = r.append_to(a, b"a" * 10)
    b2 = r.append_to(b, b"b" * 10)
    assert r.get_kv(0, a2) == b"x"
    assert r.get_kv(0, b2) == b"x"
    assert r.get_kv(100, a2) == b"a"
    assert r.get_kv(100, b2) == b"b"


# -------------------------------------------------------------------
# consolidate
# -------------------------------------------------------------------


def test_consolidate_returns_stats():
    r = PersistentRope(block_size=16)
    r.append(b"x" * 100)
    result = r.consolidate()
    assert "freed" in result
    assert "reachable" in result
    assert "corrupt" in result
    assert result["corrupt"] == 0


def test_consolidate_drains_deferred_free():
    r = PersistentRope(block_size=16)
    r.append(b"x" * 100)
    # Don't call collect_garbage; let consolidate do it
    deferred_before = len(r._deferred_free)
    assert deferred_before > 0
    result = r.consolidate()
    assert result["freed"] == deferred_before
    assert len(r._deferred_free) == 0


def test_consolidate_resets_mutation_count():
    r = PersistentRope(block_size=16, consolidation_threshold=10)
    r.append(b"x" * 100)
    assert r._mutation_count > 0
    r.consolidate()
    assert r._mutation_count == 0


def test_auto_consolidate_triggers_in_foreground():
    r = PersistentRope(block_size=16, consolidation_threshold=5, auto_consolidate=True)
    # Each append is a mutation. After 5 appends, auto-consolidate fires.
    for _ in range(10):
        r.append(b"x")
    # Mutation count should be at or near 0 (auto-consolidated)
    assert r._mutation_count < 5
    # Data should still be intact
    assert r.length() == 10
    for i in range(10):
        assert r.get_kv(i) == b"x"


def test_consolidate_on_empty_rope():
    r = PersistentRope(block_size=16)
    result = r.consolidate()
    assert result["reachable"] == 0
    assert result["corrupt"] == 0


# -------------------------------------------------------------------
# Combined: rebalance + consolidate
# -------------------------------------------------------------------


def test_rebalance_then_consolidate():
    r = PersistentRope(block_size=16)
    r.append(b"x" * 1000)
    r.unlink(100)
    r.append(b"y" * 100)
    r.rebalance_to_root()
    result = r.consolidate()
    assert result["corrupt"] == 0
    # Data is intact
    assert r.length() == 1000
    for i in range(900):
        assert r.get_kv(i) == b"x"
    for i in range(100):
        assert r.get_kv(900 + i) == b"y"


def test_rebalance_idempotent():
    """Rebalancing an already-balanced tree should not change correctness."""
    r = PersistentRope(block_size=16)
    r.append(b"x" * 100)
    r.rebalance_to_root()
    s1 = r.stats()
    r.rebalance_to_root()
    s2 = r.stats()
    # Same number of leaves, same number of internals
    assert s1["n_leaves"] == s2["n_leaves"]
    # Internals may differ slightly but both should be balanced (O(log n))
    import math
    expected_internals_max = int(s1["n_leaves"] * math.log2(s1["n_leaves"] + 1)) + 10
    assert s2["n_internals"] <= expected_internals_max
