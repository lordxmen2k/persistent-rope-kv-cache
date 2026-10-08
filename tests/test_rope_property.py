# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for PersistentRope using Hypothesis.

These tests generate random sequences of operations and assert invariants.
Run: pytest tests/test_rope_property.py -v
"""
import sys
sys.path.insert(0, "src")

import pytest
from hypothesis import given, settings, strategies as st
from rope import PersistentRope, NULL_NODE, OutOfRangeError


# Generate small blobs (each byte is one token in this reference)
blob_strategy = st.binary(min_size=1, max_size=8)
small_n = st.integers(min_value=1, max_value=8)


@st.composite
def op_sequence(draw):
    """A list of operations to apply."""
    ops = []
    n_ops = draw(st.integers(min_value=3, max_value=30))
    for _ in range(n_ops):
        op = draw(st.sampled_from(["append", "unlink", "fork", "get_kv"]))
        if op == "append":
            blob = draw(blob_strategy)
            ops.append(("append", blob))
        elif op == "unlink":
            n = draw(st.integers(min_value=0, max_value=4))
            ops.append(("unlink", n))
        elif op == "fork":
            ops.append(("fork",))
        else:  # get_kv
            pos = draw(st.integers(min_value=0, max_value=20))
            ops.append(("get_kv", pos))
    return ops


@given(blob=blob_strategy)
@settings(max_examples=200, deadline=None)
def test_append_then_resolve_round_trip(blob):
    """After append, every byte of the blob is reachable via get_kv."""
    r = PersistentRope(block_size=4)
    r.append(blob)
    assert r.length() == len(blob)
    for i, b in enumerate(blob):
        assert r.get_kv(i) == bytes([b])


@given(blob1=blob_strategy, blob2=blob_strategy)
@settings(max_examples=200, deadline=None)
def test_append_twice_concatenates(blob1, blob2):
    r = PersistentRope(block_size=4)
    r.append(blob1)
    r.append(blob2)
    expected = blob1 + blob2
    assert r.length() == len(expected)
    for i, b in enumerate(expected):
        assert r.get_kv(i) == bytes([b])


@given(blob=blob_strategy, n=st.integers(min_value=0, max_value=20))
@settings(max_examples=200, deadline=None)
def test_unlink_keeps_suffix(blob, n):
    """Unlink drops the leftmost n tokens, keeps the suffix."""
    r = PersistentRope(block_size=4)
    r.append(blob)
    n_eff = min(n, len(blob))
    r.unlink(n_eff)
    expected = blob[n_eff:]
    assert r.length() == len(expected)
    for i, b in enumerate(expected):
        assert r.get_kv(i) == bytes([b])


@given(blob=blob_strategy)
@settings(max_examples=100, deadline=None)
def test_fork_preserves_all_data(blob):
    r = PersistentRope(block_size=4)
    r.append(blob)
    fork_root = r.fork()
    for i, b in enumerate(blob):
        assert r.get_kv(i) == r.get_kv(i, fork_root)


@given(blob1=blob_strategy, blob2=blob_strategy)
@settings(max_examples=100, deadline=None)
def test_fork_then_diverge(blob1, blob2):
    """Fork then append to both sides: shared prefix preserved, suffixes differ."""
    r = PersistentRope(block_size=4)
    r.append(blob1)
    fork_root = r.fork()
    a_side = r.append_to(r.root_id(), blob2)
    b_side = r.append_to(fork_root, blob2)
    for i, b in enumerate(blob1):
        assert r.get_kv(i, a_side) == r.get_kv(i, b_side) == bytes([b])
    for i, b in enumerate(blob2):
        assert r.get_kv(len(blob1) + i, a_side) == bytes([b])
        assert r.get_kv(len(blob1) + i, b_side) == bytes([b])


@given(blob=blob_strategy, k=st.integers(min_value=1, max_value=20))
@settings(max_examples=100, deadline=None)
def test_share_prefix_round_trip(blob, k):
    r = PersistentRope(block_size=4)
    r.append(blob)
    a = r.root_id()
    k_eff = min(k, len(blob))
    if k_eff == 0:
        return  # trivially true
    b = r.share_prefix(a, k_eff)
    assert r.length_of(b) == k_eff
    for i in range(k_eff):
        assert r.get_kv(i, b) == bytes([blob[i]])


@given(blob=blob_strategy, n=st.integers(min_value=0, max_value=20))
@settings(max_examples=100, deadline=None)
def test_evict_then_promote(blob, n):
    """Evict + promote: data unchanged, just tier tags change."""
    r = PersistentRope(block_size=4)
    r.append(blob)
    n_eff = min(n, len(blob))
    r.evict(n_eff)
    r.promote(n_eff)
    # Data should be unchanged
    for i, b in enumerate(blob):
        assert r.get_kv(i) == bytes([b]), f"data mismatch at {i}"


@given(
    blobs=st.lists(blob_strategy, min_size=2, max_size=10),
)
@settings(max_examples=50, deadline=None)
def test_snapshot_consistent_with_data(blobs):
    """The total length from resolve() equals r.length()."""
    r = PersistentRope(block_size=4)
    for b in blobs:
        r.append(b)
    snap = r.resolve()
    total = sum(s[2] for s in snap)
    assert total == r.length()


@given(blob=blob_strategy)
@settings(max_examples=50, deadline=None)
def test_random_unlink_append_cycles(blob):
    """A long sequence of unlink + append doesn't corrupt data."""
    r = PersistentRope(block_size=4)
    r.append(blob)
    for _ in range(20):
        if r.length() > 0:
            r.unlink(1)
        r.append(b"x")
    # Length should be (original - 20) + 20 = original
    assert r.length() == len(blob)
    for i in range(len(blob) - 20):
        if i + 20 < len(blob):
            assert r.get_kv(i) == bytes([blob[i + 20]])
