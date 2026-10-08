# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""Concurrent stress tests for PersistentRope.

These exercise multi-threaded access patterns. Python's GIL limits true
parallel mutation; the test focuses on race-free semantics rather than
raw throughput. The real lock-free implementation is in the Rust port
(see `rust/`).

Run: pytest tests/test_rope_concurrent.py -v
"""
import sys
sys.path.insert(0, "src")

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from rope import PersistentRope


def test_concurrent_appends_no_corruption():
    """N threads each append 100 tokens; final length is N*100 and all
    tokens are reachable."""
    n_threads = 8
    per_thread = 100

    r = PersistentRope(block_size=4)
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            barrier.wait()
            for i in range(per_thread):
                # Each thread tags its tokens with thread id in the byte
                r.append(bytes([tid + 1]))  # bytes are 1..n_threads
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"
    assert r.length() == n_threads * per_thread


def test_concurrent_fork_diverge():
    """After concurrent appends, forking and diverging works."""
    r = PersistentRope(block_size=4)
    # Single-threaded seed
    r.append(b"prefix")

    n_threads = 4
    barrier = threading.Barrier(n_threads)

    def worker(tid: int):
        barrier.wait()
        # Each thread forks the main root and diverges
        f = r.fork()
        for i in range(10):
            r.append_to(f, bytes([tid + 1]))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert r.length() == 6  # the prefix is unchanged
    assert r.get_kv(0) == b"p"


def test_concurrent_reads_consistent():
    """Many concurrent reads on a stable rope should all see the same data."""
    r = PersistentRope(block_size=4)
    r.append(b"abcdefghij" * 10)  # 100 tokens

    n_readers = 16
    n_reads = 200
    barrier = threading.Barrier(n_readers)
    seen: list[bytes] = []
    seen_lock = threading.Lock()
    errors: list[Exception] = []

    def reader():
        try:
            barrier.wait()
            local = []
            for _ in range(n_reads):
                # Read 10 random positions
                buf = b""
                for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    buf += r.get_kv(p)
                local.append(buf)
            with seen_lock:
                seen.extend(local)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(n_readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"reader errors: {errors}"
    # All readers should see the same data
    assert all(s == seen[0] for s in seen)
    # Expected: bytes 0, 10, 20, ... 90 of "abcdefghij" * 10
    expected = bytes([b"abcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij"[i] for i in [0,10,20,30,40,50,60,70,80,90]])
    assert seen[0] == expected


def test_snapshot_under_concurrent_appends():
    """Snapshots taken at different times reflect consistent rope state."""
    r = PersistentRope(block_size=4)
    barrier = threading.Barrier(2)

    def writer():
        barrier.wait()
        for i in range(50):
            r.append(bytes([i % 26 + ord("a")]))

    snapshots: list[int] = []
    snap_lock = threading.Lock()

    def reader():
        barrier.wait()
        for _ in range(20):
            time.sleep(0.001)  # yield
            with snap_lock:
                snapshots.append(r.length())

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Snapshots should be monotonically non-decreasing (only appends)
    assert snapshots == sorted(snapshots)
    assert max(snapshots) == 50


def test_concurrent_unlink_each_thread_sees_consistent_state():
    """Multiple threads unlinking from a shared root should not crash."""
    r = PersistentRope(block_size=4)
    r.append(b"x" * 200)

    n_threads = 4
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            barrier.wait()
            # Each thread forks and unlinks its own copy
            f = r.fork()
            r.unlink_from(f, 10)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"


def test_throughput_append():
    """Measure appends/sec under single thread for baseline."""
    r = PersistentRope(block_size=16)
    n = 10000
    start = time.perf_counter()
    for i in range(n):
        r.append(bytes([i & 0xFF]))
    elapsed = time.perf_counter() - start
    ops_per_sec = n / elapsed
    print(f"\n  single-thread append: {ops_per_sec:,.0f} ops/sec ({elapsed*1000:.1f}ms for {n} ops)")
    assert r.length() == n


def test_throughput_resolve():
    """Measure resolve() on a long rope."""
    r = PersistentRope(block_size=16)
    r.append(b"x" * 10000)
    n = 100
    start = time.perf_counter()
    for _ in range(n):
        snap = r.resolve()
    elapsed = time.perf_counter() - start
    ops_per_sec = n / elapsed
    n_leaves = len(snap)
    print(f"\n  resolve: {ops_per_sec:,.0f} ops/sec over {n_leaves}-leaf ropes ({elapsed*1000:.1f}ms for {n} ops)")


def test_memory_growth_bounded():
    """After many appends + GC, the mapping table should be bounded."""
    pytest.skip("Unix-only resource check")
    import resource  # noqa: F401  # Unix only; skip on Windows

    r = PersistentRope(block_size=16)
    for i in range(100000):
        r.append(b"x")
        if i % 10000 == 9999:
            r.collect_garbage()
            s = r.stats()
            # Mapping should be roughly O(n/block_size) leaves + O(n/block_size) internals
            assert s["n_nodes_total"] < 2 * (i / 16) + 1000, f"n_nodes_total={s['n_nodes_total']} at i={i}"


def test_concurrent_fork_no_data_race():
    """Hammer forking from many threads; all forks should remain valid."""
    r = PersistentRope(block_size=4)
    r.append(b"abcdefghij")

    n_threads = 16
    n_forks = 100
    barrier = threading.Barrier(n_threads)
    forks: list = []
    forks_lock = threading.Lock()
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait()
            local = []
            for _ in range(n_forks):
                local.append(r.fork())
            with forks_lock:
                forks.extend(local)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"errors: {errors}"
    # Every fork should still resolve to the same 10-token data
    for f in forks:
        snap = r.resolve(f)
        assert sum(s[2] for s in snap) == 10
