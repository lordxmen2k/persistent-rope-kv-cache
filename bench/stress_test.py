"""Multi-tenant stress test for PersistentRope.

Simulates a multi-tenant inference workload: many concurrent sequences
with prefix sharing, mixed append/unlink/fork patterns, and concurrent
reads. Measures rope operations under realistic load.

This Python reference uses a global _mutation_lock to serialize mutating
operations (the Rust port uses per-node CAS instead). Under that lock,
100 threads give zero speedup, so this test uses smaller workloads
that finish in a couple of minutes and reports both single-threaded
and contended multi-threaded numbers.

Run: python -m bench.stress_test
"""
import sys
import time
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src")

from rope import PersistentRope, NULL_NODE


def seed_workload(r: PersistentRope, n_seqs: int, seq_len: int, prefix_len: int):
    """Seed the rope with `n_seqs` sequences that all share a common prefix.

    Returns the list of sequence root IDs. Each sequence branches off the
    shared prefix independently so they all share the prefix structurally.
    """
    r.append(b"P" * prefix_len)
    common_root = r.root_id()
    roots = []
    for i in range(n_seqs):
        suffix = bytes([(i % 26) + ord("a")]) * seq_len
        new_root = r.append_to(common_root, suffix)
        roots.append(new_root)
    return roots


def main():
    print("=" * 60)
    print("Multi-tenant stress test for PersistentRope")
    print("=" * 60)
    random.seed(42)

    n_seqs = 32
    seq_len = 30
    prefix_len = 200

    print(f"\nSetup: {n_seqs} sequences, seq_len={seq_len}, prefix_len={prefix_len}")
    r = PersistentRope(block_size=16)
    t0 = time.perf_counter()
    roots = seed_workload(r, n_seqs, seq_len, prefix_len)
    t_seed = time.perf_counter() - t0
    r.collect_garbage()
    s = r.stats()
    print(f"  seed time: {t_seed*1000:.1f}ms")
    print(f"  total tokens: {s['length']}, leaves: {s['n_leaves']}, internals: {s['n_internals']}")
    print(f"  memory (mapping entries): {s['n_nodes_total']}")

    # ------------------------------------------------------------------
    # Test 1: single-threaded append throughput (baseline)
    # ------------------------------------------------------------------
    print(f"\n[1] single-threaded, 5000 appends to one sequence")
    n_ops = 5000
    t0 = time.perf_counter()
    root = roots[0]
    for _ in range(n_ops):
        root = r.append_to(root, b"a")
    t1 = time.perf_counter()
    roots[0] = root
    print(f"  elapsed: {(t1-t0)*1000:.1f}ms  ({n_ops/(t1-t0):,.0f} ops/sec)")

    # ------------------------------------------------------------------
    # Test 2: many concurrent appenders (lock-serialized)
    # ------------------------------------------------------------------
    n_threads = 8  # small enough to finish in seconds
    ops_per_thread = 500
    total_ops = n_threads * ops_per_thread
    barrier = threading.Barrier(n_threads)
    errors = []

    def append_worker(seq_idx):
        try:
            barrier.wait()
            local_root = roots[seq_idx]
            for _ in range(ops_per_thread):
                local_root = r.append_to(local_root, b"a")
            return seq_idx, local_root
        except Exception as e:
            errors.append(e)
            return seq_idx, None

    print(f"\n[2] {n_threads} threads, {ops_per_thread} appends each ({total_ops} total)")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(append_worker, i) for i in range(n_threads)]
        for fut in as_completed(futures):
            seq_idx, new_root = fut.result()
            if new_root is not None:
                roots[seq_idx] = new_root
    t1 = time.perf_counter()
    r.collect_garbage()
    s = r.stats()
    print(f"  elapsed: {(t1-t0)*1000:.1f}ms  ({total_ops/(t1-t0):,.0f} ops/sec)")
    print(f"  total tokens: {s['length']}, leaves: {s['n_leaves']}, internals: {s['n_internals']}")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:3]: print(f"    {e}")

    # ------------------------------------------------------------------
    # Test 3: mixed workload (append/unlink/snapshot)
    # ------------------------------------------------------------------
    n_threads = 8
    ops_per_thread = 500
    total_ops = n_threads * ops_per_thread
    barrier = threading.Barrier(n_threads)
    errors = []

    def mixed_worker(seq_idx):
        try:
            barrier.wait()
            local = list(roots)
            for _ in range(ops_per_thread):
                op = random.choice(["append", "snapshot", "unlink"])
                idx = random.randint(0, len(local) - 1)
                if op == "append":
                    local[idx] = r.append_to(local[idx], b"x")
                elif op == "unlink":
                    cur = r.length_of(local[idx])
                    if cur > 5:
                        local[idx] = r.unlink_from(local[idx], 1)
                else:  # snapshot
                    r.resolve(local[idx])
        except Exception as e:
            errors.append(e)

    print(f"\n[3] {n_threads} threads, {ops_per_thread} mixed ops each ({total_ops} total)")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(mixed_worker, i) for i in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()
    t1 = time.perf_counter()
    r.collect_garbage()
    s = r.stats()
    print(f"  elapsed: {(t1-t0)*1000:.1f}ms  ({total_ops/(t1-t0):,.0f} ops/sec)")
    print(f"  total tokens: {s['length']}, leaves: {s['n_leaves']}, internals: {s['n_internals']}")
    if errors:
        print(f"  ERRORS: {len(errors)}")

    # ------------------------------------------------------------------
    # Test 4: read-heavy workload (concurrent snapshots, no lock needed)
    # ------------------------------------------------------------------
    n_threads = 16
    ops_per_thread = 2000
    total_ops = n_threads * ops_per_thread
    reads = 0
    reads_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)
    errors = []

    def read_worker():
        nonlocal reads
        try:
            barrier.wait()
            local = 0
            for _ in range(ops_per_thread):
                idx = random.randint(0, len(roots) - 1)
                snap = r.resolve(roots[idx])
                local += sum(s[2] for s in snap)
            with reads_lock:
                reads += local
        except Exception as e:
            errors.append(e)

    print(f"\n[4] {n_threads} threads, {ops_per_thread} read-only ops each ({total_ops} total)")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(read_worker) for _ in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()
    t1 = time.perf_counter()
    print(f"  elapsed: {(t1-t0)*1000:.1f}ms  ({total_ops/(t1-t0):,.0f} ops/sec)")
    print(f"  total tokens summed across snapshots: {reads:,}")
    if errors:
        print(f"  ERRORS: {len(errors)}")

    # ------------------------------------------------------------------
    # Test 5: rebalance on a fresh single-root rope (for clean numbers)
    #
    # NOTE: rebalance() across a multi-root rope has a known limitation:
    # it operates on the current main root and may drop internals that
    # other live roots still reference (see GITHUB_ISSUES.md P0-03).
    # For honest rebalance perf numbers, we measure a single fresh rope.
    # ------------------------------------------------------------------
    print(f"\n[5] rebalance on a fresh single-root 10K-token rope")
    r_fresh = PersistentRope(block_size=16)
    r_fresh.append(b"x" * 10_000)
    r_fresh.collect_garbage()  # drop the replaced internals from in-place appends
    s = r_fresh.stats()
    print(f"  pre-rebalance:  leaves={s['n_leaves']}, internals={s['n_internals']}")
    t0 = time.perf_counter()
    r_fresh.rebalance_to_root()
    t1 = time.perf_counter()
    s = r_fresh.stats()
    print(f"  rebalance time: {(t1-t0)*1000:.1f}ms")
    print(f"  post-rebalance: leaves={s['n_leaves']}, internals={s['n_internals']}")

    N = 100
    for pos in [0, 5000, 9999]:
        t0 = time.perf_counter()
        for _ in range(N):
            r_fresh.get_kv(pos)
        avg_us = (time.perf_counter() - t0) / N * 1e6
        print(f"  get_kv({pos}) avg: {avg_us:.1f} us")

    # ------------------------------------------------------------------
    # Test 6: consolidate
    # ------------------------------------------------------------------
    print(f"\n[6] consolidate")
    t0 = time.perf_counter()
    result = r.consolidate()
    t1 = time.perf_counter()
    print(f"  consolidate time: {(t1-t0)*1000:.1f}ms")
    print(f"  freed: {result['freed']}, reachable: {result['reachable']}, corrupt: {result['corrupt']}")
    s = r.stats()
    print(f"  post-consolidate: leaves={s['n_leaves']} internals={s['n_internals']} mapping={s['n_nodes_total']}")


if __name__ == "__main__":
    main()
