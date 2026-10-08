# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks for PersistentRope.

Run: python -m bench.bench_rope
"""
import sys
import time
sys.path.insert(0, "src")

from rope import PersistentRope


def bench(name: str, n: int, setup, op) -> float:
    """Run `op` n times. After each op, call `setup` to restore state.

    `setup` returns the rope (or any state) the op should work on.
    `op` takes that state and runs.
    """
    start = time.perf_counter()
    for _ in range(n):
        state = setup()
        op(state)
    elapsed = time.perf_counter() - start
    ops = n / elapsed
    print(f"  {name:40s} {ops:>12,.0f} ops/sec  ({elapsed*1000:.1f}ms for {n})")
    return ops


def main():
    print("PersistentRope microbenchmarks (block_size=16)")
    print("=" * 60)

    # Append throughput at various sizes
    for n in [1_000, 10_000]:
        def setup_append():
            return PersistentRope(block_size=16)
        def op_append(r):
            r.append(b"x")
        bench(f"append (1 token) x {n}", n, setup_append, op_append)

    # Build a moderately long rope, then bench reads (no mutation)
    r = PersistentRope(block_size=16)
    r.append(b"x" * 10_000)

    # Reads on a right-skewed tree are O(depth). Sequential appends produce
    # one leaf per 16 tokens, so this is a ~625-leaf / 625-internal tree
    # with depth proportional to n_leaves. get_kv near the head (0) is fast
    # because the head path is short; mid/late positions walk the spine.
    r_skewed = r
    bench("length() x 10000", 10_000, lambda: r_skewed, lambda r: r.length())
    bench("get_kv(0) x 10000 [skewed]", 10_000, lambda: r_skewed, lambda r: r.get_kv(0))
    bench("get_kv(5000) x 10000 [skewed]", 10_000, lambda: r_skewed, lambda r: r.get_kv(5000))
    bench("get_kv(9999) x 10000 [skewed]", 10_000, lambda: r_skewed, lambda r: r.get_kv(9999))
    bench("resolve() x 1000 [skewed]", 1_000, lambda: r_skewed, lambda r: r.resolve())
    bench("fork() x 1000 [skewed]", 1_000, lambda: r_skewed, lambda r: r.fork())

    # Same data, rebalanced. After rebalance, depth is O(log n_leaves).
    r_bal = PersistentRope(block_size=16)
    r_bal.append(b"x" * 10_000)
    r_bal.rebalance_to_root()
    bench("get_kv(0) x 10000 [balanced]", 10_000, lambda: r_bal, lambda r: r.get_kv(0))
    bench("get_kv(5000) x 10000 [balanced]", 10_000, lambda: r_bal, lambda r: r.get_kv(5000))
    bench("get_kv(9999) x 10000 [balanced]", 10_000, lambda: r_bal, lambda r: r.get_kv(9999))
    bench("resolve() x 1000 [balanced]", 1_000, lambda: r_bal, lambda r: r.resolve())

    # Unlink throughput (re-seed per op)
    def setup_unlink_small():
        r = PersistentRope(block_size=16)
        r.append(b"x" * 1000)
        return r
    def op_unlink_small(r):
        r.unlink(1)
    bench("unlink(1) x 1000", 1_000, setup_unlink_small, op_unlink_small)

    def setup_unlink_big():
        r = PersistentRope(block_size=16)
        r.append(b"x" * 1000)
        return r
    def op_unlink_big(r):
        r.unlink(100)
    bench("unlink(100) x 100", 100, setup_unlink_big, op_unlink_big)

    # Stats on a representative rope (with GC for accurate counts)
    r3 = PersistentRope(block_size=16)
    r3.append(b"x" * 10_000)
    r3.collect_garbage()
    print()
    print("Final state stats (post-GC, sequential 10K-token rope):")
    for k, v in r3.stats().items():
        print(f"  {k}: {v}")
    print()
    print("Note: sequential appends create one leaf per 16 tokens (block_size).")
    print("Tree depth is O(n_leaves); rebalance() makes it O(log n_leaves).")
    print("See the [skewed] vs [balanced] rows above for the impact on get_kv.")


if __name__ == "__main__":
    main()
