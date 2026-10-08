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

    bench("length() x 10000", 10_000, lambda: r, lambda r: r.length())
    bench("get_kv(0) x 10000", 10_000, lambda: r, lambda r: r.get_kv(0))
    bench("get_kv(5000) x 10000", 10_000, lambda: r, lambda r: r.get_kv(5000))
    bench("get_kv(9999) x 10000", 10_000, lambda: r, lambda r: r.get_kv(9999))
    bench("resolve() x 1000", 1_000, lambda: r, lambda r: r.resolve())
    bench("fork() x 1000", 1_000, lambda: r, lambda r: r.fork())

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
    print("Final state stats (post-GC):")
    for k, v in r3.stats().items():
        print(f"  {k}: {v}")
    print()
    print("Note: sequential appends produce a right-skewed tree (O(n) depth).")
    print("The Rust port rebalances; the Python reference is intentionally naive.")


if __name__ == "__main__":
    main()
