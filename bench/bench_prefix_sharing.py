"""Prefix-sharing benchmark: demonstrates the rope's structural sharing.

Compares:
  - N sequences built from the same 200-token shared prefix
  - N sequences with no shared prefix

The rope's persistent design means shared-prefix sequences share leaves
structurally. The Python reference uses refcounting to keep those shared
nodes alive; the Rust port uses crossbeam-epoch.

We measure:
  - Setup time (per sequence)
  - Mapping-table size after seeding all N sequences
  - Average memory per sequence (mapping entries / N)

A real benefit: if 1000 sequences share a 200-token prefix, the mapping
table should have roughly 1 set of leaves for the prefix + 1 set of
leaves per sequence, not 1000 copies of the prefix.

Run: python -m bench.bench_prefix_sharing
"""
import sys
import time

sys.path.insert(0, "src")

from rope import PersistentRope


def bench(n_seqs: int, prefix_len: int, seq_len: int, share: bool) -> dict:
    """Build n_seqs sequences. If share, all branch from a common prefix.

    Returns timing + memory stats.
    """
    r = PersistentRope(block_size=16)

    # Build a "shared" prefix
    t0 = time.perf_counter()
    if share:
        r.append(b"P" * prefix_len)
        common = r.root_id()

    # Build N sequences
    roots = []
    for i in range(n_seqs):
        suffix = bytes([(i % 26) + ord("a")]) * seq_len
        if share:
            new_root = r.append_to(common, suffix)
        else:
            # Independent sequence of prefix_len + seq_len tokens
            r.append(b"P" * prefix_len)
            new_root = r.append_to(r.root_id(), suffix)
        roots.append(new_root)
    t_setup = time.perf_counter() - t0

    r.collect_garbage()
    s = r.stats()

    return {
        "n_seqs": n_seqs,
        "share": share,
        "t_setup_ms": t_setup * 1000,
        "per_seq_us": t_setup / n_seqs * 1e6,
        "n_leaves": s["n_leaves"],
        "n_internals": s["n_internals"],
        "n_nodes_total": s["n_nodes_total"],
        "n_nodes_per_seq": s["n_nodes_total"] / n_seqs,
    }


def main():
    print("=" * 60)
    print("Prefix-sharing benchmark")
    print("=" * 60)
    print()
    print(f"{'config':<32} {'setup ms':>10} {'us/seq':>8} "
          f"{'leaves':>8} {'internals':>10} {'nodes/seq':>10}")
    print("-" * 80)

    for n_seqs in [10, 100]:
        for share in [False, True]:
            res = bench(n_seqs, prefix_len=200, seq_len=50, share=share)
            cfg = f"{'shared' if share else 'independent':>11s} N={n_seqs:<4}"
            print(f"{cfg:<32} "
                  f"{res['t_setup_ms']:>10.1f} "
                  f"{res['per_seq_us']:>8.1f} "
                  f"{res['n_leaves']:>8,} "
                  f"{res['n_internals']:>10,} "
                  f"{res['n_nodes_per_seq']:>10.1f}")

    print()
    print("=" * 60)
    print("Sharing benefit analysis (N=100, prefix=200, seq=50)")
    print("=" * 60)
    indep = bench(100, 200, 50, share=False)
    shared = bench(100, 200, 50, share=True)

    # Independent: each seq has its own (prefix+seq) leaves/internals
    # Shared: 1 prefix + N suffixes
    indep_total = indep["n_nodes_total"]
    shared_total = shared["n_nodes_total"]
    saved = indep_total - shared_total
    pct = saved / indep_total * 100

    print(f"  independent mapping size:  {indep_total:>10,} nodes")
    print(f"  shared mapping size:       {shared_total:>10,} nodes")
    print(f"  nodes saved by sharing:    {saved:>10,} ({pct:.1f}%)")
    print()
    print("  Without sharing, every sequence duplicates the prefix")
    print("  leaves/internals. The rope's persistent structure shares them,")
    print("  so the mapping table grows with the UNION of the data, not the sum.")


if __name__ == "__main__":
    main()
