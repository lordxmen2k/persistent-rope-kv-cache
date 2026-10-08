# Persistent Rope KV Cache

Research prototype exploring a position-keyed persistent rope with
Bw-tree-style indirection as the index for KV cache in LLM inference.

**Status:** Phase 1+2+3 implemented. Python correctness reference,
concurrent stress tests, vLLM-compatible adapter, and HuggingFace
integration demo all run on this machine. The real lock-free Bw-tree
indirection lives in the Rust port (see `rust/` — scaffold only).

The hypothesis: a position-keyed persistent rope cleanly unifies four
mechanisms that current serving systems (vLLM, SGLang, Mooncake) handle
separately — and a Bw-tree mapping table makes the index latch-free
under multi-tenant load.

## What's shipped

| Phase | File | Status |
|---|---|---|
| Design | `tech_note.md` (v0.2) | 26 references, reframed toward lock-free Bw-tree-on-rope |
| Phase 1 | `src/rope/core.py` | PersistentRope: append, unlink, fork, share_prefix, evict, promote, resolve, get_kv, rebalance, consolidate |
| Phase 1 | `tests/test_rope_basic.py` | 33 unit tests |
| Phase 1 | `tests/test_rope_property.py` | 9 property tests (~1000 generated cases each) |
| Phase 1 | `tests/test_rope_maintenance.py` | 16 rebalance/consolidate tests |
| Phase 2 | `tests/test_rope_concurrent.py` | 9 multi-threaded stress tests |
| Phase 3 | `src/rope/vllm_compat.py` | RopeBlockManager with vLLM-style API |
| Phase 3 | `tests/test_vllm_compat.py` | 9 vLLM-API tests |
| Phase 3 | `examples/hf_integration.py` | distilgpt2 + RopeBlockManager end-to-end demo |
| Bench | `bench/bench_rope.py` | microbenchmarks (single-op throughput) |
| Bench | `bench/bench_prefix_sharing.py` | structural sharing (97% memory savings) |
| Bench | `bench/stress_test.py` | multi-tenant concurrent stress |
| Bench | `bench/bench_with_model.py` | distilgpt2 + rope (rope is 0.14% of model cost) |
| Bench | `bench/REPORT.md` | honest performance report for the book |
| Issues | `GITHUB_ISSUES.md` | 36 prioritized issues (P0–P3) |

**Test summary:** 75 passed, 1 skipped (Unix-only), 79% coverage.

## Quick start

```powershell
# Install (editable) and test
pip install -e ".[test]"
pytest -v

# Run the HF integration demo (requires transformers + torch)
pip install -e ".[hf]"
python -m examples.hf_integration --max-new-tokens 30

# Run the benchmarks
python -m bench.bench_rope
python -m bench.bench_prefix_sharing
python -m bench.stress_test
python -m bench.bench_with_model   # downloads distilgpt2

# Or read the report
cat bench/REPORT.md
```

The HF demo loads `distilgpt2` (~80M params), runs generation, then
demonstrates beam-search-style branching and WARM-tier spill. Runs in
<2s on CPU.

## Project layout

    KV_Cache/                                    (repo root)
      .gitignore
      LICENSE                                    Apache 2.0
      README.md                                  this file
      pyproject.toml                             package metadata
      tech_note.md                               design proposal (draft v0.2)
      KV_Cahce_PAT.txt                           gitignored, used during setup only
      src/rope/
        __init__.py                              public API
        types.py                                 NodeID, Tier, Delta, NULL_NODE
        core.py                                  PersistentRope (the data structure)
        vllm_compat.py                           RopeBlockManager (vLLM-style API)
      tests/
        test_rope_basic.py                       unit tests (33)
        test_rope_property.py                    property tests (9)
        test_rope_maintenance.py                 rebalance/consolidate tests (16)
        test_rope_concurrent.py                  concurrent stress tests (9)
        test_vllm_compat.py                      vLLM-API tests (9)
      bench/
        bench_rope.py                            microbenchmarks (single-op)
        bench_prefix_sharing.py                  structural sharing analysis
        stress_test.py                           multi-tenant concurrent stress
        bench_with_model.py                      real-model (distilgpt2) integration
        REPORT.md                                honest performance report
      examples/
        hf_integration.py                        distilgpt2 + rope end-to-end
      rust/                                      Rust port scaffold (lock-free)

## What's NOT shipped (and why)

- **Real lock-free Bw-tree in Rust.** The scaffold exists at `rust/`
  but the implementation is not done. The Python reference uses a
  global root lock for thread safety, which is correct but not
  lock-free. The actual Bw-tree mapping-table indirection with CAS
  is the genuine research contribution and lives in the Rust port
  (per the tech note, contribution angle C).

- **vLLM runtime integration.** vLLM is not installed on this machine
  and has rough Windows support. The `RopeBlockManager` class exposes
  a vLLM-compatible interface that the user can plug in by:
  1. Installing vLLM on a Linux/CUDA machine
  2. Subclassing or monkey-patching `vllm.core.block_manager.BlockManager`
  3. Routing the scheduler's `allocate` / `free` / `fork` / `swap_*` /
     `append_slot` calls to `RopeBlockManager`

- **Rebalance across multiple live roots.** The current `rebalance()`
  operates on the rope's main root and may drop internals that other
  live roots still reference (filed as P0-11 in `GITHUB_ISSUES.md`).
  The single-root rebalance in the bench is correct; the multi-root
  case needs refcount-aware node reclamation.

## Open research questions (from tech_note §7)

1. Optimal block size `B` for the rope leaf
2. Lock-free concurrent rope (Bw-tree style) for multi-tenant serving
3. Differentiable indexing (RMI over rope) for O(1) lookups
4. Mamba / SSM integration as a heterogeneous leaf type
5. Cross-machine rope: distributed vs. per-node with hash-keyed cross-node

## License

Apache 2.0. See [`LICENSE`](./LICENSE).

## Author

Gerald Enrique Nelson Mc Kenzie — lordxmen2k@gmail.com
