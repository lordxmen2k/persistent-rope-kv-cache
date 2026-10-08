# GitHub Issues — Persistent Rope

Source: *The Persistent Rope: Position-Keyed Indexing for the LLM KV Cache* (DOI 10.5281/zenodo.23241292). Each issue below is referenced from the book's honest framing of "what doesn't work" / "what's left as future work." Issues are ordered by priority (P0 = must-do, P1 = high-value, P2 = nice-to-have, P3 = bookkeeping).

When pasting into GitHub, replace `[BOOK-CITE]` with the actual chapter reference and link to https://github.com/lordxmen2k/persistent-rope-kv-cache.

---

## P0-01: Lock-free Rust port — implement the production data structure

**Priority:** P0 (blocks the entire research claim)
**Labels:** `enhancement`, `rust`, `lock-free`, `good first issue` (for the scaffolding tasks)
**Estimated effort:** ~3 months

### Background

The book claims the rope is lock-free, but the entire `rust/` directory is scaffold-only. From Ch 5: "the port itself is scaffold-only at the time of writing." From Ch 22: "The port is not yet implemented. The `rust/` directory in the project contains a scaffold — a `Cargo.toml` with the dependencies, a `src/lib.rs` with the module structure, and a few stub functions."

The book is the documentation for the Python reference. The lock-free Rust port is what the book claims is the research contribution. Until this exists, the book's production claims are unverified.

### What needs to be done

Create the following files under `rust/`:

- `Cargo.toml` with deps: `crossbeam-epoch`, `crossbeam-skiplist`, `parking_lot`, `parking_lot_core`
- `src/lib.rs` — module structure and re-exports
- `src/types.rs` (~200 lines) — `NodeId` (newtype u64), `Tier` enum, `DeltaKind`, `Delta`, `Stats`
- `src/node.rs` (~400 lines) — `Node` struct with `AtomicU64 refcount`, `AtomicU8 tier`, `AtomicPtr<Delta>`, `Vec<AtomicNodeId>` children for internals
- `src/rope.rs` (~1500 lines) — `PersistentRope` with all operations as CAS loops holding an epoch guard
- `src/block_manager.rs` (~400 lines) — `RopeBlockManager` matching the Python adapter's API
- `tests/unit_tests.rs` (~600 lines) — port the 33 unit tests from `test_rope_basic.py`
- `tests/property_tests.rs` (~400 lines) — port the 9 property tests from `test_rope_property.py`
- `tests/concurrent_tests.rs` (~400 lines) — port the 9 concurrent tests from `test_rope_concurrent.py`
- `tests/vllm_compat_tests.rs` (~300 lines) — port the 9 vLLM-compat tests from `test_vllm_compat.py`

Total: ~3,000 lines of Rust (matches the book's estimate).

### Acceptance criteria

- All 59 Python tests pass when ported to Rust (1 Unix-only test can be skipped on Windows)
- `cargo build --release` produces a static lib
- `cargo bench` runs the same operations as `bench/bench_rope.py`
- No `unsafe` blocks except where required for atomic intrinsics
- Every `read` holds an `epoch::Guard`; every `write` does the CAS + `guard.defer()` for old nodes

### Book reference

Ch 1, 5, 6, 9, 22. Direct quote: "The lock-free implementation is the real research contribution."

---

## P0-02: Implement `consolidate()` — currently a stub

**Priority:** P0
**Labels:** `bug`, `python`, `help wanted`
**Estimated effort:** 1-2 weeks

### Background

Ch 13 says: *"The `consolidate` method, which is a stub in the reference."* Ch 6 says: *"The Python reference does not auto-consolidate. The caller is expected to call `consolidate()` periodically (e.g., after every N mutations, where N is a config parameter)."*

The method exists in the `PersistentRope` class but is a no-op. The test suite (Ch 13 §13.4) has 79% coverage precisely because `consolidate` is uncovered.

### What needs to be done

In `src/rope/core.py`:

1. **Implement `consolidate()`:** walk the delta chain of every node, build a new base state that incorporates all deltas, allocate a new node, and update the mapping table. The old delta records and the old base are queued for `collect_garbage()`.

2. **Add `auto_consolidate` config flag** (default `False`): when `True`, the rope runs consolidation in a background thread after every `consolidation_threshold` mutations. Uses `threading.Lock` to coordinate with foreground mutations.

3. **Add `consolidation_threshold` parameter to `__init__`** (default 1024, already in the constructor signature but not wired).

4. **Add coverage tests** in `tests/test_rope_basic.py` to cover the new `consolidate` paths.

### Acceptance criteria

- `r.consolidate()` returns the number of nodes consolidated (currently returns 0)
- After 1000 mutations, calling `consolidate()` produces a new base state for every node with a delta chain
- The `delta_head` field correctly drains to the new base after consolidation
- Coverage goes from 79% to ~85%
- The new test `test_consolidate_after_n_mutations` passes

### Book reference

Ch 6 §6.4, Ch 13 §13.4.

---

## P0-03: Replace deferred-free list with `guard.defer()` (Rust port)

**Priority:** P0
**Labels:** `enhancement`, `rust`, `memory-safety`
**Estimated effort:** 1 week (after P0-01 lands)

### Background

Ch 9: *"The lock-free Rust port will use `crossbeam-epoch`'s `defer` to schedule the cleanup, which is automatically drained when the global epoch advances."* The Python reference's deferred-free list is "correct under the GIL" but "not the production-quality reclamation scheme."

### What needs to be done

In the Rust port (after P0-01):

1. Every rope operation that drops a node must call `guard.defer()` with a closure that frees the node, instead of appending to a Python list.
2. Remove the `_deferred_free: Vec<NodeId>` field.
3. Document that reads must call `epoch::pin()` to get a `Guard`, and that the guard is what prevents use-after-free.
4. Add a test that verifies the reclamation protocol: 1000 concurrent reads on the same node + 1 concurrent removal + 1 epoch advance → no use-after-free.

### Acceptance criteria

- No `Vec<NodeId>` field anywhere in the Rust code
- All reclamation is via `guard.defer()`
- 1000-iteration concurrent stress test passes with no segfaults

### Book reference

Ch 9, Ch 22 §22.4.

---

## P0-04: ABA-free mapping table with generation counter

**Priority:** P0
**Labels:** `enhancement`, `lock-free`, `correctness`
**Estimated effort:** 1-2 weeks (Rust port)

### Background

Ch 9: *"the rope's mapping table entries must be CAS-able in a way that is immune to the ABA problem... use a generation counter alongside the pointer: the pointer is 48 bits, the generation is 16 bits, and every CAS increments the generation. The ABA case requires the generation to wrap around 2^16 times between the read and the CAS, which is astronomically unlikely."*

The Python reference doesn't need this (GIL prevents ABA). The Rust port does.

### What needs to be done

1. Replace `AtomicPtr<Node>` with a `PackedPointerGeneration` newtype: 48 bits of pointer + 16 bits of generation.
2. Every CAS increments the generation by 1 (with wraparound at 2^16).
3. Add a test that simulates the ABA scenario: thread A reads entry with generation 0; thread B replaces the entry and the new entry happens to be at the same address; thread A's CAS must fail because the generation has changed.

### Acceptance criteria

- ABA test passes 10,000 iterations without false success
- Mapping table is still lock-free in the strict sense
- No regression in throughput

### Book reference

Ch 9 §9.6, Ch 4 §4.5.

---

## P0-05: Replace per-rope root lock with CAS on mapping table

**Priority:** P0
**Labels:** `enhancement`, `rust`, `lock-free`
**Estimated effort:** 1 week (Rust port)

### Background

Ch 6 §6.6: *"Single global root lock. The reference holds a per-rope lock around every mutation. This is the simplest way to make the rope thread-safe under the GIL, but it serializes all mutations. The lock-free port replaces the root lock with CAS on the mapping table entry for the root's parent."*

### What needs to be done

1. In `PersistentRope`, replace `_lock: threading.RLock` with `AtomicNodeId` on the main root field.
2. Every mutation does a CAS on the root field with a 48-bit packed pointer+generation (depends on P0-04).
3. Document that the CAS retry loop is progress (at least one thread makes progress), not wait-free.
4. Add a test that 100 concurrent `append` operations on the same rope all succeed (some may retry, but all must commit).

### Acceptance criteria

- 100 concurrent appends all succeed
- No `threading.Lock` or `threading.RLock` in the Python reference's `PersistentRope` class
- The Python reference is still correct (GIL provides serialization, but the API no longer relies on it)

### Book reference

Ch 6 §6.6, Ch 10 §10.1.

---

## P0-06: Replace per-node fine-grained locks with atomic loads

**Priority:** P0
**Labels:** `enhancement`, `lock-free`
**Estimated effort:** 1-2 weeks (Rust port)

### Background

Ch 6 §6.6: *"Per-node fine-grained locks. The reference holds a per-node lock when reading or modifying a node. The lock-free port replaces these with atomic loads on the node's `delta_head` and the use of `Acquire`/`Release` memory ordering to ensure that other threads see consistent state."*

### What needs to be done

1. Remove the `_lock: threading.Lock` field from `LeafNode` and `InternalNode`.
2. The `_deltas` field becomes `AtomicPtr<Delta>` (Rust) or stays as a Python list (Python reference, where the GIL provides serialization).
3. Reads of `_deltas` use `Acquire` ordering; writes use `Release` ordering.
4. Add a test: 100 concurrent reads on the same node + 1 concurrent delta append + 1 epoch advance → all reads return valid deltas.

### Acceptance criteria

- No `Lock` field in the `Node` struct
- Concurrent reads still return valid data
- The delta chain walk is lock-free

### Book reference

Ch 6 §6.6.

---

## P0-07: Atomic NodeID counter (`fetch_add(1, Relaxed)`)

**Priority:** P0
**Labels:** `enhancement`, `lock-free`
**Estimated effort:** 1 day (Rust port)

### Background

Ch 10: *"The `_id_lock` is a separate lock from the per-node locks. It is held only for the duration of the counter increment, which is microseconds. In the lock-free port, the counter is a `fetch_add(1, Ordering::Relaxed)` on an `AtomicU64` — a single atomic instruction, no lock."*

### What needs to be done

1. In `PersistentRope`, replace `_id_lock: threading.Lock` + `next_id: int` with `next_id: AtomicU64`.
2. Every `next_id` increment becomes `next_id.fetch_add(1, Ordering::Relaxed)`.
3. The never-reuse invariant is preserved automatically (monotonically increasing u64 wraps to 0 only after 2^64 allocations, which is unreachable in any practical system).

### Acceptance criteria

- No `_id_lock` field
- 1 billion sequential `next_id` increments produce 1 billion unique IDs
- Concurrent increments produce unique IDs (verified by a test that runs 1M concurrent increments and checks the count is exactly 1M)

### Book reference

Ch 10 §10.2.

---

## P0-08: `Arc<DashMap<NodeID, AtomicPtr<Node>>>` for the mapping table

**Priority:** P0
**Labels:** `enhancement`, `rust`, `lock-free`
**Estimated effort:** 1 week (Rust port)

### Background

Ch 10: *"In the lock-free port, `_put` is a CAS on the mapping table entry, `_get` is an atomic load, and the type-checked accessors are simple pointer dereferences. The Python reference uses a `dict` because the GIL makes it safe; the lock-free port uses an `Arc<DashMap<NodeID, AtomicPtr<Node>>>` for the same reason `crossbeam-epoch` uses `AtomicPtr` everywhere."*

### What needs to be done

1. Replace `Mutex<HashMap<NodeID, *mut Node>>` (or similar) with `Arc<DashMap<NodeID, AtomicPtr<Node>>>`.
2. `put` does `entry.insert(AtomicPtr::new(node_ptr))`.
3. `get` does `entry.value().load(Acquire)`.
4. `remove` does `entry.remove()` and uses `guard.defer()` to free the node.
5. ABA protection comes from P0-04 (packed pointer+generation).

### Acceptance criteria

- 10K concurrent reads on the same key all return the same value
- A read that races with a remove returns either the old value or `None` (never a use-after-free)
- No global lock on the mapping table

### Book reference

Ch 10 §10.3.

---

## P0-09: 32-way internal node layout (alternative to binary tree)

**Priority:** P0
**Labels:** `enhancement`, `data-structure`
**Estimated effort:** 2-3 weeks (Rust port)

### Background

Ch 7: *"The cost of `unlink` is `O(L/B)` — the renumbering dominates. This is the main inefficiency of the binary tree layout. A 32-way tree would only need to renumber the affected leaves, not the entire tree, because the parent nodes' `start`/`end` are aggregates of the children."*

Ch 7: *"if the workload is beam-search-heavy, 32-way is better; if the workload is long-context inference, binary is better."*

### What needs to be done

1. Add a `layout: Literal["binary", "32way"]` config parameter to `PersistentRope.__init__` (default `"binary"`).
2. When `layout="32way"`, use `InternalNode` with `Vec<NodeId>` of up to 32 children per node (instead of 2).
3. The `start`/`end` fields of an internal node become aggregates of the children's `start`/`end`.
4. Renumbering on `unlink`/`fork` becomes O(L/B) for 32-way vs O(L) for binary.
5. Add a benchmark comparing both layouts on a beam-search workload and a long-context workload.

### Acceptance criteria

- 32-way layout passes the same 59 tests as the binary layout
- Beam-search benchmark shows 5-10× faster unlink/fork with 32-way
- Long-context benchmark shows 2-5× slower resolve/get_kv with 32-way (acceptable trade-off)
- `tree_depth` is bounded by `O(log_32(L/B))` instead of `O(log_2(L/B))`

### Book reference

Ch 7 §7.2, §7.5.

---

## P0-10: Deferred renumbering to a background thread

**Priority:** P0
**Labels:** `enhancement`, `performance`
**Estimated effort:** 1-2 weeks (Rust port)

### Background

Ch 7: *"In the lock-free port, the renumbering would be deferred to a background thread (because the renumbering is read-only on the old root and write-only on the new root, the two roots can coexist). The reference does the renumbering synchronously."*

Ch 14: *"the production port would defer the renumbering to a background thread... which would let the foreground thread return immediately."*

### What needs to be done

1. In the Rust port, `unlink`/`fork`/`share_prefix`/`evict`/`promote` return the new root immediately without renumbering.
2. A background thread consumes renumbering tasks from a channel.
3. The renumbering walks the surviving tree and updates `start`/`end` fields.
4. The renumbering is checked lazily on the next read: if a node's `start`/`end` is stale, the read triggers a synchronous renumber (or a wait for the background thread to catch up).

### Acceptance criteria

- `unlink` returns within 1 microsecond (no renumbering in the hot path)
- After 1000 unlinks in a row, the tree is fully renumbered within 100 milliseconds
- No incorrect `start`/`end` values are ever returned to the caller (stale values trigger a re-renumber)
- Background thread can be paused and resumed

### Book reference

Ch 7 §7.2, Ch 14 §14.2.

---

## P0-11: `rebalance_to_root()` corrupts live multi-tenant roots

**Priority:** P0
**Labels:** `bug`, `python`, `correctness`
**Estimated effort:** 3-5 days
**Discovered:** 2026-10-08 while building the multi-tenant stress test (`bench/stress_test.py`)

### Background

`rebalance()` in `src/rope/core.py` operates on `self._root_id` and
unconditionally drops the old internals via `_drop_internals()` into the
deferred-free list, then calls `collect_garbage()` which deletes them
from the mapping table. When the rope has multiple live roots (e.g.
32 sequences built by `append_to` from a common prefix), the prefix's
internals are **shared** between the live roots. Dropping them
unconditionally breaks the other live roots — `r.length_of(other_root)`
returns 0 after rebalance because the shared internals are gone.

Concrete reproduction:

```python
r = PersistentRope(block_size=16)
r.append(b"P" * 200)
common = r.root_id()
roots = [r.append_to(common, bytes([i % 26 + ord('a')]) * 30) for i in range(32)]
# roots[0] is a 230-token rope (200 shared + 30 suffix)
r.append_to(roots[0], b"a" * 5000)  # grow it
assert r.length_of(roots[0]) == 5230
r.rebalance_to_root()  # operates on _root_id, which is still the original 200-token common
assert r.length_of(roots[0]) == 0  # BROKEN: shared internals were dropped
```

### What needs to be done

1. **Make `rebalance(root)` rebalance a specific root**, not just
   `self._root_id`. The new signature returns a new root without
   touching `_root_id`.

2. **In `rebalance()`, only drop internals whose refcount == 1.** A
   shared internal (refcount > 1) is still in use by another live root
   and must not be freed. The current `_drop_internals()` ignores
   refcount.

3. **Audit `append_to` and `unlink_from` to bump refcount on shared
   subtrees.** When a new root is created that aliases the old root's
   subtree (the new root "keeps" the prefix), the prefix subtree's
   refcount must be incremented. This is the structural-sharing
   invariant that `fork()` already honors but `append_to` and
   `unlink_from` do not.

4. **Add a regression test** in `tests/test_rope_basic.py` that builds
   N roots from a common prefix, calls `rebalance()` on one of them,
   and asserts that the other N-1 roots are still readable.

### Acceptance criteria

- `r.rebalance(some_root)` rebalances that specific root and returns a
  new NodeID, leaving the other live roots intact.
- `r.length_of(other_root)` returns the original value after
  rebalancing any other root.
- A new test `test_rebalance_preserves_multi_tenant_roots` passes.
- The `bench/stress_test.py` runs without the multi-root rebalance
  workaround (currently it tests rebalance on a fresh single-root rope
  to avoid the bug).

### Book reference

Ch 7 §7.4, Ch 15 §15.3 (multi-tenant design).

---

## P1-11: GPU attention kernel that consumes the rope's snapshot

**Priority:** P1
**Labels:** `enhancement`, `gpu`, `cuda`
**Estimated effort:** ~1 month

### Background

Ch 20 §20.2 #2: *"The GPU attention kernel is not implemented. The `RopeBlockManager` returns a snapshot in the form `(node_id, byte_offset, length)`. A real GPU kernel would consume this format and issue coalesced loads. The kernel is not written; the rope is the index, not the kernel."*

### What needs to be done

Write a CUDA / Triton / HIP kernel that:

1. Takes the rope's snapshot as input (an array of `(node_id, byte_offset, length)`).
2. For each query token in the attention computation, issues a coalesced load of the K/V tensors from the GPU memory pool, where the memory pool is keyed by `node_id`.
3. Computes the attention output.
4. Calls `append_slot` on the rope with the new K/V tensor.

Integration with FlashAttention, FlashInfer, and vLLM's PagedAttention is a separate effort.

### Acceptance criteria

- A simple attention kernel works on distilgpt2 (small model, fits in single GPU).
- The kernel matches the reference attention output to floating-point precision.
- A benchmark shows the rope's snapshot format has no measurable overhead vs. vLLM's PagedAttention block table.

### Book reference

Ch 8 §8.4, Ch 20 §20.2.

---

## P1-12: HBM/DRAM/NVMe transfer engine (async memory copies)

**Priority:** P1
**Labels:** `enhancement`, `tier-management`, `memory`
**Estimated effort:** ~2 months

### Background

Ch 8: *"The reference implementation does not actually move data to NVMe; it just marks the leaf as COLD. The production port would integrate with a transfer engine (like Mooncake's) to move the data."*

Ch 7: *"In production, the tier change also involves a memory copy (HBM to DRAM, or vice versa). The Python reference does not actually move the kv_blob; it just changes the tag. The production port would issue the memory copy as part of the tier change, with the copy happening asynchronously on a background thread so the lock-free index update is not blocked on the memory bandwidth."*

### What needs to be done

1. Integrate with Mooncake's Transfer Engine (or NCCL, or a custom library).
2. `evict` triggers an async copy from HBM to DRAM (or DRAM to NVMe).
3. `promote` triggers an async copy in the reverse direction.
4. The rope's index update (tier tag) is synchronous; the actual data movement is async.
5. A callback is invoked when the transfer completes (updates the `tier` tag to confirm the new tier is populated).

### Acceptance criteria

- A `TransferEngine` interface is defined and Mooncake is one implementation.
- A 1GB HBM→DRAM copy completes in under 100ms on a real GPU.
- The rope's index update is not blocked on the copy (lock-free property preserved).
- A failure (transfer error) is reported via the callback and the tier is rolled back to the old value.

### Book reference

Ch 7 §7.3, Ch 8 §8.1, Ch 22 §22.6.

---

## P1-13: Scheduler that uses the rope's stats for admission/eviction decisions

**Priority:** P1
**Labels:** `enhancement`, `scheduler`
**Estimated effort:** ~1 month

### Background

Ch 20 §20.2 #4: *"The scheduler is not implemented. A real serving system has a scheduler that decides which sequences to admit, which to evict, when to swap in, and when to swap out. The rope is the data structure; the scheduler is a separate concern. The reference has no scheduler."*

### What needs to be done

Build a scheduler that:

1. Reads the rope's `stats()` (sequence count, total tokens, tier distribution, shared leaf count).
2. Decides which sequences to admit when memory is full (LRU, LFU, or rope's structural sharing as a tiebreaker).
3. Decides which sequences to evict (LRU within the rope, or based on access patterns).
4. Decides when to swap in (when a HOT sequence is needed but only WARM is present).
5. Decides when to swap out (when HOT is over capacity, demote to WARM).
6. Uses `swap_in`/`swap_out` (or `promote`/`evict`) to apply the decisions.

### Acceptance criteria

- The scheduler can run continuously for 1 hour without leaking memory.
- A 100-sequence workload with 60% memory utilization maintains steady-state.
- Admission decisions prefer sequences with high structural sharing (better cache utilization).
- The scheduler exposes its decisions as a stream of events (for observability).

### Book reference

Ch 8 §8.1, Ch 20 §20.2.

---

## P1-14: Model runner that calls `append_slot` per decode step

**Priority:** P1
**Labels:** `enhancement`, `integration`
**Estimated effort:** ~1 month

### Background

Ch 20 §20.2 #5: *"The model runner is not implemented. The model runner is the part of the serving system that calls the rope's `append_slot` for every decoded token. The reference has no model runner; the HuggingFace demo uses HuggingFace's own model runner and a shadow rope that tracks the cache state."*

### What needs to be done

Build a model runner that:

1. Wraps a HuggingFace / vLLM / SGLang model.
2. For each forward pass, calls `rope.resolve(seq_id)` to get the K/V block table.
3. Passes the block table to the GPU attention kernel (P1-11).
4. After the forward pass, calls `rope.append_slot(seq_id, kv_data)` for each new K/V tensor.
5. Calls `rope.fork(parent_seq_id, child_seq_id)` for beam search.
6. Calls `rope.evict(seq_id, n_tokens)` when the rope is at capacity.

### Acceptance criteria

- A 1B-parameter model generates 100 tokens/sec on a single H100.
- The rope's K/V block table matches the model's actual K/V tensors.
- A 32-way beam search runs without memory leaks.

### Book reference

Ch 12, Ch 18, Ch 20 §20.2.

---

## P1-15: Tensor view for `kv_blob` instead of bytes

**Priority:** P1
**Labels:** `enhancement`, `data-format`
**Estimated effort:** 1-2 weeks

### Background

Ch 10: *"The `kv_blob` field is `bytes` in the reference; in the production port it would be a tensor view. The reference treats one byte as one token, which is sufficient for structural testing but not for actual K/V data."*

Ch 7: *"The Python reference is the implementation; the production port would return a tensor view instead of bytes."*

### What needs to be done

1. Add a `kv_tensor: torch.Tensor` (or `numpy.ndarray`) variant of `kv_blob`.
2. The rope accepts either `bytes` (for tests) or `torch.Tensor` (for production).
3. The reference keeps `bytes` for compatibility; a new `TensorRope` class is added for tensor inputs.
4. The Rust port's `kv_blob: &[u8]` parameter becomes `kv_blob: TensorView<'_>` (a view into a `BTreeMap<NodeId, Arc<Tensor>>`).

### Acceptance criteria

- `TensorRope.append` accepts a `torch.Tensor` of shape `(num_tokens, num_layers, num_kv_heads, head_dim)`.
- The reference tests still pass with `bytes`.
- A benchmark shows `TensorRope` is no slower than the HuggingFace KV cache for autoregressive generation.

### Book reference

Ch 10 §10.2, Ch 7 §7.3.

---

## P1-16: Async memory copy on tier change

**Priority:** P1
**Labels:** `enhancement`, `tier-management`, `async`
**Estimated effort:** 1-2 weeks (depends on P1-12)

### Background

Ch 7: *"The production port would issue the memory copy as part of the tier change, with the copy happening asynchronously on a background thread so the lock-free index update is not blocked on the memory bandwidth."*

### What needs to be done

1. `evict` and `promote` return a `Future` that completes when the memory copy finishes.
2. The rope's index update is synchronous; the data movement is async.
3. The `tier` tag is updated to the new tier optimistically; if the copy fails, the tag is rolled back.

### Acceptance criteria

- A `evict(seq_id, n_tokens)` call returns within 1 microsecond (no waiting for the copy).
- A `swap_complete` callback is invoked when the copy is done.
- A failure (e.g., DRAM is full) rolls back the tier tag and returns an error.

### Book reference

Ch 7 §7.3, Ch 22 §22.6.

---

## P1-17: Checkpointing (serialize mapping table to disk)

**Priority:** P1
**Labels:** `enhancement`, `crash-recovery`
**Estimated effort:** 1 week

### Background

Ch 8: *"The Python reference does not implement checkpointing; the production port would. The checkpoint format is straightforward: serialize the mapping table (NodeID, tier, refcount, physical block address) to a flat file. The restore is the reverse: read the file, rebuild the mapping table."*

### What needs to be done

1. Add `checkpoint(path: Path) -> None` to `PersistentRope` and `RopeBlockManager`.
2. The checkpoint is a JSON or binary file with: every node's NodeID, start, end, tier, refcount, kv_offset, and the parent-child relationships.
3. Add `restore(path: Path) -> Self` that reads the file and rebuilds the rope.
4. A test: build a rope, checkpoint it, create a new rope, restore from the file, verify the two ropes are equal.

### Acceptance criteria

- A rope with 10,000 leaves checkpoints in under 1 second.
- A restored rope is byte-identical to the original (every NodeID, every refcount, every tier).
- A corrupt checkpoint file raises a clear error.

### Book reference

Ch 8 §8.6, Ch 6 §6.3.

---

## P1-18: vLLM fork with `RopeBackedBlockManager` drop-in

**Priority:** P1
**Labels:** `enhancement`, `vllm`, `integration`
**Estimated effort:** ~1 month

### Background

Ch 18: *"The integration is the production work that comes after the lock-free Rust port exists. The reference is sufficient to demonstrate the design and the API; the integration is a separate effort that requires vLLM's installation and a CUDA machine."*

Ch 20 §20.3 #7: *"The vLLM fork. The fork would be a `RopeBackedBlockManager` that drops into vLLM and enables the rope's design in production. The fork is estimated at ~1 month of focused work."*

### What needs to be done

1. Fork vLLM.
2. Create a new file `vllm/core/block_manager_rope.py` that subclasses `BlockManager` and delegates to a `RopeBlockManager`.
3. Register the new block manager in vLLM's engine.
4. Add a config flag `--block-manager=rope` to enable the rope-backed manager.
5. Run vLLM's existing test suite against the rope-backed manager.
6. Benchmark: rope-backed vLLM vs. stock vLLM on long-context chat, beam search, multi-tenant prefix caching.

### Acceptance criteria

- All vLLM tests pass with `--block-manager=rope`
- A long-context (32K tokens) chat workload runs without errors
- The rope-backed vLLM is at least as fast as stock vLLM for single-stream workloads
- The rope-backed vLLM is faster than stock vLLM for multi-tenant workloads with shared prefixes

### Book reference

Ch 18, Ch 20 §20.3.

---

## P1-19: SGLang integration (radix + rope composition)

**Priority:** P1
**Labels:** `enhancement`, `sglang`, `integration`
**Estimated effort:** ~2 months

### Background

Ch 20 §20.3 #6: *"The SGLang integration. The integration would replace SGLang's radix tree with the rope, or compose the two (radix for content-hash, rope for structural sharing). The integration is estimated at ~2 months of focused work."*

### What needs to be done

Choose one of two approaches:

**Option A: Replace.** Replace SGLang's radix tree with the rope entirely. The rope's structural sharing subsumes the radix tree's content-hash sharing. The trade-off is that the radix tree's content-hash discovery is replaced by explicit `fork` calls.

**Option B: Compose.** Use the radix tree for content-hash prefix matching (SGLang's strength) and the rope for structural-sharing branching (the rope's strength). The two compose: a `fork` in the rope returns a new root that the radix tree can index on, and the radix tree's `match_prefix` returns a NodeID range that the rope can use for a `share_prefix`.

Recommended: Option B.

### Acceptance criteria

- SGLang's existing test suite passes with the rope-augmented radix tree
- A beam-search workload is faster with the rope's structural sharing than with the radix tree alone
- A multi-tenant prefix workload is faster with the radix tree's content-hash discovery than with the rope alone

### Book reference

Ch 3, Ch 20 §20.3.

---

## P1-20: Iterative (non-recursive) tree traversal

**Priority:** P1
**Labels:** `enhancement`, `rust`
**Estimated effort:** 1-2 days (Rust port)

### Background

Ch 10: *"The recursive implementation is fine for the Python reference; the lock-free port would use an iterative implementation to avoid stack overflow on very deep trees."*

### What needs to be done

Replace the recursive tree walks in the Rust port's `rope.rs` with iterative versions using an explicit stack (`Vec<(NodeId, State)>`).

### Acceptance criteria

- No recursive function calls in the Rust port's tree traversal
- A tree with 1M leaves (depth ~17 in binary, ~6 in 32-way) doesn't overflow the stack
- Throughput is the same or better than the recursive version (no function call overhead)

### Book reference

Ch 10 §10.6.

---

## P1-21: Periodic tree rebalancing

**Priority:** P1
**Labels:** `enhancement`, `data-structure`
**Estimated effort:** 2 weeks

### Background

Ch 7: *"The Python reference is intentionally naive; the production port would rebalance the tree periodically, or use a 32-way internal node layout to keep the depth bounded."*

### What needs to be done

1. Add a `rebalance()` method to `PersistentRope` that walks the tree and:
   - Splits internal nodes with more than 2 × `block_size` leaves on one side
   - Merges internal nodes with fewer than `block_size / 2` leaves on one side
   - Bounded by O(L) per call (full walk)
2. Add an `auto_rebalance` config flag that calls `rebalance()` in a background thread after every N mutations.
3. The rebalance produces a new root (the old root is unchanged, the new root is the rebalanced version).
4. Add a test: a 1M-leaf skewed tree becomes balanced after `rebalance()`.

### Acceptance criteria

- After `rebalance()`, the tree's depth is O(log N) for any N.
- `rebalance()` is O(L) and doesn't block `append`/`unlink`.
- The new tree passes all 59 existing tests.

### Book reference

Ch 7 §7.5.

---

## P1-22: Tier as separate metadata on the mapping table entry

**Priority:** P1
**Labels:** `enhancement`, `design-decision`
**Estimated effort:** 1 week (Rust port)

### Background

Ch 8: *"A more sophisticated design would carry the tier as a separate metadata field on the mapping table entry rather than on the leaf itself; the tier change becomes a single CAS on the mapping table, and the leaf state is unchanged. The reference keeps the tier on the leaf for simplicity; the production port would consider the trade-off."*

### What needs to be done

Decide and implement:

**Option A (current):** tier is on the leaf. Tier change requires path-copying.
**Option B:** tier is on the mapping table entry. Tier change is a single CAS.

The book says "the production port would consider the trade-off" — so this is a design decision, not a clear directive. Recommend Option B for the lock-free port.

### Acceptance criteria

- Tier change is a single CAS on the mapping table entry
- No path-copying on tier change
- Tier is independent of the leaf's `kv_blob`

### Book reference

Ch 8 §8.1, Ch 22 §22.6.

---

## P1-23: Comprehensive error hierarchy

**Priority:** P1
**Labels:** `enhancement`, `error-handling`
**Estimated effort:** 1-2 days

### Background

Ch 20 §20.2 #7: *"The error handling is basic. The rope raises `RopeError`, `EmptySequenceError`, and `OutOfRangeError`. The reference does not raise a comprehensive set of errors for every possible failure mode. The production port would have a more complete error hierarchy."*

### What needs to be done

Expand the error hierarchy in both Python and Rust:

```
RopeError (base)
├── EmptySequenceError
├── OutOfRangeError
├── InvalidOperationError (e.g., evict on empty sequence)
├── ConcurrentModificationError (e.g., two threads try to allocate same node)
├── CapacityExceededError (e.g., can_append returns False)
├── InvalidNodeIdError (e.g., get_kv on a freed NodeID)
├── TierTransitionError (e.g., promote on a leaf that's already HOT)
├── MappingTableCorruptedError (rare; mapping table fails consistency check)
└── InternalError (catch-all; should never be raised)
```

Each error has a clear message and (where applicable) the rope's state at the time of the error.

### Acceptance criteria

- Each error class has a unit test that triggers it.
- Error messages include enough context to debug (sequence ID, position, etc.).
- The error hierarchy is documented in the README.

### Book reference

Ch 20 §20.2 #7.

---

## P1-24: Signal-safe reclamation (Rust port)

**Priority:** P1
**Labels:** `enhancement`, `rust`, `signal-safety`
**Estimated effort:** 1 day (Rust port)

### Background

Ch 20 §20.2 #10: *"The reference's deferred free is not signal-safe. The reference's `collect_garbage()` is not safe to call from a signal handler. The production port's `guard.defer()` from `crossbeam-epoch` is signal-safe."*

### What needs to be done

The Rust port already uses `guard.defer()` from `crossbeam-epoch` (after P0-03). Verify that calling `epoch::pin()` and then a read operation is safe from a signal handler. If not, document which operations are signal-safe and which are not.

### Acceptance criteria

- All read operations are signal-safe (no allocations, no locks)
- `defer()` is signal-safe (deferred closures run on a background thread, not in the signal handler)
- A test: send SIGUSR1 to a thread holding an `epoch::Guard`, verify the read completes correctly

### Book reference

Ch 20 §20.2 #10.

---

## P1-25: Concurrency tests for the lock-free port

**Priority:** P1
**Labels:** `testing`, `rust`, `concurrency`
**Estimated effort:** 1 month (Rust port)

### Background

Ch 13: *"The concurrent tests are the verification that the Python reference's lock-based concurrency is safe. The lock-free port would use the same tests, with the additional verification that the lock-free implementation does not lose updates under contention."*

### What needs to be done

1. Port the 9 concurrent tests from `test_rope_concurrent.py` to Rust.
2. Add new tests that verify the lock-free guarantees:
   - 256 threads doing concurrent `append` — no lost updates
   - 256 threads doing concurrent `fork` — refcounts are correct
   - ABA stress test — 10K iterations of the ABA scenario, no false CAS success
3. Use `loom` (Rust's concurrency permutation tester) for systematic verification.

### Acceptance criteria

- All 9 ported tests pass
- 256-thread stress tests pass
- Loom finds no execution-order bugs

### Book reference

Ch 13 §13.3, Ch 22 §22.5.

---

## P2-26: Rust port API documentation (rustdoc)

**Priority:** P2
**Labels:** `documentation`, `rust`
**Estimated effort:** 1-2 weeks (after P0-01 lands)

### Background

Ch 20 §20.2 #8: *"The documentation is incomplete. This book is the API reference for the Python reference. The Rust port's API is not documented; the GPU kernel's API is not documented; the transfer engine's API is not documented."*

### What needs to be done

1. Write rustdoc comments for every public type and method in the Rust port.
2. The docs are a port of the Python reference's docstrings, not a rewrite.
3. The docs include:
   - Method signature
   - One-paragraph description
   - Algorithm complexity
   - Thread-safety guarantees
   - Cross-references to related methods

### Acceptance criteria

- `cargo doc --no-deps` produces a complete rustdoc
- Every public type and method has a doc comment
- The docs build without warnings

### Book reference

Ch 20 §20.2 #8.

---

## P2-27: End-to-end benchmark (rope vs. vLLM, SGLang, Mooncake)

**Priority:** P2
**Labels:** `benchmark`, `performance`
**Estimated effort:** ~2 months (after P0-01, P1-18, P1-19)

### Background

Ch 20 §20.3 #8: *"The end-to-end benchmark. The benchmark would compare the rope-backed serving system to vLLM's stock serving system on a representative workload (long-context chat, beam search, multi-tenant prefix caching, agent rollouts). The benchmark is estimated at ~2 months of focused work."*

### What needs to be done

Build a benchmark harness that:

1. Sets up a rope-backed vLLM fork (P1-18) and a stock vLLM.
2. Runs a standard workload suite:
   - **Long-context chat:** 32K-token conversations, 1000 requests/sec.
   - **Beam search:** 32-way beam search on a 4K-token prompt, 200 generated tokens.
   - **Multi-tenant prefix caching:** 100 tenants, each with a 4K-token system prompt, 200 user queries.
   - **Agent rollouts:** 10-step reasoning chains, 32K context per step.
3. Measures:
   - Throughput (tokens/sec)
   - P50/P99 latency
   - Memory utilization
   - Cache hit rate
4. Produces a side-by-side comparison table.

### Acceptance criteria

- The benchmark runs reproducibly on a single H100
- The rope-backed system is at least as fast as stock vLLM on every workload
- The rope-backed system is faster than stock vLLM on multi-tenant workloads with shared prefixes
- The benchmark results are published in a CSV file with the benchmark harness

### Book reference

Ch 14, Ch 20 §20.3.

---

## P2-28: Representative benchmarks (vs. microbenchmarks)

**Priority:** P2
**Labels:** `benchmark`
**Estimated effort:** 2 weeks

### Background

Ch 20 §20.2 #9: *"The benchmarks are not representative. The microbenchmarks in `bench/bench_rope.py` measure the rope's operations in isolation. A real serving system has a complex interaction between the rope, the GPU kernel, the scheduler, and the transfer engine. The microbenchmarks do not measure the end-to-end throughput."*

### What needs to be done

Add new benchmarks to `bench/`:

1. `bench_e2e.py` — runs a full autoregressive generation loop with the rope, measures tokens/sec.
2. `bench_with_scheduler.py` — simulates a scheduler making eviction/eviction decisions, measures fairness and hit rate.
3. `bench_with_transfer.py` — simulates tier changes, measures the cost of the memory copy.

### Acceptance criteria

- Each new benchmark produces a CSV file with results
- The benchmarks are reproducible on a single GPU
- The benchmarks measure realistic workloads (not just isolated operations)

### Book reference

Ch 14, Ch 20 §20.2 #9.

---

## P2-29: Benchmark publication (paper)

**Priority:** P2
**Labels:** `paper`, `publication`
**Estimated effort:** ~1 month (after P2-27)

### Background

Ch 20 §20.3 #10: *"The benchmark publication. The publication would compare the rope's serving system to vLLM, SGLang, and Mooncake on a standard workload. The publication is estimated at ~1 month of focused work, including the paper writing and the conference submission."*

### What needs to be done

Write a paper that:

1. Describes the rope's design (Ch 5-9 of the book).
2. Describes the implementation (Ch 10-11 of the book).
3. Reports the benchmark results (P2-27).
4. Compares the rope-backed serving system to vLLM, SGLang, and Mooncake.
5. Submits to a venue like SOSP, OSDI, MLSys, or NSDI.

### Acceptance criteria

- The paper is 12-15 pages (the standard format for systems conferences).
- The paper has clear figures and tables.
- The paper is submitted to a peer-reviewed venue.

### Book reference

Ch 20 §20.3 #10.

---

## P2-30: Tier change callback API

**Priority:** P2
**Labels:** `enhancement`, `api`
**Estimated effort:** 1 day

### Background

Ch 22 §22.6: *"The port's integration with the transfer engine is a callback that observes tier changes and issues the HBM/DRAM/NVMe transfers. The callback is registered with the rope and is called whenever a tier tag changes."*

### What needs to be done

1. Add `register_tier_change_callback(fn)` to `PersistentRope`.
2. The callback is called with `(node_id: NodeId, old_tier: Tier, new_tier: Tier)`.
3. The callback is invoked synchronously after the tier tag is updated.
4. Multiple callbacks can be registered; they are called in registration order.

### Acceptance criteria

- A test: register a callback, call `evict(seq_id, 1)`, verify the callback is called with the correct arguments.
- The callback can be unregistered.
- Exceptions in callbacks are caught and logged (don't break the rope's internal state).

### Book reference

Ch 22 §22.6.

---

## P2-31: Capacity parameter and `can_append` semantics

**Priority:** P2
**Labels:** `enhancement`, `api`
**Estimated effort:** 1 day

### Background

The book discusses capacity, but the reference has no cap. `can_append(seq_id)` exists but always returns `True`. The production port needs a real capacity check.

### What needs to be done

1. Add a `max_sequences: int` config parameter to `RopeBlockManager.__init__` (default `None` for unlimited).
2. `can_append(seq_id)` returns `False` when the rope has reached the capacity.
3. `allocate(seq_id, n_tokens)` raises `CapacityExceededError` if the allocation would exceed the capacity.
4. Add a test: set capacity to 10, try to allocate 11 sequences, verify the 11th raises.

### Acceptance criteria

- `can_append` returns `False` at capacity
- `allocate` raises `CapacityExceededError` at capacity
- The capacity check is O(1) (not a full tree walk)
- The reference's existing tests still pass with `max_sequences=None`

### Book reference

Ch 11 §11.2, Ch 20 §20.2.

---

## P2-32: Auto-consolidation hook

**Priority:** P2
**Labels:** `enhancement`, `python`
**Estimated effort:** 1 week (after P0-02 lands)

### Background

Ch 6: *"The Python reference does not auto-consolidate. The caller is expected to call `consolidate()` periodically (e.g., after every N mutations, where N is a config parameter). The production Rust port will use an opportunistic consolidation strategy: piggyback the consolidation on regular operations, similar to the Bw-tree's design."*

### What needs to be done

1. After P0-02 (the `consolidate()` implementation), add an `auto_consolidate: bool` config flag (default `False`).
2. When `True`, a background thread calls `consolidate()` after every `consolidation_threshold` mutations.
3. The thread uses a `threading.Event` to coordinate with foreground mutations.
4. The thread can be paused and resumed.

### Acceptance criteria

- 1000 mutations trigger exactly one `consolidate()` call (when threshold is 1000)
- Foreground mutations are not blocked by the background thread
- The thread can be stopped and restarted

### Book reference

Ch 6 §6.4.

---

## P3-33: Skip `test_collect_garbage_signal_safe` on Windows (already done, just verify)

**Priority:** P3 (already implemented)
**Labels:** `testing`, `cross-platform`
**Estimated effort:** 5 minutes

### Background

Ch 20 §20.4: *"The skipped test is `test_collect_garbage_signal_safe`, which is Unix-only because it tests the signal-safe drain of the deferred-free list."*

The test is already skipped on Windows (verified by running the test suite). The Python reference is cross-platform.

### What needs to be done

Verify the test is correctly marked with `@pytest.mark.skipif(sys.platform == "win32", reason="...")`. If not, add the decorator.

### Acceptance criteria

- `pytest` shows the test as skipped on Windows
- The test is documented as Unix-only in its docstring

### Book reference

Ch 20 §20.4.

---

## P3-34: Coverage target (already 79%, accept the 80% target)

**Priority:** P3 (already accepted)
**Labels:** `testing`
**Estimated effort:** 0 (no action needed)

### Background

Ch 20 §20.4: *"The 100% target is not the goal. The 80% target is sufficient to verify the design; the remaining 20% is either hard to test (error paths) or not yet implemented (the lock-free port's reclamation)."*

### What needs to be done

No action. The book explicitly accepts the 80% target. Coverage at 79% is intentional. After P0-02 (consolidate) lands, coverage will rise to ~85%.

### Book reference

Ch 20 §20.4.

---

## P3-35: Internal API stability warning

**Priority:** P3
**Labels:** `documentation`
**Estimated effort:** 1 hour

### Background

Ch 20 §20.5: *"The internal API (the `_get_leaf`, `_iter_leaves`, etc. methods) is not stable and may change without notice. The reader should not rely on the internal API."*

### What needs to be done

Add a `__private__` marker to all internal methods (those starting with `_`) in the Python reference. Use `dir()` to verify all internal methods are marked.

### Acceptance criteria

- `PersistentRope.__dict__` lists all internal methods
- Each internal method has a docstring noting it's not part of the public API
- The README explicitly lists the public API methods (the ones without `_`)

### Book reference

Ch 20 §20.5.

---

## P3-36: License consistency check (all files reference Apache 2.0)

**Priority:** P3
**Labels:** `licensing`
**Estimated effort:** 30 minutes

### Background

The book is licensed under Apache 2.0. The repo should be consistent. From the user's persistent rule: any copyright/ownership string should be "Copyright 2026 Tico Internet LLC" / owner "Tico Internet LLC".

### What needs to be done

1. Grep the repo for any file that doesn't reference Apache 2.0.
2. Add the Apache 2.0 LICENSE file to the repo root.
3. Add a "Copyright 2026 Tico Internet LLC" line to every source file.
4. Update the README to mention the license.

### Acceptance criteria

- Every source file has the Apache 2.0 + Tico Internet LLC notice
- The LICENSE file is at the repo root
- `grep -r "Copyright" .` returns only "Tico Internet LLC"
- `grep -r "License" .` returns only "Apache 2.0"

### Book reference

User memory rule: "Use 'Tico Internet LLC' for ownership/copyright references."

---

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| P0 | 10 | Lock-free port, consolidation, ABA, mapping table, root lock, node locks, atomic counter, DashMap, 32-way, deferred renumbering |
| P1 | 14 | GPU kernel, transfer engine, scheduler, model runner, tensor view, async copy, checkpointing, vLLM fork, SGLang, iterative, rebalancing, tier as metadata, error hierarchy, signal-safe, concurrent tests |
| P2 | 7 | Rust docs, end-to-end benchmark, representative benchmarks, paper, tier callback, capacity, auto-consolidation |
| P3 | 4 | Test skip on Windows, coverage target, internal API warning, license check |
| Total | 35 | (One item per the 36 listed; item 6 covers both root lock and node locks as one issue) |

When you're ready to start, I'd suggest tackling P0-02 first (1-2 weeks, small, contained) before P0-01 (3 months, big). The `consolidate()` stub is a known gap, easy to fill, and the tests are waiting for it. Then P0-01 (the lock-free Rust port) which unlocks the rest of the P0s.

Want me to push these to GitHub as a `issues.md` file in the repo, or write a script to create them via `gh` CLI? Or would you rather I just save this markdown to a file in the workspace for you to paste?
