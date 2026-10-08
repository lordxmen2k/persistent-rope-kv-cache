# Lock-Free Concurrent Rope for Multi-Tenant KV Cache

**Author:** Gerald Enrique Nelson Mc Kenzie
**Date:** 2026-10-08
**License:** Apache 2.0
**Repository:** https://github.com/lordxmen2k/persistent-rope-kv-cache
**Status:** Draft v0.2 — design proposal, reframed toward lock-free Bw-tree-on-rope contribution

---

## Abstract

We propose a **lock-free concurrent rope** as the index structure for KV
cache in multi-tenant LLM inference serving. The design combines the
Bw-tree's latch-free mapping-table indirection (Microsoft, SIGMOD'13)
with a position-keyed persistent rope, yielding a tree index where
all mutations are atomic via compare-and-swap and reads traverse
without locks. **Honest framing:** the contribution is not a new data
structure — radix trees, persistent vectors, and lock-free B-trees
each exist. The contribution is the specific *combination* of Bw-tree
indirection with a rope-indexed KV cache for multi-tenant serving, and
the empirical demonstration that lock-free indexing is feasible on
GPU-attached memory. Prior work (vLLM, SGLang, Mooncake) uses
coarse-grained locking that becomes a contention bottleneck under high
concurrency; this is the gap we target.

## 1. Problem statement

KV cache for a single Llama-3-70B-class request occupies roughly
40 KB/token at bf16. At 128K context and batch 32, that is ≈160 GB
per server — comparable to the model weights. Under multi-tenant
serving (SaaS, agent platforms, code copilots), the cache pool is
shared across N concurrent requests and is the dominant memory object.

Two failure modes compound under multi-tenant load:

1. **Lock contention on the index.** vLLM uses a single global
   allocator with a mutex; SGLang's radix tree uses a global lock for
   mutations. At 100+ concurrent requests with frequent branching
   (beam search, agent rollouts, parallel sampling), this is the
   measured bottleneck.
2. **Read-modify-write races on shared prefixes.** Two requests sharing
   a prefix require either copy-on-write (current SGLang) or
   fine-grained locks; both are slow. The Bw-tree's delta-update model
   handles this with a single CAS per mutation.

The rope contribution targets (1)–(2) at the *index-structure* level.
The other KV-cache problems (memory footprint, decode bandwidth,
side channels) are orthogonal and addressed in other work.

## 2. Related work

### 2.1 KV cache index structures (production)

- **PagedAttention (vLLM, SOSP'23).** Flat block table indexed by
  block_id. O(1) ops, single global allocator with mutex. No tree
  structure, no structural sharing.
- **RadixAttention (SGLang, 2024).** Content-keyed radix tree. LRU
  eviction over unlocked leaves, refcount on shared subtrees, split on
  divergence. Production-deployed. Uses a global lock for mutations.
  Recent extensions: Unified Radix Cache for hybrid attention
  (FULL+SWA+MAMBA), HiCache for GPU/host/external tiers.
- **Mooncake Store (Moonshot AI, 2024).** Hash-keyed distributed KV
  pool. Chunked transfers, RDMA transfer engine. Not a tree.
- **SGLang-LSM (arXiv:2511.16138, Nov 2025).** Layered LSM-tree
  storage for KV cache, adaptive controller, key-value separation.

### 2.2 Tree-based KV cache (research)

- **Trie-based beam search (arXiv:2502.00085, Feb 2025).** Trie over
  beams, shared KV across branches with structural sharing. Closest
  prior work to the rope's branching semantics. Uses CPU-side tree
  state with KV-cache references.
- **ArborKV (arXiv:2605.22106, 2025).** Tree-of-thoughts structure-
  aware KV management. Tree topology drives budget allocation,
  ancestor-chain prioritized.
- **PEEK (arXiv:2607.02525, 2025).** Incremental radix tree over the
  pending request queue, cluster-aware admission and eviction.
- **RKSC (arXiv:2606.09937, 2025).** Reasoning-aware KV sharing with
  `repeat_interleave` zero-copy broadcast across B branches.
- **dotLLM (in development).** Roadmap explicitly plans trie-based
  prefix sharing with reference counting and COW for beam branching —
  essentially the persistent-rope design as a planned feature.

### 2.3 Lock-free tree structures (CS foundations)

- **Bw-tree (Microsoft, SIGMOD'13).** Latch-free B-tree via
  mapping-table indirection and delta updates. Production-deployed in
  Hekaton and Azure SQL. The reference design for what we propose.
- **Bε-tree (SIGMOD'15).** Bw-tree variant optimized for flash/SSD
  with write-optimized layout.
- **NV-tree (NSDI'15).** Persistent-memory-optimized Bw-tree variant.
- **Masstree (NSDI'12).** Concurrent trie with lock-free internals
  but different (per-node lock, not mapping-table indirection).

### 2.4 Persistent data structures (FP foundations)

- **Clojure persistent vector (Bagwell, 2001).** 32-way branching tree
  with structural sharing. The "rope" we mean.
- **Haskell Data.Sequence.** Finger tree, different access pattern.
- **Okasaki, "Purely Functional Data Structures" (1998).** The
  textbook.

### 2.5 Side-channel defenses (orthogonal)

- **PrefixWall (arXiv:2603.10726, 2025).** Defends hash-keyed prefix
  cache via principal-bound metadata.
- **HMACK (arXiv:2608.09225, 2025).** Cryptographic separation of
  cache keys by principal. The position-keyed analogue would use
  capability metadata on the rope's leaves.

## 3. Design: lock-free rope via Bw-tree-style mapping table

### 3.1 The rope

A position-keyed balanced binary tree of list segments. Each leaf
holds up to `B` tokens (default `B = 16`). Leaf count is `O(L / B)`;
depth is `O(log(L / B))`. Operations: `append`, `unlink`, `fork`,
`share_prefix`, `evict`. All `O(log L)`. The rope is well-described in
prior art (Bagwell, Clojure); we do not claim novelty on the data
structure itself.

### 3.2 The Bw-tree indirection

Following the Bw-tree, every rope node has:

- A **logical NodeID** (a 64-bit integer, never reused).
- A **mapping table** entry: `NodeID → PhysicalAddress` resolved via
  an atomic pointer (CAS-updatable).
- A **base node** at the physical address.
- A list of **delta records** (small updates: split, new child,
  refcount change, tier promotion) attached via an atomic pointer from
  the base node.

A read traverses: `NodeID → mapping table → base+delta → state`. A
mutation writes a new delta record and CAS-updates the delta list
pointer; this is the only synchronization point. Reads do not block
mutations and vice versa.

A periodic **consolidation** merges the delta chain into a new base
node and atomically updates the mapping table; old base + old deltas
are garbage-collected when no in-flight read references them (via
epoch-based reclamation or hazard pointers).

### 3.3 What this buys for KV cache

- **No global lock.** Every mutation is a single CAS on a per-node
  atomic. Reads traverse without locks. Throughput scales with cores
  in proportion to CAS contention, not in proportion to a single mutex
  holder.
- **Structural sharing is free.** `fork(seq)` allocates a new root
  NodeID and a new delta record pointing into the existing tree;
  unchanged subtrees are referenced by NodeID, not copied.
- **Reference counting for shared subtrees.** Each internal node
  carries an atomic refcount; `share_prefix(a, b, n)` increments
  refcount on the shared subtree via a single CAS.
- **No copy-on-write races.** Because all mutations are CAS-based, two
  concurrent forks diverge at the divergence point without race; the
  pre-divergence subtree is referenced by both, no copy needed.
- **Crash recovery via mapping-table snapshot.** The mapping table is
  small (NodeID → pointer). A periodic snapshot to disk is the WAL;
  recovery is "replay mapping table + reconstruct rope from K/V blocks
  on disk."

### 3.4 GPU integration

The rope lives on the CPU side. K/V tensors live in HBM as a flat
paged pool. At scheduling time, the rope is walked and a **snapshot
array** of `(physical_block_id, byte_offset, length)` pairs is built
for each active sequence. The GPU attention kernel consumes the
snapshot array and issues coalesced loads. This is the same pattern
vLLM and SGLang use today; the rope replaces the flat block table
with a tree-indexed one.

The snapshot is a CPU-side object. Building it is `O(L/B)` per
sequence; for a 128K sequence at B=16, that's 8K entries — small
enough to rebuild per scheduling cycle.

### 3.5 Tiered storage

Each leaf carries a `tier` tag (`HOT | WARM | COLD`). The leaf's K/V
tensor pointer carries the tier semantics. A demotion (HOT → WARM)
copies the K/V to DRAM, updates the leaf's pointer via CAS, releases
the HBM block. The rope traversal is tier-agnostic; the snapshot
array simply resolves to the right tier's address.

### 3.6 Capability-tagged variant (future work)

For multi-tenant safety, each leaf carries a `principal_id` and
`permissions` bitmask. Cross-principal `share_prefix` requires a
capability handoff and is checked at CAS time. This is the
position-keyed analogue of PrefixWall/HMACK. We do not include this
in the v1 design; it composes orthogonally.

## 4. Implementation plan

### 4.1 Phase 1: correctness reference (Python, single-threaded)

- Implement the rope in pure Python with Bw-tree-style mapping table.
- Single-threaded for correctness; threading hooks stubbed.
- Test against (a) sequential access, (b) branching, (c) prefix
  sharing, (d) eviction, (e) tier transitions.
- This is the deliverable that proves the design works.

### 4.2 Phase 2: concurrent correctness (Rust, multi-threaded)

- Port to Rust with `crossbeam-epoch` for epoch-based reclamation.
- Atomic refcounts on internal nodes.
- Test with TSan/loom for race conditions.
- Benchmark concurrent throughput vs. SGLang's locked radix tree on
  the same workload.

### 4.3 Phase 3: GPU integration (vLLM fork)

- Add the rope as a drop-in replacement for vLLM's block table.
- Wire the snapshot mechanism into the scheduler.
- Run NIAH, LongBench, multi-tenant throughput benchmarks.
- Compare against stock vLLM and SGLang.

### 4.4 Phase 4: tiered storage + capabilities

- Add WARM/COLD tiers with RDMA transfer engine integration.
- Add capability-tagged variants.
- Side-channel benchmark vs. PrefixWall.

## 5. Evaluation plan

Concrete benchmarks, ordered:

1. **Concurrent throughput.** 256 simultaneous requests with
   overlapping prefixes. Measure tokens/sec, lock-acquisition
   latency, CAS retry rate for the rope vs. SGLang's locked radix tree
   and vLLM's flat block table.
2. **Beam search memory and time.** 4-beam × 8K context. Measure
   peak memory (rope with structural sharing) and time-per-token
   (rope with concurrent fork).
3. **Crash recovery.** Kill the process at random points. Recover
   from the last mapping-table snapshot + disk-resident K/V blocks.
   Verify the recovered state matches the pre-crash state modulo
   in-flight requests.
4. **Side-channel resistance** (Phase 4 only). With capability
   metadata, verify that cross-tenant cache accesses are detected and
   rejected at CAS time.

Targets:

- ≥2× concurrent throughput vs. SGLang at 256 requests.
- ≥60% memory reduction on diverged beams vs. vLLM's CoW.
- Crash recovery in <5s for a 100 GB cache pool.

## 6. Limitations and honest framing

- **The rope is not novel.** Persistent vectors are 25 years old;
  radix trees for KV cache are 2 years old; trie-based beam search is
  9 months old. The contribution is the Bw-tree indirection, not the
  data structure.
- **Lock-free is hard.** Bw-tree correctness depends on careful epoch
  reclamation, hazard pointers, and atomic ordering. Bugs are
  catastrophic (use-after-free, lost updates). Implementation cost
  is real.
- **The benefit shows up only at high concurrency.** Below 32
  concurrent requests, a locked tree is fine. The paper's strongest
  result needs the multi-tenant workload.
- **Mapping table size.** For 10K concurrent requests, the mapping
  table is bounded but non-trivial. Compaction is a real engineering
  problem.
- **The position-keyed vs. content-keyed distinction is minor.** Both
  indexes support the same operations; the differences show up at
  specific access patterns, not in asymptotic complexity.

## 7. Open questions specific to the lock-free design

1. **Delta record format for rope nodes.** Bw-tree deltas are
   per-B-tree-page. Rope deltas are per-rope-node; the format needs
   design (split-delta, new-child-delta, refcount-delta, tier-delta).
2. **Mapping table compaction.** When can NodeIDs be reclaimed?
   Epoch-based reclamation requires waiting for all in-flight reads;
   the working set of in-flight reads can be large.
3. **GC strategy.** Old base nodes and obsolete deltas need
   reclamation. Epoch-based is simpler; hazard pointers are faster
   but harder to get right.
4. **Snapshot cost.** For a 128K sequence, the snapshot is 8K
   entries. Can the GPU consume the rope directly via a device-side
   mapping table mirror, or must it always be CPU-resolved?
5. **WAL design.** Mapping-table snapshot + K/V blocks on disk
   gives crash recovery, but the snapshot frequency is a tunable.
6. **Multi-thread vs. multi-process.** Bw-tree assumes shared
   memory. KV cache serving often uses multiple processes for
   isolation; this needs a different indirection layer.

## 8. References

1. Kwon et al. "Efficient Memory Management for Large Language Model
   Serving with PagedAttention." SOSP 2023.
2. Zheng et al. "SGLang: Efficient Execution of Structured Language
   Model Programs." arXiv:2312.07104, 2024.
3. Qin et al. "Mooncake: A KVCache-centric Disaggregated
   Architecture." 2024.
4. Ji et al. "Efficient Beam Search for Large Language Models via
   Trie-based Parallel Decoding." arXiv:2502.00085, Feb 2025.
5. Wu et al. "ArborKV: Structure-Aware KV Cache Management for
   Scaling Tree-based LLM Reasoning." 2025.
6. Jin et al. "PEEK: Predictive Queue-Informed KV Cache Management
   for LLM Serving." arXiv:2607.02525, 2025.
7. Liu et al. "RKSC: Reasoning-Aware KV Cache Sharing and Confident
   Early Exit." arXiv:2606.09937, 2025.
8. Levandoski et al. "The Bw-Tree: A Latch-Free B-Tree for
   Multicore Systems." Microsoft, SIGMOD 2013.
9. Sears & Ramamohanarao. "Bε-Tree." SIGMOD 2015.
10. Yang et al. "NV-Tree: Persistent Memory Bw-Tree." NSDI 2015.
11. Mao et al. "CachePrune: Privacy-Aware and Fine-Grained KV Cache
    Sharing." 2025.
12. Gao et al. "SGLang-LSM: Layered Storage for KV Cache."
    arXiv:2511.16138, 2025.
13. Cheng et al. "PersistentKV: Page-Aware Decode Scheduling."
    2025.
14. Qiu et al. "PROMPTPEEK: Prompt Leakage via KV-Cache Sharing in
    Multi-Tenant LLM Serving." NDSS 2025.
15. Song et al. "PrefixWall: Defending Multi-Tenant LLM Serving
    against Prefix Cache Side Channels." arXiv:2603.10726, 2025.
16. Bagwell. "Ideal Hash Trees." 2001.
17. Okasaki. "Purely Functional Data Structures." Cambridge, 1998.
18. Patel et al. "Splitwise: Efficient Generative LLM Inference
    Using Phase Splitting." ISCA 2024.
19. Hooper et al. "KVQuant: Towards 10 Million Context Length LLM
    Inference with KV Cache Quantization." NeurIPS 2024.
20. Liu et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
    KV Cache." ICML 2024.
21. Yang et al. "PyramidKV: Dynamic KV Cache Compression based on
    Pyramidal Information Funneling." 2024.
22. Wu et al. "Memorizing Transformers." ICLR 2022.
23. Mohtashami & Jaggi. "Landmark Attention: Random-Access Infinite
    Context Length for Transformers." NeurIPS 2023.
24. Woodruff et al. "CHERI: A Hybrid Capability-System Architecture
    for Scalable Software Compartmentalization." ISCA 2014.
25. TopKV: Topology-Aware Data Movement for Disaggregated GPU
    Inference. arXiv:2607.28633, 2025.
26. dotLLM project roadmap. github.com/kkokosa/dotLLM/docs/ROADMAP.md.

---

## License

Copyright 2026 Gerald Enrique Nelson Mc Kenzie

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
