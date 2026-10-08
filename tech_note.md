# Persistent-Rope KV Cache: Position-Keyed Structural Sharing for LLM Inference

**Author:** Gerald Enrique Nelson Mc Kenzie
**Date:** 2026-10-08
**License:** Apache 2.0
**Repository:** https://github.com/lordxmen2k/persistent-rope-kv-cache
**Status:** Draft v0.1 — design proposal, no implementation yet

---

## Abstract

We propose a position-keyed persistent rope (linked trie) as the index
structure for KV cache in LLM inference serving. The rope unifies four
mechanisms that current production systems handle separately: (1) O(1)
eviction at any point via unlink, (2) cross-request sharing via shared
sublist pointers, (3) beam search and parallel sampling branching via
persistent structural sharing, and (4) tiered storage via node-type tags.
The design is principled, the data structure is well-understood, and the
implementation cost is bounded. Honest framing: the rope does not solve
the quadratic cost of attention or the bandwidth bottleneck of decode;
what it offers is a *cleaner abstraction* over what PagedAttention,
SGLang, and Mooncake already do, with the specific wins of structural
sharing for branches and free O(1) eviction semantics.

## 1. Problem statement

KV cache for a single request on Llama-3-70B-class models occupies

```
mem = batch × seq_len × n_layers × 2 × n_kv_heads × head_dim × bytes
    ≈ 40 KB/token  (at 4 KV heads, 80 layers, 128 head dim, bf16)
```

At 128K context and batch 32, this is ≈160 GB — comparable to the model
weights themselves. Five concrete failure modes follow.

1. **Linear memory growth with context.** Long-context applications
   (RAG over a code repo, million-token chat) blow past HBM capacity.
2. **Memory fragmentation under mixed workloads.** Different requests
   have different lengths; naive contiguous allocation wastes 60–80%
   of the reserved memory (vLLM SOSP'23).
3. **Memory-bandwidth bound at decode.** Each token reads the entire KV
   cache from HBM. Decode is bandwidth-bound, not compute-bound, even
   on H100.
4. **Cross-request reuse is unsafe.** Sharing KV across tenants
   introduces timing side channels (PROMPTPEEK, NDSS'25; EarlyBird;
   InputSnatch) that can reconstruct another user's prompt from
   observable TTFT.
5. **Branching amplifies everything.** Beam search and parallel sampling
   multiply the cache by the branching factor; copy-on-write on
   divergence is the current mitigation and it is wasteful.

The rope targets (1)–(3) and (5) at the *index-structure* level. Side
channels (4) are an orthogonal problem that requires capability-tagged
pointers (see §6).

## 2. Related work (honest framing)

| System | Index structure | Branch sharing | Eviction | Sharing across requests | Production |
|---|---|---|---|---|---|
| HuggingFace `DynamicCache` | flat tensor | none | none | none | yes |
| vLLM PagedAttention | block table (flat array) | copy-on-write | free list | APC (unsafe) | yes |
| SGLang | radix tree (trie on tokens) | copy-on-write | LRU/LFU | LPM scheduler | yes |
| Mooncake Store | hash-keyed distributed store | copy-on-write | LRU | prefix-cache pool | yes |
| H2O / SnapKV / Pyramid | n/a (eviction only) | n/a | importance | n/a | research |
| KIVI / KVQuant | n/a (quantization only) | n/a | mixed-precision | n/a | research |
| Memorizing Transformer | ANN over external KV | n/a | FIFO | retrieval-based | research |

**Where the rope fits.** It is closest to SGLang's radix tree in spirit
(both index structures, both target prefix reuse), but differs in two
key ways:

- **Position-keyed, not content-keyed.** SGLang's radix tree indexes
  blocks by token-content prefix hash. The rope indexes by sequence
  position. This makes it better for ordered sequential access (the
  common case during decode) and for branching (because forks happen at
  a position, not at a content boundary).
- **Persistent semantics.** A fork creates a new root pointer that
  shares unchanged subtrees. Old beams remain valid without
  copy-on-write. This is Clojure-style persistent vectors, applied to
  KV cache.

**What the rope does not do.**

- It does not compress KV. Quantization (KIVI/KVQuant), pruning (H2O),
  or eviction (SnapKV) are orthogonal and compose with the rope.
- It does not eliminate the quadratic attention cost. FlashAttention
  is still required.
- It does not address the bandwidth bottleneck of decode. Mixed-tier
  storage (HBM/DRAM/SSD) helps; the rope is the index over those tiers.

## 3. Design: the persistent rope

### 3.1 Structure

A balanced binary tree of list segments, where each leaf holds up to
`B` tokens (default `B = 16`, same as PagedAttention). The leaf count
is `O(L / B)`. The depth is `O(log(L / B))`.

```
                        [root: 0..L)
                       /          \
                [left: 0..L/2)   [right: L/2..L)
                  /     \           /      \
              [a] [b] [c] [d]   [e] [f] [g] [h]   <- leaves
               │   │   │   │     │   │   │   │
              K/V K/V K/V K/V   K/V K/V K/V K/V
```

Each leaf node contains:

```
struct LeafNode {
    LeafNode* left_sibling;   // doubly-linked list at the leaf level
    LeafNode* right_sibling;
    RopeNode* parent;         // up to the root
    uint32_t  start;          // absolute sequence position
    uint32_t  end;            // exclusive
    KVBlock   kv;             // K/V tensors for [start, end)
    Tier      tier;           // HOT | WARM | COLD
    uint32_t  refcount;       // for safe shared-subtree lifetime
};
```

The internal `RopeNode` carries only `start`, `end`, `left`, `right`,
`refcount`, and `tier`. K/V lives only at the leaves.

### 3.2 Operations

- **`append(seq, kv_block)`** — extend the rightmost leaf; if full,
  split it. O(log L).
- **`unlink(seq, n)`** — drop the leftmost n tokens; if a leaf becomes
  empty, refcount-decrement and free. O(log L) amortized.
- **`fork(seq)`** — return a new root pointer; share all unchanged
  subtrees; copy only the rightmost divergent leaf. O(log L).
- **`share_prefix(seq_a, seq_b, n)`** — point `seq_b`'s prefix subtree
  to `seq_a`'s; refcount++. O(log L).
- **`evict(seq, n)`** — like unlink, but the evicted nodes are spilled
  to a colder tier (WARM or COLD) rather than freed. The leaf's `tier`
  tag is updated.

All operations are O(log L), with `L` being the sequence length. This
is strictly worse than PagedAttention's O(1) block-table operations,
but the gap is small in practice (a 128K sequence has ≈13 levels).

### 3.3 Why a rope and not a doubly-linked list

A doubly-linked list of pages is simpler and gives O(1) tail append /
head unlink. But:

- **Random access is O(n).** The rope gives O(log n) at the cost of one
  extra level of indirection. For sparse attention patterns this
  matters.
- **Branching requires O(n) copy.** A rope fork shares the entire
  unchanged prefix in O(log n).
- **Tiered storage needs to address arbitrary positions.** A list
  walk is fine for sequential access; a rope is fine for both.

For workloads that are purely sequential (e.g. single-stream decode),
the linked list is strictly better. The rope pays its complexity cost
only when branching or sparse access shows up.

### 3.4 Tiered storage via node tags

Each leaf carries a `tier` tag: `HOT` (HBM), `WARM` (CPU DRAM), `COLD`
(NVMe SSD). On read, the rope traversal hits the right tier
transparently. On eviction from `HOT` to `WARM`, the leaf's K/V is
copied to DRAM and the HBM block is released; the leaf's `tier` tag
flips. A periodic daemon promotes `WARM` → `HOT` for hot leaves and
demotes `HOT` → `WARM` based on access frequency (LRU-2).

This is the design Mooncake ships, but with a rope-indexed tier
boundary rather than a hash-keyed one. The advantage: position-based
eviction is sequential and predictable, and prefetching is straightforward
(read the next N leaves in order).

### 3.5 Side-channel defense

The rope *enables* a capability-tagged variant. Each leaf carries a
`principal_id` and `permissions` field. A cross-tenant access requires
a capability minted by the principal, which is not derivable from the
rope pointer alone. This is the CHERI (Univ. Cambridge ISCA'14+)
insight applied to KV cache.

Implementation: at the rope level, `share_prefix` requires explicit
capability handoff; refcount is incremented only after capability
check. The closest existing work is PrefixWall (arXiv:2603.10726) and
HMACK (arXiv:2608.09225), both of which defend the *hash-keyed*
prefix cache. The capability-tagged rope is the position-keyed
analogue and has not been published to our knowledge.

## 4. Implementation sketch

A reference implementation in pure Python (no GPU) targets correctness
and benchmarking. A production implementation in Rust or C++ with
CUDA kernels targets performance.

### 4.1 Python reference (correctness target)

```python
class RopeNode:
    __slots__ = ("start", "end", "left", "right", "refcount", "tier", "parent")

class LeafNode(RopeNode):
    __slots__ = ("left_sibling", "right_sibling", "kv", "principal_id", "perms")

class PersistentRope:
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root: RopeNode | None = None

    def append(self, kv: torch.Tensor, tier: Tier = Tier.HOT) -> RopeNode:
        ...

    def fork(self) -> "PersistentRope":
        # structural sharing: new root, same left subtree
        ...

    def unlink(self, n: int) -> None:
        ...
```

### 4.2 Production implementation (perf target)

The hot path is GPU attention. The rope lives on the CPU side as an
index; the actual K/V tensors live in HBM. On attention, the GPU
kernel receives a flat list of `(block_id, byte_offset)` pairs resolved
from the rope at scheduling time, then issues coalesced loads. This is
the same pattern vLLM uses today; the rope replaces vLLM's flat block
table with a tree-structured one.

Locking: per-leaf mutex for mutation, RCU-style for read-mostly
access. Lock-free variants (Bw-tree, Bε-tree) are research-grade but
not necessary for v1.

## 5. Evaluation plan

Concrete benchmarks, in order of priority:

1. **Long-context NIAH** at 4K, 16K, 64K, 256K context. Measure
   retrieval accuracy vs. PagedAttention baseline.
2. **LongBench** (16 subsets) at 4K, 16K, 32K context. Measure average
   score vs. PagedAttention, SGLang, and the StreamingLLM / H2O /
   SnapKV family of eviction baselines.
3. **Beam search memory** — 4-beam × 8K context. Measure peak memory
   of the rope (with structural sharing) vs. vLLM (with copy-on-write).
   Hypothesis: 60–80% memory reduction on diverged beams.
4. **Prefix-sharing throughput** — synthetic workload with 90% prefix
   overlap. Measure tokens/sec and TTFT vs. vLLM APC.
5. **Tiered storage** — HBM/DRAM/SSD tier mix. Measure effective
   bandwidth vs. flat HBM and flat DRAM.

Implementation: start with the Python reference and validate against
(1) and (2) on Llama-3-8B. If results hold, port to a vLLM fork and
run (3) and (4) on a single H100.

## 6. Limitations and honest framing

- **GPU perf cost.** The rope adds a tree walk on every cache lookup.
  PagedAttention is one level of indirection with a fixed stride; the
  rope is `O(log L)` with variable stride. For very long contexts
  (≥256K) this matters; for typical 8K–32K contexts it does not.
- **Concurrency is harder than flat tables.** Tree rebalancing under
  multi-tenant load is non-trivial. vLLM's flat block table has a
  simple global allocator; the rope needs per-subtree locking.
- **The rope is not a replacement for eviction or quantization.** It
  composes with them, but it does not by itself reduce memory
  pressure. H2O / SnapKV / KIVI still need to run on top.
- **Side-channel defense is not free.** Capability-tagged pointers add
  per-leaf metadata and require runtime checks. The throughput cost is
  small but non-zero.
- **Empirical honesty.** Until we run the benchmarks, all claims about
  memory reduction and throughput are hypotheses, not results. The
  repo will track measurements as they come in.

## 7. Open questions

1. **Optimal block size `B`.** vLLM defaults to 16. Real workload
   sweeps suggest 32–64 for long-context; we should re-sweep for the
   rope.
2. **Lock-free concurrent rope.** Bw-tree (Microsoft, SIGMOD'13) is the
   closest existing design; porting it to GPU-attached KV cache is
   open.
3. **Differentiable indexing.** The rope is currently content-blind.
   Learned indexing (e.g., RMI — Recursive Model Indexes) over the
   rope could give O(1) lookups with learned models replacing
   internal nodes.
4. **Mamba / SSM integration.** If part of the cache is replaced by
   Mamba state, the rope's leaf type becomes heterogeneous. Clean
   extension, but the interaction with structural sharing is
   non-trivial.
5. **Cross-machine rope.** Mooncake shards the KV pool across nodes.
   Does the rope index become a distributed structure, or is the rope
   per-node and the cross-node lookup is hash-keyed? Open.

## 8. References

1. Kwon et al. "Efficient Memory Management for Large Language Model
   Serving with PagedAttention." SOSP 2023.
2. Zheng et al. "SGLang: Efficient Execution of Structured Language
   Model Programs." 2024.
3. Qin et al. "Mooncake: A KVCache-centric Disaggregated Architecture
   for LLM Serving." 2024.
4. Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative
   Inference of Large Language Models." NeurIPS 2023.
5. Li et al. "SnapKV: LLM Knows What You are Looking for Before
   Generation." ICML 2024.
6. Yang et al. "PyramidKV: Dynamic KV Cache Compression based on
   Pyramidal Information Funneling." 2024.
7. Liu et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
   KV Cache." ICML 2024.
8. Xiao et al. "Efficient Streaming Language Models with Attention
   Sinks." ICLR 2024.
9. Hooper et al. "KVQuant: Towards 10 Million Context Length LLM
   Inference with KV Cache Quantization." NeurIPS 2024.
10. Wu et al. "Memorizing Transformers." ICLR 2022.
11. Mohtashami & Jaggi. "Landmark Attention: Random-Access Infinite
    Context Length for Transformers." NeurIPS 2023.
12. Wu et al. "PROMPTPEEK: Prompt Leakage via KV-Cache Sharing in
    Multi-Tenant LLM Serving." NDSS 2025.
13. Song et al. "The Early Bird Catches the Leak: Unveiling Timing
    Side Channels in LLM Serving Systems." 2024.
14. Woodruff et al. "CHERI: A Hybrid Capability-System Architecture
    for Scalable Software Compartmentalization." ISCA 2014.
15. Cheng et al. "Adaptive CHERI Compartmentalization for
    Heterogeneous Systems." ISCA 2025.
16. Kwon et al. "Disaggregated LLM Serving via Phase-Aware
    Scheduling." DistServe, 2024.
17. Patel et al. "Splitwise: Efficient Generative LLM Inference Using
    Phase Splitting." ISCA 2024.
18. Su et al. "RotateKV: Accurate and Robust 2-Bit KV Cache
    Quantization." IJCAI 2025.
19. He et al. "ZipCache: Accurate and Efficient KV Cache
    Quantization." NeurIPS 2024.
20. TopKV: Topology-Aware Data Movement for Disaggregated GPU
    Inference. arXiv:2607.28633, 2025.
21. Persistent rope data structure. Bagwell & Stucki, Clojure
    implementation. 2008.
22. Jamba team. "Jamba: A Hybrid Transformer-Mamba Language Model."
    2024.

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
