# Persistent Rope KV Cache

Research prototype exploring position-keyed persistent data structures for
KV cache management in LLM inference.

The hypothesis: a position-keyed rope (linked trie) cleanly unifies four
mechanisms current serving systems handle separately:

  1. O(log n) eviction at any point via unlink
  2. Sharing across requests via shared sublist pointers
  3. Branching (beam search, parallel sampling) via structural sharing
  4. Tiered storage via node-type tags

## Status

Scaffolding. The design proposal is in [`tech_note.md`](./tech_note.md);
no code yet. The proposal is the source of truth until implementation
starts.

## Workspace layout

    KV_Cache/                                    (repo root)
      .gitignore                                 secrets + python + OS junk
      LICENSE                                    Apache 2.0
      README.md                                  this file
      tech_note.md                               design proposal (draft v0.1)
      KV_Cahce_PAT.txt                           gitignored, used during setup only
                                                (will be deleted once SSH is set up)

## Open research questions (from tech_note §7)

- Optimal block size for the rope leaf (default 16, may want 32–64)
- Lock-free concurrent rope (Bw-tree style) for multi-tenant serving
- Differentiable indexing (RMI over rope) for O(1) lookups
- Mamba / SSM integration as a heterogeneous leaf type
- Cross-machine rope: distributed or per-node with hash-keyed cross-node

## License

Apache 2.0. See [`LICENSE`](./LICENSE).
