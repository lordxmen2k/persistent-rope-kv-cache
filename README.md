# Persistent Rope KV Cache

Research prototype exploring position-keyed persistent data structures for
KV cache management in LLM inference.

The hypothesis: a position-keyed rope (linked trie) cleanly unifies four
mechanisms current serving systems handle separately:

  1. O(1) eviction at any point via unlink
  2. Sharing across requests via shared sublist pointers
  3. Branching (beam search, parallel sampling) via structural sharing
  4. Tiered storage via node-type tags

## Status

Scaffolding. No code yet. See `tech_note.md` (forthcoming) for the design.

## Workspace layout

    KV_Cache/                 (this dir, repo root)
      .gitignore
      README.md
      KV_Cahce_PAT.txt        (gitignored, holds GitHub PAT during setup only)
