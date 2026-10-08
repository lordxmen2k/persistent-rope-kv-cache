# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""vLLM-compatible block manager interface.

This module exposes a RopeBlockManager class that mimics vLLM's BlockManager
API surface, allowing the rope-based KV cache to be plugged into vLLM
when the user installs it. The interface is intentionally narrow: only
the methods vLLM actually calls are implemented.

Reference: vLLM BlockManager API (vllm/core/block_manager.py)

This module does NOT depend on vLLM; it's a standalone adapter.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .core import PersistentRope
from .types import NodeID, NULL_NODE, Tier


@dataclass
class RopeBlock:
    """A logical block of K/V cache.

    A RopeBlock wraps a leaf in the persistent rope and carries the
    metadata vLLM expects (block_id, refcount, tier, num_tokens).
    """
    block_id: int            # unique within a RopeBlockManager
    rope_node: NodeID        # back-reference to the rope leaf
    ref_count: int = 1
    tier: Tier = Tier.HOT
    num_tokens: int = 0      # how many tokens are stored in this block
    # In a real integration, `kv_data` would be a tensor of shape
    # (num_tokens, n_layers, n_kv_heads, head_dim). Here it's bytes.
    kv_data: bytes = b""

    def can_append(self, n: int = 1) -> bool:
        return self.num_tokens + n <= 16  # block_size hardcoded here


class RopeBlockManager:
    """A vLLM-compatible block manager backed by a PersistentRope.

    vLLM's BlockManager has a much larger API; this class exposes only
    the methods that are actually called by the scheduler and the model
    runner. To integrate:

    1. Subclass or monkey-patch vLLM's BlockManager.
    2. Replace its allocator with a `RopeBlockManager` instance.
    3. Route `allocate`, `free`, `fork`, `swap_out`, `swap_in` to this class.

    The mapping between vLLM concepts and ours:
      vLLM seq.seq_id          -> our seq_handle
      vLLM block_table         -> our RopeBlock list (resolved from rope snapshot)
      vLLM allocate/free       -> our append_to_root / unlink
      vLLM fork (CoW on divergence) -> our share_prefix + append_to
    """

    def __init__(self, block_size: int = 16, num_layers: int = 0,
                 num_kv_heads: int = 0, head_dim: int = 0):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope = PersistentRope(block_size=block_size)
        # seq_id -> root NodeID
        self._seqs: dict[int, NodeID] = {}
        # block_id counter
        self._next_block_id = 1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    def create_sequence(self, seq_id: int) -> int:
        """Register a new sequence. Returns the seq_id."""
        with self._lock:
            if seq_id in self._seqs:
                raise ValueError(f"sequence {seq_id} already exists")
            # Empty rope for this sequence; the rope's main root is the
            # "current" state, but we keep per-seq roots.
            self._seqs[seq_id] = NULL_NODE
            return seq_id

    def delete_sequence(self, seq_id: int) -> None:
        """Remove a sequence. The rope's main root is not modified;
        per-seq root is dropped (and refcounts in the shared rope
        will be decremented when the rope is GC'd)."""
        with self._lock:
            if seq_id not in self._seqs:
                raise KeyError(f"sequence {seq_id} not found")
            del self._seqs[seq_id]

    def get_seq_root(self, seq_id: int) -> Optional[NodeID]:
        return self._seqs.get(seq_id)

    # ------------------------------------------------------------------
    # Block-level operations
    # ------------------------------------------------------------------

    def allocate(self, seq_id: int, n_tokens: int, kv_data: bytes) -> list[RopeBlock]:
        """Allocate n_tokens worth of K/V for a sequence.

        Returns the new RopeBlocks. In a real integration, kv_data would
        be a tensor; here it's bytes.
        """
        with self._lock:
            if seq_id not in self._seqs:
                raise KeyError(f"sequence {seq_id} not found")
            # Use the per-sequence root
            current_root = self._seqs[seq_id]
            new_root = self.rope.append_to(current_root, kv_data, n=n_tokens)
            self._seqs[seq_id] = new_root
            # Build RopeBlock list from the snapshot
            blocks: list[RopeBlock] = []
            offset = 0
            for nid, off, length in self.rope.resolve(new_root):
                # Find a free block_id
                bid = self._next_block_id
                self._next_block_id += 1
                blocks.append(RopeBlock(
                    block_id=bid, rope_node=nid, ref_count=1,
                    tier=Tier.HOT, num_tokens=length, kv_data=kv_data[off:off+length],
                ))
            return blocks

    def free(self, seq_id: int, n_tokens: int) -> None:
        """Free the leftmost n_tokens of a sequence."""
        with self._lock:
            if seq_id not in self._seqs:
                raise KeyError(f"sequence {seq_id} not found")
            current_root = self._seqs[seq_id]
            new_root = self.rope.unlink_from(current_root, n_tokens)
            self._seqs[seq_id] = new_root

    def fork(self, parent_seq_id: int, child_seq_id: int) -> None:
        """Fork a sequence. The child shares the parent's K/V via
        structural sharing; further divergence is via CoW on append."""
        with self._lock:
            if parent_seq_id not in self._seqs:
                raise KeyError(f"parent sequence {parent_seq_id} not found")
            if child_seq_id in self._seqs:
                raise ValueError(f"child sequence {child_seq_id} already exists")
            parent_root = self._seqs[parent_seq_id]
            child_root = self.rope.fork(parent_root)
            self._seqs[child_seq_id] = child_root

    def can_append(self, seq_id: int, n_tokens: int) -> bool:
        """vLLM asks: can I append n_tokens to this sequence? Always yes
        (rope grows dynamically)."""
        return seq_id in self._seqs

    def append_slot(self, seq_id: int, kv_data: bytes) -> list[RopeBlock]:
        """Append a single token's K/V to a sequence (vLLM's per-step API)."""
        return self.allocate(seq_id, n_tokens=len(kv_data), kv_data=kv_data)

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def swap_out(self, seq_id: int, n_tokens: int) -> None:
        """Evict n_tokens to WARM tier (simulating HBM->DRAM swap)."""
        with self._lock:
            if seq_id not in self._seqs:
                raise KeyError(f"sequence {seq_id} not found")
            current_root = self._seqs[seq_id]
            new_root = self.rope.evict_from(current_root, n_tokens)
            self._seqs[seq_id] = new_root

    def swap_in(self, seq_id: int, n_tokens: int) -> None:
        """Promote n_tokens to HOT tier (DRAM->HBM swap)."""
        with self._lock:
            if seq_id not in self._seqs:
                raise KeyError(f"sequence {seq_id} not found")
            current_root = self._seqs[seq_id]
            new_root = self.rope.promote_from(current_root, n_tokens)
            self._seqs[seq_id] = new_root

    # ------------------------------------------------------------------
    # Snapshot (for GPU kernel)
    # ------------------------------------------------------------------

    def get_snapshot(self, seq_id: int) -> list[tuple[NodeID, int, int]]:
        """Return the snapshot array for a sequence (what a GPU kernel
        would consume for paged attention)."""
        with self._lock:
            if seq_id not in self._seqs:
                raise KeyError(f"sequence {seq_id} not found")
            return self.rope.resolve(self._seqs[seq_id])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "rope_stats": self.rope.stats(),
            "n_sequences": len(self._seqs),
        }


# Convenience: re-exported as a class vLLM users can plug in
vLLMCompatibleInterface = RopeBlockManager
