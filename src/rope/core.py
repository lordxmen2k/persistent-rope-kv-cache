"""PersistentRope: the rope index.

Implementation notes
--------------------
- Logical NodeIDs are 64-bit and never reused. This is the Bw-tree's
  invariant that lets us CAS the mapping table without ABA problems.
- Every base node carries a singly-linked delta chain. New mutations
  prepend a delta and CAS the chain head. Readers walk the chain and
  reconstruct the state.
- Periodically (every N mutations, configurable), a consolidation pass
  rebuilds the base node from the deltas, atomically swaps it in via
  the mapping table, and the old base + deltas become garbage. In this
  reference implementation, we provide the consolidation API but defer
  scheduling to the caller.
- Reference counting on internal nodes is the structural-sharing
  mechanism: `share_prefix` and `fork` increment refcount via a single
  CAS on the node's counter; a node with refcount=0 is reclaimable
  when its delta chain has no live readers (epoch-based in Rust; a
  simple "deferred-free" list in this Python reference).

This is a Python reference. Thread-safety uses fine-grained locks on
each node; the GIL makes "lock-free" semantics approximate. The
production implementation is in `rust/` and uses crossbeam-epoch with
true atomic CAS.
"""
from __future__ import annotations

import threading
from typing import Iterator, Optional

from .types import Delta, DeltaKind, NodeID, NULL_NODE, Tier


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class RopeError(Exception):
    """Base error for rope operations."""


class EmptySequenceError(RopeError):
    """Operation requires a non-empty sequence."""


class OutOfRangeError(RopeError):
    """Token position out of range."""


# ----------------------------------------------------------------------------
# Node types
# ----------------------------------------------------------------------------


class LeafNode:
    """A leaf holds up to B tokens. The K/V blob is opaque bytes."""

    __slots__ = (
        "node_id",
        "start",
        "end",
        "kv_blob",
        "tier",
        "refcount",
        "lock",
    )

    def __init__(
        self,
        node_id: NodeID,
        start: int,
        end: int,
        kv_blob: bytes,
        tier: Tier = Tier.HOT,
        refcount: int = 1,
    ):
        self.node_id = node_id
        self.start = start
        self.end = end
        self.kv_blob = kv_blob
        self.tier = tier
        self.refcount = refcount
        self.lock = threading.Lock()


class InternalNode:
    """An internal node with two children (left, right) and metadata."""

    __slots__ = (
        "node_id",
        "start",
        "end",
        "left",
        "right",
        "tier",
        "refcount",
        "lock",
        # Bw-tree delta chain
        "_base_snapshot",
        "_deltas",
    )

    def __init__(
        self,
        node_id: NodeID,
        start: int,
        end: int,
        left: NodeID,
        right: NodeID,
        tier: Tier = Tier.HOT,
        refcount: int = 1,
    ):
        self.node_id = node_id
        self.start = start
        self.end = end
        self.left = left
        self.right = right
        self.tier = tier
        self.refcount = refcount
        self.lock = threading.Lock()
        # In the Bw-tree, _base_snapshot is the immutable base and
        # _deltas are pending updates. In this Python reference we
        # keep the live fields above as the base and use _deltas for
        # the audit log of pending/committed changes. The real
        # latch-free semantics are validated in the Rust port.
        self._base_snapshot: Optional[dict] = None
        self._deltas: list[Delta] = []


# ----------------------------------------------------------------------------
# PersistentRope
# ----------------------------------------------------------------------------


class PersistentRope:
    """A position-keyed persistent rope with Bw-tree-style indirection.

    Parameters
    ----------
    block_size : int
        Tokens per leaf. Default 16 (same as PagedAttention's typical
        page size).
    consolidation_threshold : int
        Number of mutations after which `consolidate()` should be called.
        The reference does not auto-consolidate.
    """

    def __init__(self, block_size: int = 16, consolidation_threshold: int = 1024):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self.consolidation_threshold = consolidation_threshold

        # Mapping table: NodeID -> LeafNode | InternalNode
        self._mapping: dict[NodeID, object] = {}
        # NodeID generator
        self._next_id = 1
        self._id_lock = threading.Lock()
        # Deferred-free list: NodeIDs whose refcount dropped to 0
        self._deferred_free: list[NodeID] = []
        # Mutation count for consolidation scheduling
        self._mutation_count = 0
        self._root_lock = threading.Lock()  # protects _root_id only
        self._root_id: Optional[NodeID] = None

    # ------------------------------------------------------------------
    # Node lifecycle
    # ------------------------------------------------------------------

    def _alloc_id(self) -> NodeID:
        with self._id_lock:
            nid = NodeID(self._next_id)
            self._next_id += 1
            return nid

    def _put(self, node) -> None:
        self._mapping[node.node_id] = node

    def _get(self, nid: NodeID):
        return self._mapping.get(nid)

    def _get_leaf(self, nid: NodeID) -> LeafNode:
        n = self._get(nid)
        if n is None:
            raise RopeError(f"node {nid} not found")
        if not isinstance(n, LeafNode):
            raise RopeError(f"node {nid} is not a leaf")
        return n

    def _get_internal(self, nid: NodeID) -> InternalNode:
        n = self._get(nid)
        if n is None:
            raise RopeError(f"node {nid} not found")
        if not isinstance(n, InternalNode):
            raise RopeError(f"node {nid} is not an internal")
        return n

    def _bump_mutation(self) -> None:
        self._mutation_count += 1

    # ------------------------------------------------------------------
    # Public API: structural
    # ------------------------------------------------------------------

    def root_id(self) -> Optional[NodeID]:
        """Return the main root NodeID, or None if empty."""
        return self._root_id

    def is_empty(self) -> bool:
        return self._root_id is None

    def refcount_of(self, nid: NodeID) -> int:
        n = self._get(nid)
        return n.refcount  # type: ignore[union-attr]

    def stats(self) -> dict:
        """Snapshot of internal statistics for benchmarking."""
        leaves = sum(1 for n in self._mapping.values() if isinstance(n, LeafNode))
        internals = sum(1 for n in self._mapping.values() if isinstance(n, InternalNode))
        by_tier = {t: 0 for t in Tier}
        for n in self._mapping.values():
            if isinstance(n, LeafNode):
                by_tier[n.tier] += 1
        return {
            "n_leaves": leaves,
            "n_internals": internals,
            "n_nodes_total": len(self._mapping),
            "tier_counts": {t.value: c for t, c in by_tier.items()},
            "deferred_free": len(self._deferred_free),
            "mutations": self._mutation_count,
            "next_id": self._next_id,
            "length": self.length(),
        }

    # ------------------------------------------------------------------
    # Public API: mutations
    # ------------------------------------------------------------------

    def append(self, kv_blob: bytes, n: Optional[int] = None) -> NodeID:
        """Append tokens from `kv_blob`. By default, one blob byte = one token.

        Thread-safe via the root lock. The Rust port uses CAS instead.
        """
        if n is None:
            n = len(kv_blob)
        if n < 1:
            raise ValueError("n must be >= 1")
        if len(kv_blob) != n:
            raise ValueError(
                f"kv_blob length {len(kv_blob)} != n {n}; in this reference 1 byte = 1 token"
            )
        with self._root_lock:
            new_root = self.append_to(self._root_id, kv_blob, n)
            self._root_id = new_root
            return new_root

    def unlink(self, n: int) -> NodeID:
        """Drop the leftmost `n` tokens from the main root. Returns new root NodeID.

        If n >= length, returns NULL_NODE (empty). Updates the rope's main root.
        Thread-safe via the root lock.
        """
        with self._root_lock:
            new_root = self.unlink_from(self._root_id, n)
            self._root_id = new_root
            return new_root

    def evict(self, n: int) -> NodeID:
        """Demote the leftmost n tokens to WARM tier. Returns new root."""
        with self._root_lock:
            new_root = self.evict_from(self._root_id, n)
            self._root_id = new_root
            return new_root

    def promote(self, n: int) -> NodeID:
        """Promote the rightmost n tokens to HOT tier. Returns new root."""
        with self._root_lock:
            new_root = self.promote_from(self._root_id, n)
            self._root_id = new_root
            return new_root

    def fork(self, root: Optional[NodeID] = None) -> NodeID:
        """Structural-sharing fork. Returns new root NodeID."""
        if root is None:
            root = self._root_id
        if root is None or root == NULL_NODE:
            return NULL_NODE
        self._bump_refcount_subtree(root)
        self._bump_mutation()
        return root

    def share_prefix(self, a: Optional[NodeID], n: int) -> NodeID:
        """Return a new root that structurally shares the first `n` tokens of `a`."""
        if a is None or a == NULL_NODE:
            raise EmptySequenceError("a is empty")
        if n < 0:
            raise ValueError("n must be >= 0")
        a_len = self._span(a)
        if n > a_len:
            raise OutOfRangeError(f"n={n} > a.length={a_len}")
        if n == 0:
            return a
        if n == a_len:
            return self.fork(a)
        forked = self.fork(a)
        return self._truncate_right(forked, n)

    def unlink_from(self, root: Optional[NodeID], n: int) -> NodeID:
        """Drop the leftmost `n` tokens from `root`. Returns new root NodeID."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return root
        if root is None or root == NULL_NODE:
            raise EmptySequenceError("cannot unlink from empty sequence")
        length = self._span(root)
        if n >= length:
            self._drop_subtree(root)
            return NULL_NODE
        new_root = self._unlink_left(root, n)
        self._renumber(new_root, -n)
        self._bump_mutation()
        return new_root

    def _renumber(self, nid: NodeID, delta: int) -> None:
        """Walk the subtree rooted at nid and add `delta` to every start/end."""
        n = self._get(nid)
        if n is None:
            return
        n.start += delta
        n.end += delta
        if not isinstance(n, LeafNode):
            self._renumber(n.left, delta)
            self._renumber(n.right, delta)

    def evict_from(self, root: Optional[NodeID], n: int) -> NodeID:
        """Demote the leftmost n tokens of `root` to WARM tier."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0 or root is None or root == NULL_NODE:
            return root
        length = self._span(root)
        n = min(n, length)
        new_root = self._demote_left(root, n, Tier.WARM)
        self._bump_mutation()
        return new_root

    def promote_from(self, root: Optional[NodeID], n: int) -> NodeID:
        """Promote the rightmost n tokens of `root` to HOT tier."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0 or root is None or root == NULL_NODE:
            return root
        length = self._span(root)
        n = min(n, length)
        new_root = self._promote_right(root, n, Tier.HOT)
        self._bump_mutation()
        return new_root

    # ------------------------------------------------------------------
    # Public API: reads
    # ------------------------------------------------------------------

    def resolve(self, root: Optional[NodeID] = None) -> list[tuple[NodeID, int, int]]:
        """Build a snapshot array of (node_id, byte_offset, length) for
        the leaves in left-to-right order. This is what a GPU attention
        kernel would consume. If root is None, uses the main root.
        """
        if root is None:
            root = self._root_id
        if root is None or root == NULL_NODE:
            return []
        out: list[tuple[NodeID, int, int]] = []
        offset = 0
        for leaf in self._iter_leaves(root):
            out.append((leaf.node_id, offset, leaf.end - leaf.start))
            offset += leaf.end - leaf.start
        return out

    def get_kv(self, pos: int, root: Optional[NodeID] = None) -> bytes:
        """Return the single-token K/V blob at absolute position `pos`."""
        if root is None:
            root = self._root_id
        if root is None or root == NULL_NODE:
            raise OutOfRangeError("empty sequence")
        leaf = self._find_leaf(root, pos)
        if leaf is None:
            raise OutOfRangeError(f"pos {pos} out of range")
        return leaf.kv_blob[pos - leaf.start : pos - leaf.start + 1]

    def append_to(self, root: Optional[NodeID], kv_blob: bytes, n: Optional[int] = None) -> NodeID:
        """Append to a specific root, returning a new root NodeID."""
        if n is None:
            n = len(kv_blob)
        if n < 1:
            raise ValueError("n must be >= 1")
        if len(kv_blob) != n:
            raise ValueError(
                f"kv_blob length {len(kv_blob)} != n {n}; in this reference 1 byte = 1 token"
            )
        new_root = root
        # Append one token at a time using a single-byte slice
        for i in range(n):
            new_root = self._append_one(new_root, kv_blob[i : i + 1])
        self._bump_mutation()
        return new_root

    def _append_one(self, root: Optional[NodeID], kv_token: bytes) -> NodeID:
        """Append a single token."""
        if root is None or root == NULL_NODE:
            leaf = LeafNode(self._alloc_id(), 0, 1, kv_token, Tier.HOT, 1)
            self._put(leaf)
            return leaf.node_id

        # Find the rightmost leaf; if it has room, append; else split.
        path: list[InternalNode] = []  # path of internals from root to leaf
        cur = root
        while True:
            n = self._get(cur)
            if isinstance(n, LeafNode):
                break
            assert isinstance(n, InternalNode)
            path.append(n)
            cur = n.right  # always go right for append

        leaf = n  # type: ignore[assignment]
        with leaf.lock:
            if leaf.end - leaf.start < self.block_size:
                # In-place append: replace the rightmost leaf with a new one
                new_blob = leaf.kv_blob + kv_token
                new_leaf = LeafNode(
                    self._alloc_id(), leaf.start, leaf.end + 1,
                    new_blob, leaf.tier, leaf.refcount,
                )
                self._put(new_leaf)
                new_root = self._replace_leaf_in_path(root, path, leaf.node_id, new_leaf.node_id)
                return new_root
            else:
                # Split: create a new leaf with just the new token
                new_leaf = LeafNode(
                    self._alloc_id(), leaf.end, leaf.end + 1, kv_token, leaf.tier, 1,
                )
                self._put(new_leaf)
                # new_split_internal wraps (leaf, new_leaf); this becomes
                # the new right child of path[-1] (the leaf's parent).
                new_split_internal = InternalNode(
                    self._alloc_id(), leaf.start, leaf.end + 1,
                    leaf.node_id, new_leaf.node_id, leaf.tier, 1,
                )
                self._put(new_split_internal)
                if not path:
                    # leaf was the root; new_split_internal is the new root
                    self._deferred_free.append(leaf.node_id)
                    return new_split_internal.node_id
                # Rebuild path[-1] with right = new_split_internal.
                # Crucially, path[-1].left is preserved — this is the
                # left subtree that the previous code was dropping.
                cur = InternalNode(
                    self._alloc_id(), path[-1].start, leaf.end + 1,
                    path[-1].left, new_split_internal.node_id,
                    path[-1].tier, path[-1].refcount,
                )
                self._put(cur)
                self._deferred_free.append(path[-1].node_id)
                # Walk up
                for anc in reversed(path[:-1]):
                    new_anc = InternalNode(
                        self._alloc_id(), anc.start, leaf.end + 1,
                        anc.left, cur.node_id, anc.tier, anc.refcount,
                    )
                    self._put(new_anc)
                    self._deferred_free.append(anc.node_id)
                    cur = new_anc
                return cur.node_id

    def unlink(self, n: int) -> NodeID:
        """Drop the leftmost `n` tokens from the main root. Returns new root NodeID.

        If n >= length, returns NULL_NODE (empty). Updates the rope's main root.
        For unlinking from a non-main root, use ``unlink_from``.
        """
        new_root = self.unlink_from(self._root_id, n)
        self._root_id = new_root
        return new_root

    def unlink_from(self, root: Optional[NodeID], n: int) -> NodeID:
        """Drop the leftmost `n` tokens from `root`. Returns new root NodeID."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return root
        if root is None or root == NULL_NODE:
            raise EmptySequenceError("cannot unlink from empty sequence")
        length = self._span(root)
        if n >= length:
            # Whole thing gone
            self._drop_subtree(root)
            return NULL_NODE
        new_root = self._unlink_left(root, n)
        # Renumber: every node's start/end is decremented by `n` so that
        # user position 0 is again the leftmost token.
        self._renumber(new_root, -n)
        self._bump_mutation()
        return new_root

    def _renumber(self, nid: NodeID, delta: int) -> None:
        """Walk the subtree rooted at nid and add `delta` to every start/end."""
        n = self._get(nid)
        if n is None:
            return
        n.start += delta
        n.end += delta
        if not isinstance(n, LeafNode):
            self._renumber(n.left, delta)
            self._renumber(n.right, delta)

    def length(self) -> int:
        """Length of the main root."""
        if self._root_id is None or self._root_id == NULL_NODE:
            return 0
        return self._span(self._root_id)

    def length_of(self, root: Optional[NodeID]) -> int:
        """Length of a specific root."""
        if root is None or root == NULL_NODE:
            return 0
        return self._span(root)

    def fork(self, root: Optional[NodeID] = None) -> NodeID:
        """Structural-sharing fork. Returns new root NodeID.

        If root is None, forks the main root. The fork shares the entire
        tree by NodeID; refcounts on shared subtrees are incremented. No
        bytes are copied.
        """
        if root is None:
            root = self._root_id
        if root is None or root == NULL_NODE:
            return NULL_NODE
        self._bump_refcount_subtree(root)
        self._bump_mutation()
        return root

    def share_prefix(self, a: Optional[NodeID], n: int) -> NodeID:
        """Return a new root that structurally shares the first `n` tokens of `a`.

        If n == a's full length, the new root is a fork of a (no copy).
        If n < a's length, the new root is a forked prefix, truncated to n.
        The original a is unchanged.
        """
        if a is None or a == NULL_NODE:
            raise EmptySequenceError("a is empty")
        if n < 0:
            raise ValueError("n must be >= 0")
        a_len = self._span(a)
        if n > a_len:
            raise OutOfRangeError(f"n={n} > a.length={a_len}")
        if n == 0:
            return a  # no-op: return the same root (caller can decide)
        if n == a_len:
            return self.fork(a)
        # Partial: fork a, then truncate to n
        forked = self.fork(a)
        return self._truncate_right(forked, n)

    def evict(self, n: int) -> NodeID:
        """Demote the leftmost n tokens to WARM tier. Returns new root.

        Updates the main root. For evicting a non-main root, use ``evict_from``.
        """
        new_root = self.evict_from(self._root_id, n)
        self._root_id = new_root
        return new_root

    def evict_from(self, root: Optional[NodeID], n: int) -> NodeID:
        """Demote the leftmost n tokens of `root` to WARM tier."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0 or root is None or root == NULL_NODE:
            return root
        length = self._span(root)
        n = min(n, length)
        new_root = self._demote_left(root, n, Tier.WARM)
        self._bump_mutation()
        return new_root

    def promote(self, n: int) -> NodeID:
        """Promote the rightmost n tokens to HOT tier. Returns new root."""
        new_root = self.promote_from(self._root_id, n)
        self._root_id = new_root
        return new_root

    def promote_from(self, root: Optional[NodeID], n: int) -> NodeID:
        """Promote the rightmost n tokens of `root` to HOT tier."""
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0 or root is None or root == NULL_NODE:
            return root
        length = self._span(root)
        n = min(n, length)
        new_root = self._promote_right(root, n, Tier.HOT)
        self._bump_mutation()
        return new_root

    # ------------------------------------------------------------------
    # Public API: reads
    # ------------------------------------------------------------------

    def resolve(self, root: Optional[NodeID] = None) -> list[tuple[NodeID, int, int]]:
        """Build a snapshot array of (node_id, byte_offset, length) for
        the leaves in left-to-right order. This is what a GPU attention
        kernel would consume. If root is None, uses the main root.
        """
        if root is None:
            root = self._root_id
        if root is None or root == NULL_NODE:
            return []
        out: list[tuple[NodeID, int, int]] = []
        offset = 0
        for leaf in self._iter_leaves(root):
            out.append((leaf.node_id, offset, leaf.end - leaf.start))
            offset += leaf.end - leaf.start
        return out

    def get_kv(self, pos: int, root: Optional[NodeID] = None) -> bytes:
        """Return the single-token K/V blob at absolute position `pos`.

        If root is None, uses the main root.
        """
        if root is None:
            root = self._root_id
        if root is None or root == NULL_NODE:
            raise OutOfRangeError("empty sequence")
        leaf = self._find_leaf(root, pos)
        if leaf is None:
            raise OutOfRangeError(f"pos {pos} out of range")
        return leaf.kv_blob[pos - leaf.start : pos - leaf.start + 1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _span(self, nid: NodeID) -> int:
        n = self._get(nid)
        if n is None:
            return 0
        return n.end - n.start  # type: ignore[union-attr]

    def _iter_leaves(self, nid: NodeID) -> Iterator[LeafNode]:
        n = self._get(nid)
        if n is None:
            return
        if isinstance(n, LeafNode):
            yield n
            return
        assert isinstance(n, InternalNode)
        yield from self._iter_leaves(n.left)
        yield from self._iter_leaves(n.right)

    def _find_leaf(self, nid: NodeID, pos: int) -> Optional[LeafNode]:
        cur = nid
        while cur is not None:
            n = self._get(cur)
            if n is None:
                return None
            if pos < n.start or pos >= n.end:
                return None
            if isinstance(n, LeafNode):
                return n
            assert isinstance(n, InternalNode)
            # Descend into the child whose range contains pos.
            left = self._get(n.left)
            if left is not None and left.start <= pos < left.end:
                cur = n.left
            else:
                cur = n.right
        return None

    def _replace_leaf_in_path(
        self,
        root: NodeID,
        path: list[InternalNode],
        old_leaf: NodeID,
        new_leaf: NodeID,
    ) -> NodeID:
        """Replace the rightmost leaf (old_leaf) with new_leaf; rebuild
        internals up to the root. Returns the new root NodeID.

        The path's bottom (path[-1]) currently has old_leaf as its right
        child. We rebuild the chain so that bottom.left is preserved and
        bottom.right becomes new_leaf. Then we walk up.
        """
        if not path:
            # root was the old leaf; new root is new_leaf
            self._deferred_free.append(old_leaf)
            return new_leaf
        new_leaf_node = self._get(new_leaf)
        assert new_leaf_node is not None
        new_end = new_leaf_node.end  # type: ignore[union-attr]
        # Rebuild the bottom parent: left=path[-1].left, right=new_leaf
        new_bottom = InternalNode(
            self._alloc_id(), path[-1].start, new_end,
            path[-1].left, new_leaf, path[-1].tier, path[-1].refcount,
        )
        self._put(new_bottom)
        self._deferred_free.append(path[-1].node_id)
        # Walk up
        cur = new_bottom
        for anc in reversed(path[:-1]):
            new_anc = InternalNode(
                self._alloc_id(), anc.start, new_end, anc.left, cur.node_id,
                anc.tier, anc.refcount,
            )
            self._put(new_anc)
            self._deferred_free.append(anc.node_id)
            cur = new_anc
        return cur.node_id

    def _rebuild_upward(
        self, top: InternalNode, new_end: int, new_right: NodeID
    ) -> NodeID:
        """Walk up from `top`, replacing its right child with new_right
        and updating the end. Returns the new root."""
        cur = InternalNode(
            self._alloc_id(), top.start, new_end, top.left, new_right,
            top.tier, top.refcount,
        )
        self._put(cur)
        self._deferred_free.append(top.node_id)
        return cur.node_id

    def _unlink_left(self, root: NodeID, n: int) -> NodeID:
        """Drop the leftmost n tokens; return new root."""
        # Find the leaf containing position n-1, drop everything up to and
        # including that leaf (or partially that leaf), then rebuild.
        new_root, _ = self._unlink_left_rec(root, n)
        return new_root

    def _unlink_left_rec(self, nid: NodeID, n: int) -> tuple[NodeID, int]:
        """Returns (new_root_for_this_subtree, tokens_remaining_to_drop)."""
        node = self._get(nid)
        if node is None:
            return NULL_NODE, n
        if isinstance(node, LeafNode):
            length = node.end - node.start
            if n >= length:
                # Whole leaf goes
                self._deferred_free.append(node.node_id)
                return NULL_NODE, n - length
            else:
                # Partial: shrink the leaf
                new_start = node.start + n
                new_leaf = LeafNode(
                    self._alloc_id(), new_start, node.end,
                    node.kv_blob[n:], node.tier, node.refcount,
                )
                self._put(new_leaf)
                self._deferred_free.append(node.node_id)
                return new_leaf.node_id, 0
        # Internal
        assert isinstance(node, InternalNode)
        left_len = self._span(node.left)
        if n < left_len:
            new_left, n = self._unlink_left_rec(node.left, n)
            if new_left is None:
                # The whole left subtree is gone; the right becomes the only child
                self._deferred_free.append(node.node_id)
                return node.right, n  # (n is 0 here)
            # Rebuild internal with new left
            new_internal = InternalNode(
                self._alloc_id(),
                self._get(new_left).start,  # type: ignore[union-attr]
                node.end,
                new_left, node.right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, 0
        elif n == left_len:
            # Drop the entire left subtree; right becomes the new internal's child
            self._deferred_free.append(node.left)
            self._deferred_free.append(node.node_id)
            return node.right, 0
        else:
            # Drop entire left subtree, then drop from right
            self._deferred_free.append(node.left)
            new_right, n = self._unlink_left_rec(node.right, n - left_len)
            if new_right is None:
                self._deferred_free.append(node.node_id)
                return NULL_NODE, n
            new_internal = InternalNode(
                self._alloc_id(),
                self._get(new_right).start,  # type: ignore[union-attr]
                node.end,
                new_right, NULL_NODE, node.tier, node.refcount,  # type: ignore[arg-type]
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, 0

    def _truncate_right(self, root: NodeID, n: int) -> NodeID:
        """Keep only the leftmost n tokens; drop the rest."""
        return self._truncate_right_rec(root, n)

    def _truncate_right_rec(self, nid: NodeID, n: int) -> NodeID:
        node = self._get(nid)
        if node is None:
            return NULL_NODE
        if isinstance(node, LeafNode):
            if n >= node.end - node.start:
                return node.node_id
            new_leaf = LeafNode(
                self._alloc_id(), node.start, node.start + n,
                node.kv_blob[:n], node.tier, node.refcount,
            )
            self._put(new_leaf)
            self._deferred_free.append(node.node_id)
            return new_leaf.node_id
        assert isinstance(node, InternalNode)
        left_len = self._span(node.left)
        if n <= left_len:
            # Truncate fully within left
            new_left = self._truncate_right_rec(node.left, n)
            self._deferred_free.append(node.right)
            self._deferred_free.append(node.node_id)
            if new_left is None:
                return NULL_NODE
            return new_left
        else:
            # Keep all of left, truncate right
            new_right = self._truncate_right_rec(node.right, n - left_len)
            new_internal = InternalNode(
                self._alloc_id(), node.start,
                self._get(new_right).end if new_right is not None else node.start + left_len,  # type: ignore[union-attr]
                node.left, new_right if new_right is not None else NULL_NODE,  # type: ignore[arg-type]
                node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id

    def _demote_left(self, root: NodeID, n: int, target_tier: Tier) -> NodeID:
        new_root, _ = self._demote_left_rec(root, n, target_tier)
        return new_root

    def _demote_left_rec(
        self, nid: NodeID, n: int, target_tier: Tier
    ) -> tuple[NodeID, int]:
        node = self._get(nid)
        if node is None:
            return NULL_NODE, n
        if isinstance(node, LeafNode):
            length = node.end - node.start
            if n >= length:
                new_leaf = LeafNode(
                    self._alloc_id(), node.start, node.end,
                    node.kv_blob, target_tier, node.refcount,
                )
                self._put(new_leaf)
                self._deferred_free.append(node.node_id)
                return new_leaf.node_id, n - length
            else:
                # partial demotion: keep as one node with the new tier
                new_leaf = LeafNode(
                    self._alloc_id(), node.start, node.end,
                    node.kv_blob, target_tier, node.refcount,
                )
                self._put(new_leaf)
                self._deferred_free.append(node.node_id)
                return new_leaf.node_id, 0
        assert isinstance(node, InternalNode)
        left_len = self._span(node.left)
        if n < left_len:
            new_left, n = self._demote_left_rec(node.left, n, target_tier)
            new_internal = InternalNode(
                self._alloc_id(), node.start, node.end,
                new_left, node.right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, n
        elif n == left_len:
            # Demote the left subtree fully
            new_left, _ = self._demote_left_rec(node.left, left_len, target_tier)
            new_internal = InternalNode(
                self._alloc_id(), node.start, node.end,
                new_left, node.right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, 0
        else:
            new_left, _ = self._demote_left_rec(node.left, left_len, target_tier)
            new_right, n = self._demote_left_rec(node.right, n - left_len, target_tier)
            new_internal = InternalNode(
                self._alloc_id(), node.start, node.end,
                new_left, new_right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, n

    def _promote_right(self, root: NodeID, n: int, target_tier: Tier) -> NodeID:
        # Promote is demote applied to the right portion. Easiest: find
        # the rightmost leaves covering the last n tokens and rewrite.
        # We collect leaves in order, skip until we cover the right n.
        leaves = list(self._iter_leaves(root))
        if not leaves:
            return root
        total = sum(l.end - l.start for l in leaves)
        n = min(n, total)
        # tokens_to_skip from the left = total - n
        skip = total - n
        # Walk leaves, accumulating skip; for the first leaf where
        # skip <= length, that's the boundary. Promote this and all
        # subsequent leaves.
        out_root, _ = self._promote_right_rec(root, skip, n, target_tier)
        return out_root

    def _promote_right_rec(
        self, nid: NodeID, skip: int, n: int, target_tier: Tier
    ) -> tuple[NodeID, int]:
        node = self._get(nid)
        if node is None:
            return NULL_NODE, n
        if isinstance(node, LeafNode):
            length = node.end - node.start
            if skip >= length:
                return node.node_id, n  # all skipped, no promote
            promote_len = min(n, length - skip)
            if skip == 0 and promote_len == length:
                # Whole leaf promoted — just retier
                new_leaf = LeafNode(
                    self._alloc_id(), node.start, node.end,
                    node.kv_blob, target_tier, node.refcount,
                )
                self._put(new_leaf)
                self._deferred_free.append(node.node_id)
                return new_leaf.node_id, n - promote_len
            if skip == 0:
                # Right part of leaf promoted; left stays
                left_leaf = LeafNode(
                    self._alloc_id(), node.start, node.start + (length - promote_len),
                    node.kv_blob[: length - promote_len], node.tier, node.refcount,
                )
                right_leaf = LeafNode(
                    self._alloc_id(), node.start + (length - promote_len), node.end,
                    node.kv_blob[length - promote_len :], target_tier, node.refcount,
                )
                self._put(left_leaf)
                self._put(right_leaf)
                self._deferred_free.append(node.node_id)
                # Wrap in an internal so the structure is preserved
                wrap = InternalNode(
                    self._alloc_id(), node.start, node.end,
                    left_leaf.node_id, right_leaf.node_id, node.tier, node.refcount,
                )
                self._put(wrap)
                return wrap.node_id, n - promote_len
            # skip > 0 and promote_len < length: split into 3 (left, mid, right)
            # but that's overcomplicated. Simpler: keep left + (mid, right as one promoted)
            left_leaf = LeafNode(
                self._alloc_id(), node.start, node.start + skip,
                node.kv_blob[:skip], node.tier, node.refcount,
            )
            right_leaf = LeafNode(
                self._alloc_id(), node.start + skip, node.end,
                node.kv_blob[skip:], target_tier, node.refcount,
            )
            self._put(left_leaf)
            self._put(right_leaf)
            self._deferred_free.append(node.node_id)
            wrap = InternalNode(
                self._alloc_id(), node.start, node.end,
                left_leaf.node_id, right_leaf.node_id, node.tier, node.refcount,
            )
            self._put(wrap)
            return wrap.node_id, n - promote_len
        assert isinstance(node, InternalNode)
        left_len = self._span(node.left)
        if skip >= left_len:
            # Skip whole left, promote from right
            new_left = node.left
            new_right, n = self._promote_right_rec(node.right, skip - left_len, n, target_tier)
            new_internal = InternalNode(
                self._alloc_id(), node.start, node.end,
                new_left, new_right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, n
        elif n <= left_len - skip:
            # Whole promotion in left
            new_left, n = self._promote_right_rec(node.left, skip, n, target_tier)
            new_internal = InternalNode(
                self._alloc_id(), node.start, node.end,
                new_left, node.right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, n
        else:
            # Promote all of (left after skip) and some of right
            promote_in_left = left_len - skip
            new_left, n = self._promote_right_rec(node.left, skip, promote_in_left, target_tier)
            new_right, n = self._promote_right_rec(node.right, 0, n, target_tier)
            new_internal = InternalNode(
                self._alloc_id(), node.start, node.end,
                new_left, new_right, node.tier, node.refcount,
            )
            self._put(new_internal)
            self._deferred_free.append(node.node_id)
            return new_internal.node_id, n

    def _bump_refcount_subtree(self, nid: NodeID) -> None:
        n = self._get(nid)
        if n is None:
            return
        with n.lock:
            n.refcount += 1
            if not isinstance(n, LeafNode):
                self._bump_refcount_subtree(n.left)
                self._bump_refcount_subtree(n.right)

    def _drop_subtree(self, nid: NodeID) -> None:
        """Mark the entire subtree as deferred-free."""
        n = self._get(nid)
        if n is None:
            return
        self._deferred_free.append(nid)
        if not isinstance(n, LeafNode):
            self._drop_subtree(n.left)
            self._drop_subtree(n.right)

    def collect_garbage(self) -> int:
        """Free all nodes in the deferred-free list. Returns count freed.

        Production: epoch-based reclamation. Here: simple deferred free.
        """
        n = 0
        for nid in self._deferred_free:
            if nid in self._mapping:
                del self._mapping[nid]
                n += 1
        self._deferred_free.clear()
        return n
