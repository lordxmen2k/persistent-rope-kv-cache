"""Lock-free concurrent rope for KV cache indexing.

Position-keyed persistent data structure with Bw-tree-style mapping-table
indirection. Every mutation is atomic via compare-and-swap; reads traverse
without locks; structural sharing gives free branching.

See tech_note.md for the design rationale.
"""
from .types import NodeID, NULL_NODE, Tier, DeltaKind
from .core import (
    PersistentRope,
    RopeError,
    EmptySequenceError,
    OutOfRangeError,
)
from .vllm_compat import RopeBlockManager, RopeBlock, vLLMCompatibleInterface

__all__ = [
    "NodeID",
    "NULL_NODE",
    "Tier",
    "DeltaKind",
    "PersistentRope",
    "RopeError",
    "EmptySequenceError",
    "OutOfRangeError",
    "RopeBlockManager",
    "RopeBlock",
    "vLLMCompatibleInterface",
]
