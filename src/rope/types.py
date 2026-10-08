# Copyright 2026 Tico Internet LLC
# SPDX-License-Identifier: Apache-2.0
"""Core types for the rope."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType

# Logical node identifier. 64-bit unsigned; never reused within a process.
NodeID = NewType("NodeID", int)


class Tier(Enum):
    """Storage tier for a leaf's K/V blob."""
    HOT = "hot"      # HBM (in real deployment) / in-process (in this reference)
    WARM = "warm"    # host DRAM / secondary in this reference
    COLD = "cold"    # NVMe / opaque spill slot in this reference


class DeltaKind(Enum):
    """Type of a delta record attached to a base node (Bw-tree style)."""
    LEAF_SPLIT = "leaf_split"           # a leaf overflowed; replaced by two leaves
    APPEND_CHILD = "append_child"       # new leaf appended
    REFCOUNT = "refcount"               # shared-subtree refcount change
    TIER = "tier"                       # tier promotion/demotion
    PRINCIPAL = "principal"             # (Phase 4) principal tag change


@dataclass(slots=True)
class Delta:
    """A pending change to a node, applied lazily by readers."""
    kind: DeltaKind
    # The data fields depend on kind. Kept as a single dict for flexibility.
    payload: dict


# Sentinel for "no node".
NULL_NODE: NodeID = NodeID(0)
