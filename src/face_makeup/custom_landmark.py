"""
Custom (virtual) landmarks for MediaPipe Face Mesh.

Each custom landmark is defined declaratively in CUSTOM_LANDMARK_DEFS.
Derived data (EDGE_LANDMARK_DIRECTIONS, etc.) is auto-computed at import time.

Adding a new custom landmark requires only:
  1. One entry in CUSTOM_LANDMARK_DEFS
  2. One constant in constants.py (optional, for named access)

Landmark kinds:
  "mid"  — centroid of 2+ base landmarks → array coordinate
  "edge" — direction vector P->Q for lash-line edge correction → P's coordinate stored in array,
           direction info in EDGE_LANDMARK_DIRECTIONS
"""
from typing import NamedTuple


class CustomLandmarkDef(NamedTuple):
    """Definition of a single custom (virtual) landmark.

    Attributes:
        logical_id: Unique identifier, also used as the array index
                    in the extended coordinates array.
        kind: "mid" for centroid computation, "edge" for lash-line edge correction.
        base_indices: Source MediaPipe landmark indices (0-477).
    """

    logical_id: int
    kind: str  # "mid" | "edge"
    base_indices: tuple[int, ...]


# ---------------------------------------------------------------------------
# Single source of truth — add new custom landmarks here
# ---------------------------------------------------------------------------
CUSTOM_LANDMARK_DEFS: list[CustomLandmarkDef] = [
    # Lips — midpoints
    CustomLandmarkDef(1001, "mid", (37, 72)),
    CustomLandmarkDef(1002, "mid", (0, 11)),
    CustomLandmarkDef(1003, "mid", (267, 302)),
    CustomLandmarkDef(1101, "mid", (179, 178)),
    CustomLandmarkDef(1102, "mid", (86, 87)),
    CustomLandmarkDef(1103, "mid", (15, 14)),
    CustomLandmarkDef(1104, "mid", (316, 317)),
    CustomLandmarkDef(1105, "mid", (403, 402)),
    # Lash-line edge — direction vectors (upper left)
    CustomLandmarkDef(1401, "edge", (173, 190)),
    CustomLandmarkDef(1402, "edge", (157, 56)),
    CustomLandmarkDef(1403, "edge", (158, 28)),
    CustomLandmarkDef(1404, "edge", (159, 27)),
    CustomLandmarkDef(1405, "edge", (160, 29)),
    CustomLandmarkDef(1406, "edge", (161, 30)),
    CustomLandmarkDef(1407, "edge", (246, 247)),
    # Lash-line edge — direction vectors (lower left)
    CustomLandmarkDef(1501, "edge", (7, 25)),
    CustomLandmarkDef(1502, "edge", (163, 110)),
    CustomLandmarkDef(1503, "edge", (144, 24)),
    CustomLandmarkDef(1504, "edge", (145, 23)),
    CustomLandmarkDef(1505, "edge", (153, 22)),
    CustomLandmarkDef(1506, "edge", (154, 26)),
    CustomLandmarkDef(1507, "edge", (155, 112)),
    # Lash-line edge — direction vectors (upper right)
    CustomLandmarkDef(1601, "edge", (398, 414)),
    CustomLandmarkDef(1602, "edge", (384, 286)),
    CustomLandmarkDef(1603, "edge", (385, 258)),
    CustomLandmarkDef(1604, "edge", (386, 257)),
    CustomLandmarkDef(1605, "edge", (387, 259)),
    CustomLandmarkDef(1606, "edge", (388, 260)),
    CustomLandmarkDef(1607, "edge", (466, 467)),
    # Lash-line edge — direction vectors (lower right)
    CustomLandmarkDef(1701, "edge", (249, 255)),
    CustomLandmarkDef(1702, "edge", (390, 339)),
    CustomLandmarkDef(1703, "edge", (373, 254)),
    CustomLandmarkDef(1704, "edge", (374, 253)),
    CustomLandmarkDef(1705, "edge", (380, 252)),
    CustomLandmarkDef(1706, "edge", (381, 256)),
    CustomLandmarkDef(1707, "edge", (382, 341)),
]

# ---------------------------------------------------------------------------
# Auto-derived constants (no manual maintenance needed)
# ---------------------------------------------------------------------------
NUM_MP_LANDMARKS: int = 478
MAX_LANDMARK_ID: int = max(d.logical_id for d in CUSTOM_LANDMARK_DEFS)  # 1707
NUM_TOTAL: int = MAX_LANDMARK_ID + 1  # 1708

# logical_id -> (p_mp_idx, q_mp_idx) — edge type only, auto-built from defs
EDGE_LANDMARK_DIRECTIONS: dict[int, tuple[int, int]] = {
    d.logical_id: (d.base_indices[0], d.base_indices[1])
    for d in CUSTOM_LANDMARK_DEFS
    if d.kind == "edge"
}
