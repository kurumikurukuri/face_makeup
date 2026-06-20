"""
Custom (virtual) landmarks for MediaPipe Face Mesh.

Format:
  ('mid', idx1, idx2, ...) — centroid of 2+ landmarks, computed at runtime
  ('edge', idx1, idx2)     — direction vector P->Q (used by eyeliner corrector)
"""
import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# Lips
# ---------------------------------------------------------------------------
LM_1001 = ("mid", 37, 72)
LM_1002 = ("mid", 0, 11)
LM_1003 = ("mid", 267, 302)

LM_1101 = ("mid", 179, 178)
LM_1102 = ("mid", 86, 87)
LM_1103 = ("mid", 15, 14)
LM_1104 = ("mid", 316, 317)
LM_1105 = ("mid", 403, 402)

# ---------------------------------------------------------------------------
# Eyelid (left)
# ---------------------------------------------------------------------------
LM_1201 = ("mid", 247, 33)
LM_1202 = ("mid", 30, 161)
LM_1203 = ("mid", 29, 160)
LM_1204 = ("mid", 27, 159)
LM_1205 = ("mid", 28, 158)
LM_1206 = ("mid", 56, 157)
LM_1207 = ("mid", 190, 243)

# ---------------------------------------------------------------------------
# Eyelid (right)
# ---------------------------------------------------------------------------
LM_1301 = ("mid", 467, 263)
LM_1302 = ("mid", 260, 388)
LM_1303 = ("mid", 259, 387)
LM_1304 = ("mid", 257, 386)
LM_1305 = ("mid", 258, 385)
LM_1306 = ("mid", 286, 384)
LM_1307 = ("mid", 414, 463)

# ---------------------------------------------------------------------------
# Eyeliner edge direction (used by EyelinerCorrector)
# ---------------------------------------------------------------------------
LM_1401 = ("edge", 173, 190)
LM_1402 = ("edge", 157,  56)
LM_1403 = ("edge", 158,  28)
LM_1404 = ("edge", 159,  27)
LM_1405 = ("edge", 160,  29)
LM_1406 = ("edge", 161,  30)
LM_1407 = ("edge", 246, 247)
LM_1408 = ("edge",  33, 113)

LM_1501 = ("edge",   7,  25)
LM_1502 = ("edge", 163, 110)
LM_1503 = ("edge", 144,  24)
LM_1504 = ("edge", 145,  23)
LM_1505 = ("edge", 153,  22)
LM_1506 = ("edge", 154,  26)
LM_1507 = ("edge", 155, 112)

LM_1601 = ("edge", 398, 414)
LM_1602 = ("edge", 384, 286)
LM_1603 = ("edge", 385, 258)
LM_1604 = ("edge", 386, 257)
LM_1605 = ("edge", 387, 259)
LM_1606 = ("edge", 388, 260)
LM_1607 = ("edge", 466, 467)
LM_1608 = ("edge", 263, 342)

LM_1701 = ("edge", 249, 255)
LM_1702 = ("edge", 390, 339)
LM_1703 = ("edge", 373, 254)
LM_1704 = ("edge", 374, 253)
LM_1705 = ("edge", 380, 252)
LM_1706 = ("edge", 381, 256)
LM_1707 = ("edge", 382, 341)


def resolve_indices(indices: List, coords: np.ndarray) -> np.ndarray:
    """
    Resolve a landmark index list to coordinates.
    Expands ('mid', idx1, idx2, ...) tuples to centroid coordinates.

    Args:
        indices: List of int or ('mid', ...) tuples.
        coords: All landmark coordinates (N, 2).

    Returns:
        np.ndarray: Resolved coordinates (M, 2).

    Example:
        indices = [33, ('mid', 33, 133), 159]
        -> [coord[33], (coord[33]+coord[133])/2, coord[159]]
    """
    expanded = []
    for idx in indices:
        if isinstance(idx, tuple) and idx[0] == 'mid':
            mid_indices = idx[1:]
            if len(mid_indices) < 2:
                raise ValueError(f"'mid' requires at least 2 indices, got {len(mid_indices)}")
            n = len(coords)
            oob = [i for i in mid_indices if i < 0 or i >= n]
            if oob:
                raise IndexError(f"'mid' indices {oob} out of bounds for coords of length {n}")
            expanded.append(np.mean(coords[list(mid_indices)], axis=0))
        elif isinstance(idx, tuple) and idx[0] == 'edge':
            # ('edge', p_idx, q_idx) — return the base landmark P position.
            # Edge detection is applied separately by EyelinerCorrector;
            # here we just need a drawable coordinate.
            p_idx = idx[1]
            n = len(coords)
            if p_idx < 0 or p_idx >= n:
                raise IndexError(f"'edge' index {p_idx} out of bounds for coords of length {n}")
            expanded.append(coords[p_idx])
        elif isinstance(idx, int):
            n = len(coords)
            if idx < 0 or idx >= n:
                raise IndexError(f"index {idx} out of bounds for coords of length {n}")
            expanded.append(coords[idx])
        else:
            raise ValueError(f"Invalid index type: {type(idx)}. Expected int, ('mid', ...) or ('edge', ...) tuple.")
    return np.array(expanded, dtype=coords.dtype)
