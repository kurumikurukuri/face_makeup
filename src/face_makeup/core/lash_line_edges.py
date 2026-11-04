"""
Distinct lash-line edge landmarks of an eye, shared by the rule and ML correctors.
"""
from __future__ import annotations

from ..constants import EDGE_LANDMARK_DIRECTIONS, EYELINER_CORRECTION_MAP


def unique_edges(part_name: str) -> list[tuple[int, bool]]:
    """Return each edge landmark of a part once, in slot order, with its ``is_upper`` flag.

    An edge on the lower polyline is a lower-lid edge. The edge shared by both
    polylines (1507/1707 also start the upper one) is therefore corrected once
    as a lower-lid edge, so both polylines share the same point.

    Raises:
        KeyError: if *part_name* is not an eyeliner part.
    """
    correction_map = EYELINER_CORRECTION_MAP[part_name]
    lower = set(correction_map["lower"])
    edges: dict[int, bool] = {}
    for indices in correction_map.values():
        for idx in indices:
            if idx in EDGE_LANDMARK_DIRECTIONS and idx not in edges:
                edges[idx] = idx not in lower
    return list(edges.items())
