"""
Tests for the distinct lash-line edges shared by the rule and ML correctors.
"""
import pytest

from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS, EYELINER_CORRECTION_MAP
from face_makeup.core.lash_line_edges import unique_edges


@pytest.mark.parametrize("part_name", sorted(EYELINER_CORRECTION_MAP))
def test_each_edge_listed_once_in_slot_order(part_name):
    slots = [
        idx
        for indices in EYELINER_CORRECTION_MAP[part_name].values()
        for idx in indices
        if idx in EDGE_LANDMARK_DIRECTIONS
    ]
    edge_ids = [idx for idx, _ in unique_edges(part_name)]
    assert edge_ids == list(dict.fromkeys(slots))
    assert len(edge_ids) == 14


@pytest.mark.parametrize("part_name, shared", [
    ("eyeliner_left", 1507), ("eyeliner_right", 1707),
])
def test_shared_edge_is_a_lower_lid_edge(part_name, shared):
    flags = dict(unique_edges(part_name))
    assert flags[shared] is False
    lower = set(EYELINER_CORRECTION_MAP[part_name]["lower"])
    assert all(is_upper == (idx not in lower) for idx, is_upper in flags.items())


def test_unknown_part_raises():
    with pytest.raises(KeyError):
        unique_edges("lips")
