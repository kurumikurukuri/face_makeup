"""
Tests for constants.py: landmark index validity and uniqueness.
"""
from __future__ import annotations

import pytest

from face_makeup import constants


def _index_lists():
    """Discover all top-level list-type *_INDICES variables."""
    result = {}
    for name in sorted(dir(constants)):
        if name.endswith('_INDICES'):
            value = getattr(constants, name)
            if isinstance(value, list):
                result[name] = value
    return result


def _custom_landmark_ids():
    """Discover all LM_XXXX custom landmark IDs (integers defined in constants)."""
    ids = set()
    for name in dir(constants):
        if name.startswith('LM_') and isinstance(getattr(constants, name), int):
            ids.add(getattr(constants, name))
    return ids


def _valid_range():
    """Return the set of all valid index values: 0–477 + any custom LM_* IDs."""
    return set(range(478)) | _custom_landmark_ids()


class TestIndexValidity:
    """All indices must be within MediaPipe range (0–477) or be a custom
    landmark ID defined in constants."""

    @pytest.mark.parametrize('name,indices', sorted(_index_lists().items()))
    def test_indices_in_valid_range(self, name, indices):
        valid = _valid_range()
        for i, idx in enumerate(indices):
            if isinstance(idx, int):
                assert idx in valid, (
                    f"{name}[{i}]: {idx} not in valid range [0–477]"
                    + (" or custom landmark ID" if _custom_landmark_ids() else "")
                )


class TestNoDuplicates:
    """Each index list must contain no duplicate integer entries."""

    @pytest.mark.parametrize('name,indices', sorted(_index_lists().items()))
    def test_no_duplicates(self, name, indices):
        ints = [i for i in indices if isinstance(i, int)]
        dups = sorted({i for i in ints if ints.count(i) > 1})
        assert not dups, f"{name}: duplicate indices: {dups}"


class TestLashLineUsesEdgeLandmarks:
    """Every lash-line consumer references the correctable edge landmarks.

    The renderer writes the corrected lash line into the edge landmark rows
    (1401-1707), so a list naming the raw MediaPipe P index instead (e.g. 159
    rather than 1404) would silently keep the uncorrected point.
    """

    @staticmethod
    def _raw_edge_points():
        return {p for p, _q in constants.EDGE_LANDMARK_DIRECTIONS.values()}

    @staticmethod
    def _eyelid_lash_lists():
        lists = {
            "EYELID_LASH_LEFT_INDICES": constants.EYELID_LASH_LEFT_INDICES,
            "EYELID_LASH_RIGHT_INDICES": constants.EYELID_LASH_RIGHT_INDICES,
        }
        for name in dir(constants):
            value = getattr(constants, name)
            if name.startswith("EYELID_") and name.endswith("_INDICES") and isinstance(value, tuple):
                lists[f"{name}[lash]"] = value[1]
        return lists

    def test_eyelid_lash_lines_use_edge_ids(self):
        raw = self._raw_edge_points()
        for name, indices in self._eyelid_lash_lists().items():
            assert not raw & set(indices), f"{name} uses raw P indices {sorted(raw & set(indices))}"

    def test_eyelid_zones_are_left_right_symmetric(self):
        """Zone lash lists map onto each other via left (14xx) <-> right (16xx) ids."""
        lists = self._eyelid_lash_lists()
        for name, left in lists.items():
            if "LEFT" not in name:
                continue
            right = lists[name.replace("LEFT", "RIGHT")]
            left_edges = [i for i in left if 1400 <= i < 1500]
            right_edges = [i for i in right if 1600 <= i < 1700]
            assert [i + 200 for i in left_edges] == right_edges, name

    def test_full_lash_line_covers_every_upper_edge(self):
        assert constants.EYELID_LASH_LEFT_INDICES[2:9] == list(range(1401, 1408))
        assert constants.EYELID_LASH_RIGHT_INDICES[2:9] == list(range(1601, 1608))

    def test_tear_bag_and_foundation_use_edge_ids(self):
        raw = self._raw_edge_points()
        for name in (
            "TEAR_BAG_ROOT_LEFT_INDICES", "TEAR_BAG_ROOT_RIGHT_INDICES",
            "EYELINER_UPPER_LEFT_INDICES", "EYELINER_LOWER_LEFT_INDICES",
            "EYELINER_UPPER_RIGHT_INDICES", "EYELINER_LOWER_RIGHT_INDICES",
        ):
            assert not raw & set(getattr(constants, name)), name
