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
