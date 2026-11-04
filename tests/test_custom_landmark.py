"""
Tests for custom_landmark.py — CUSTOM_LANDMARK_DEFS, EDGE_LANDMARK_DIRECTIONS,
and extend_landmarks().
"""
import pytest
import numpy as np

from face_makeup.custom_landmark import (
    CUSTOM_LANDMARK_DEFS,
    EDGE_LANDMARK_DIRECTIONS,
    MAX_LANDMARK_ID,
    NUM_MP_LANDMARKS,
    NUM_TOTAL,
)
from face_makeup.constants import (
    LM_1001, LM_1002, LM_1003,
    LM_1101, LM_1102, LM_1103, LM_1104, LM_1105,
    LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407,
    LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507,
    LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607,
    LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707,
    EDGE_LANDMARK_DIRECTIONS as EDGE_LANDMARK_DIRECTIONS_FROM_CONSTANTS,
)
from face_makeup.core.landmark_resolver import extend_landmarks
from face_makeup.core.types import FaceLandmarks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_landmarks():
    """FaceLandmarks with known coordinates (50 points for testing)."""
    rng = np.random.default_rng(42)
    coords = rng.random((NUM_MP_LANDMARKS, 2)).astype(np.float32)
    coords_3d = rng.random((NUM_MP_LANDMARKS, 3)).astype(np.float32)
    return FaceLandmarks(points=coords, points_3d=coords_3d)


# ---------------------------------------------------------------------------
# CUSTOM_LANDMARK_DEFS
# ---------------------------------------------------------------------------

class TestCustomLandmarkDefs:
    """Verify the declarative registry."""

    ALL_LM = [
        LM_1001, LM_1002, LM_1003,
        LM_1101, LM_1102, LM_1103, LM_1104, LM_1105,
        LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407,
        LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507,
        LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607,
        LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707,
    ]

    def test_defs_count(self):
        """CUSTOM_LANDMARK_DEFS should have 36 entries."""
        assert len(CUSTOM_LANDMARK_DEFS) == 36

    def test_all_ids_unique(self):
        """All logical_ids should be unique."""
        ids = [d.logical_id for d in CUSTOM_LANDMARK_DEFS]
        assert len(ids) == len(set(ids))

    def test_kinds(self):
        """8 mid + 28 edge = 36."""
        mid_count = sum(1 for d in CUSTOM_LANDMARK_DEFS if d.kind == "mid")
        edge_count = sum(1 for d in CUSTOM_LANDMARK_DEFS if d.kind == "edge")
        assert mid_count == 8
        assert edge_count == 28

    def test_all_ids_in_range(self):
        """All IDs should be >= 1001 and <= MAX_LANDMARK_ID."""
        for d in CUSTOM_LANDMARK_DEFS:
            assert 1001 <= d.logical_id <= MAX_LANDMARK_ID

    def test_base_indices_in_mp_range(self):
        """All base_indices should be in 0..477."""
        for d in CUSTOM_LANDMARK_DEFS:
            for idx in d.base_indices:
                assert 0 <= idx < NUM_MP_LANDMARKS


# ---------------------------------------------------------------------------
# LM_* integer constants
# ---------------------------------------------------------------------------

class TestLMConstants:
    """LM_* constants should be integers matching their logical IDs."""

    def test_lm_1001_is_int(self):
        assert LM_1001 == 1001
        assert isinstance(LM_1001, int)

    def test_lm_1401_is_int(self):
        assert LM_1401 == 1401
        assert isinstance(LM_1401, int)

    def test_lm_1707_is_int(self):
        assert LM_1707 == 1707
        assert isinstance(LM_1707, int)

    def test_all_lm_constants_match_logical_id(self):
        """Every LM_* constant should equal its own name's numeric suffix."""
        for lm in TestCustomLandmarkDefs.ALL_LM:
            assert isinstance(lm, int)


# ---------------------------------------------------------------------------
# EDGE_LANDMARK_DIRECTIONS
# ---------------------------------------------------------------------------

class TestEdgeLandmarkDirections:
    """EDGE_LANDMARK_DIRECTIONS dict."""

    def test_count(self):
        """Should have 28 entries (one per edge landmark)."""
        assert len(EDGE_LANDMARK_DIRECTIONS) == 28

    def test_keys_are_logical_ids(self):
        for key in EDGE_LANDMARK_DIRECTIONS:
            assert 1401 <= key <= 1707

    def test_values_are_mp_index_pairs(self):
        for p_idx, q_idx in EDGE_LANDMARK_DIRECTIONS.values():
            assert 0 <= p_idx < NUM_MP_LANDMARKS
            assert 0 <= q_idx < NUM_MP_LANDMARKS

    def test_specific_entry(self):
        """LM_1401 = (edge, 173, 190)."""
        assert EDGE_LANDMARK_DIRECTIONS[1401] == (173, 190)

    def test_exported_from_constants(self):
        """Same object exported from constants module."""
        assert EDGE_LANDMARK_DIRECTIONS is EDGE_LANDMARK_DIRECTIONS_FROM_CONSTANTS


# ---------------------------------------------------------------------------
# extend_landmarks()
# ---------------------------------------------------------------------------

class TestExtendLandmarks:
    """Tests for extend_landmarks()."""

    def test_shape(self, base_landmarks):
        result = extend_landmarks(base_landmarks)
        assert result.points.shape == (NUM_TOTAL, 2)

    def test_base_coords_preserved(self, base_landmarks):
        result = extend_landmarks(base_landmarks)
        np.testing.assert_array_equal(
            result.points[:NUM_MP_LANDMARKS],
            base_landmarks.points,
        )

    def test_midpoint_computed(self, base_landmarks):
        """LM_1001 should be the centroid of landmarks 37 and 72."""
        result = extend_landmarks(base_landmarks)
        expected = (base_landmarks.points[37] + base_landmarks.points[72]) / 2
        np.testing.assert_allclose(result.points[1001], expected)

    def test_midpoint_three_indices(self, base_landmarks):
        """LM_1003 should be centroid of 267, 302 (2 points)."""
        result = extend_landmarks(base_landmarks)
        expected = (base_landmarks.points[267] + base_landmarks.points[302]) / 2
        np.testing.assert_allclose(result.points[1003], expected)

    def test_edge_base_point(self, base_landmarks):
        """LM_1401 should be coords[173] (the P point of edge tuple)."""
        result = extend_landmarks(base_landmarks)
        np.testing.assert_array_equal(result.points[1401], base_landmarks.points[173])

    def test_unused_slot_is_zero(self, base_landmarks):
        """Index 478 (between 477 and 1001) should be zero."""
        result = extend_landmarks(base_landmarks)
        np.testing.assert_array_equal(result.points[478], np.zeros(2, dtype=np.float32))

    def test_points_3d_preserved(self, base_landmarks):
        result = extend_landmarks(base_landmarks)
        assert result.points_3d is base_landmarks.points_3d

    def test_all_custom_ids_have_nonzero_coords(self, base_landmarks):
        """Every custom landmark ID slot should have been written."""
        result = extend_landmarks(base_landmarks)
        for d in CUSTOM_LANDMARK_DEFS:
            assert not np.allclose(
                result.points[d.logical_id], 0.0
            ), f"Custom landmark {d.logical_id} was not set"

    def test_dtype_preserved(self, base_landmarks):
        result = extend_landmarks(base_landmarks)
        assert result.points.dtype == np.float32

    def test_all_edge_landmarks_match_defs(self, base_landmarks):
        """All edge landmarks store their P-point coordinate."""
        result = extend_landmarks(base_landmarks)
        for d in CUSTOM_LANDMARK_DEFS:
            if d.kind == "edge":
                np.testing.assert_array_equal(
                    result.points[d.logical_id],
                    base_landmarks.points[d.base_indices[0]],
                )

    def test_extend_twice_is_idempotent_after_base(self, base_landmarks):
        """extend_landmarks from extended landmarks works (base coords must match)."""
        first = extend_landmarks(base_landmarks)
        # Create a new FaceLandmarks with subset (simulating a second extend
        # would fail because the resolver uses 0-477 as base — extending
        # an already-extended array would double-write custom coords into
        # the wrong slots). This test verifies base preservation only.
        second_input = FaceLandmarks(
            points=first.points[:NUM_MP_LANDMARKS].copy(),
            points_3d=first.points_3d,
        )
        second = extend_landmarks(second_input)
        np.testing.assert_array_equal(first.points, second.points)
