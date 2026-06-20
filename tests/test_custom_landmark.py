"""
Tests for custom_landmark.py — LM_* constants and resolve_indices().
"""
import pytest
import numpy as np

from face_makeup.custom_landmark import (
    resolve_indices,
    LM_1001, LM_1002, LM_1003,
    LM_1101, LM_1102, LM_1103, LM_1104, LM_1105,
    LM_1201, LM_1202, LM_1203, LM_1204, LM_1205, LM_1206, LM_1207,
    LM_1301, LM_1302, LM_1303, LM_1304, LM_1305, LM_1306, LM_1307,
    LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407, LM_1408,
    LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507,
    LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607, LM_1608,
    LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707,
)
from face_makeup.constants import resolve_indices as resolve_indices_from_constants


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coords():
    """Deterministic coordinate array (50 points, 2D)."""
    rng = np.random.default_rng(0)
    return rng.random((50, 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# LM_* constant format
# ---------------------------------------------------------------------------

class TestCustomLandmarkConstants:
    """Verify LM_* constants have the expected tuple format."""

    @pytest.mark.parametrize("lm, expected_tag", [
        (LM_1001, "mid"), (LM_1002, "mid"), (LM_1003, "mid"),
        (LM_1101, "mid"), (LM_1102, "mid"), (LM_1103, "mid"),
        (LM_1104, "mid"), (LM_1105, "mid"),
        (LM_1201, "mid"), (LM_1202, "mid"), (LM_1203, "mid"),
        (LM_1204, "mid"), (LM_1205, "mid"), (LM_1206, "mid"), (LM_1207, "mid"),
        (LM_1301, "mid"), (LM_1302, "mid"), (LM_1303, "mid"),
        (LM_1304, "mid"), (LM_1305, "mid"), (LM_1306, "mid"), (LM_1307, "mid"),
    ])
    def test_mid_constants_are_tuples_with_two_indices(self, lm, expected_tag):
        assert isinstance(lm, tuple)
        assert lm[0] == expected_tag
        assert len(lm) == 3  # ('mid', idx1, idx2)

    @pytest.mark.parametrize("lm", [
        LM_1401, LM_1402, LM_1403, LM_1404,
        LM_1405, LM_1406, LM_1407, LM_1408,
    ])
    def test_edge_constants_are_tuples(self, lm):
        assert isinstance(lm, tuple)
        assert lm[0] == "edge"
        assert len(lm) == 3  # ('edge', idx1, idx2)

    def test_spot_check_lm_1001(self):
        assert LM_1001 == ("mid", 37, 72)

    def test_spot_check_lm_1201(self):
        assert LM_1201 == ("mid", 247, 33)

    @pytest.mark.parametrize("lm", [
        LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507,
    ])
    def test_lower_left_eyeliner_edge_constants(self, lm):
        assert isinstance(lm, tuple)
        assert lm[0] == "edge"
        assert len(lm) == 3

    @pytest.mark.parametrize("lm", [
        LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607, LM_1608,
    ])
    def test_upper_right_eyeliner_edge_constants(self, lm):
        assert isinstance(lm, tuple)
        assert lm[0] == "edge"
        assert len(lm) == 3

    @pytest.mark.parametrize("lm", [
        LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707,
    ])
    def test_lower_right_eyeliner_edge_constants(self, lm):
        assert isinstance(lm, tuple)
        assert lm[0] == "edge"
        assert len(lm) == 3

    def test_spot_check_lm_1501(self):
        assert LM_1501 == ("edge", 7, 25)

    def test_spot_check_lm_1601(self):
        assert LM_1601 == ("edge", 398, 414)

    def test_spot_check_lm_1701(self):
        assert LM_1701 == ("edge", 249, 255)

    def test_spot_check_lm_1401(self):
        assert LM_1401 == ("edge", 173, 190)


# ---------------------------------------------------------------------------
# resolve_indices — normal cases
# ---------------------------------------------------------------------------

class TestResolveIndices:
    """Tests for resolve_indices()."""

    def test_plain_ints_return_correct_coords(self, coords):
        indices = [0, 5, 10]
        result = resolve_indices(indices, coords)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[0], coords[0])
        np.testing.assert_array_equal(result[1], coords[5])
        np.testing.assert_array_equal(result[2], coords[10])

    def test_mid_two_points_returns_midpoint(self, coords):
        indices = [("mid", 0, 1)]
        result = resolve_indices(indices, coords)
        expected = (coords[0] + coords[1]) / 2
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], expected)

    def test_mid_three_points_returns_centroid(self, coords):
        indices = [("mid", 0, 1, 2)]
        result = resolve_indices(indices, coords)
        expected = (coords[0] + coords[1] + coords[2]) / 3
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], expected)

    def test_mixed_int_and_mid(self, coords):
        indices = [3, ("mid", 4, 5), 6]
        result = resolve_indices(indices, coords)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[0], coords[3])
        np.testing.assert_allclose(result[1], (coords[4] + coords[5]) / 2)
        np.testing.assert_array_equal(result[2], coords[6])

    def test_preserves_dtype(self, coords):
        assert coords.dtype == np.float32
        result = resolve_indices([0, 1], coords)
        assert result.dtype == np.float32

    def test_empty_indices_returns_empty_array(self, coords):
        result = resolve_indices([], coords)
        assert len(result) == 0

    def test_single_int_returns_one_point(self, coords):
        result = resolve_indices([7], coords)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], coords[7])

    def test_lm_constant_used_directly(self):
        """LM_* tuple used directly in an index list resolves correctly."""
        # LM_1001 = ("mid", 37, 72) — needs at least 73 points
        lm_coords = np.arange(200).reshape(100, 2).astype(np.float32)
        result = resolve_indices([LM_1001], lm_coords)
        expected = (lm_coords[37] + lm_coords[72]) / 2
        np.testing.assert_allclose(result[0], expected)

    def test_edge_constant_resolves_to_base_point(self):
        """'edge' constants resolve to the first (base) landmark position."""
        # LM_1501 = ("edge", 7, 25) — base point is index 7
        lm_coords = np.arange(100).reshape(50, 2).astype(np.float32)
        result = resolve_indices([LM_1501], lm_coords)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], lm_coords[7])

    def test_edge_constants_1601_and_1701_resolve_correctly(self):
        """LM_1601 and LM_1701 (right-eye groups) resolve to their base points."""
        lm_coords = np.arange(1000).reshape(500, 2).astype(np.float32)
        # LM_1601 = ("edge", 398, 414) — base point 398
        r1 = resolve_indices([LM_1601], lm_coords)
        np.testing.assert_array_equal(r1[0], lm_coords[398])
        # LM_1701 = ("edge", 249, 255) — base point 249
        r2 = resolve_indices([LM_1701], lm_coords)
        np.testing.assert_array_equal(r2[0], lm_coords[249])


# ---------------------------------------------------------------------------
# resolve_indices — error cases
# ---------------------------------------------------------------------------

class TestResolveIndicesErrors:
    """Error handling in resolve_indices()."""

    def test_mid_with_single_index_raises(self, coords):
        with pytest.raises(ValueError, match="at least 2 indices"):
            resolve_indices([("mid", 0)], coords)

    def test_invalid_type_raises(self, coords):
        with pytest.raises(ValueError, match="Invalid index type"):
            resolve_indices(["not_an_int"], coords)

    def test_float_index_raises(self, coords):
        with pytest.raises(ValueError, match="Invalid index type"):
            resolve_indices([1.5], coords)


# ---------------------------------------------------------------------------
# Re-export from constants
# ---------------------------------------------------------------------------

class TestResolveIndicesReexport:
    """resolve_indices is accessible via constants module."""

    def test_resolve_indices_imported_from_constants(self, coords):
        indices = [0, ("mid", 1, 2)]
        result = resolve_indices_from_constants(indices, coords)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], coords[0])
        np.testing.assert_allclose(result[1], (coords[1] + coords[2]) / 2)

    def test_same_function_object(self):
        assert resolve_indices is resolve_indices_from_constants
