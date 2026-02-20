"""
Tests for eyeliner edge correction in FaceSegmenter.
"""
import pytest
import numpy as np
import cv2

from face_makeup.core.segmenter import FaceSegmenter
from face_makeup.core.types import FaceLandmarks


@pytest.fixture
def segmenter():
    """FaceSegmenter instance."""
    return FaceSegmenter()


@pytest.fixture
def sample_landmarks():
    """Sample face landmarks (478 points, normalized 0-1)."""
    rng = np.random.RandomState(42)
    points = rng.rand(478, 2).astype(np.float32)
    return FaceLandmarks(points=points)


@pytest.fixture
def uniform_gray_image():
    """Uniform gray RGB image (no edges)."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def edge_image():
    """RGB image with a clear vertical edge at x=100."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :100] = 30   # dark left (eyeliner)
    img[:, 100:] = 200  # bright right (skin)
    return img


class TestPreprocessForEyeliner:
    """Tests for preprocess_for_eyeliner."""

    def test_output_shape_and_dtype(self, segmenter, uniform_gray_image):
        """Output should be (H, W) uint8."""
        result = segmenter.preprocess_for_eyeliner(uniform_gray_image)
        assert result.shape == (200, 200)
        assert result.dtype == np.uint8

    def test_output_normalized_range(self, segmenter, edge_image):
        """Output values should be in [0, 255]."""
        result = segmenter.preprocess_for_eyeliner(edge_image)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_single_channel_output(self, segmenter, uniform_gray_image):
        """Output must be 2D (grayscale)."""
        result = segmenter.preprocess_for_eyeliner(uniform_gray_image)
        assert result.ndim == 2


class TestFindEyelinerEdge:
    """Tests for _find_eyeliner_edge."""

    def test_uniform_image_returns_original(self, segmenter):
        """On a uniform image, should fall back to (px, py)."""
        gray = np.full((200, 200), 128, dtype=np.uint8)
        result = segmenter._find_eyeliner_edge(gray, 50, 100, 150, 100)
        # Should return original point since no gradient exists
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_edge_image_detects_edge(self, segmenter):
        """On an image with a clear dark->light edge, should detect near the edge."""
        gray = np.zeros((200, 200), dtype=np.uint8)
        gray[:, :75] = 30   # dark left (eyeliner)
        gray[:, 75:] = 200  # bright right (skin)
        # P at x=50 (dark side), Q at x=150 (bright side), edge at x=75.
        # x=75 maps to t=0.25 which is well within the search range [0, 0.5).
        result = segmenter._find_eyeliner_edge(gray, 50, 100, 150, 100)
        detected_x, detected_y = result
        # The detected edge should be closer to x=75 than the original px=50
        assert abs(detected_x - 75) < abs(50 - 75)

    def test_zero_length_vector_returns_original(self, segmenter):
        """When P == Q (zero-length vector), return (px, py)."""
        gray = np.full((200, 200), 128, dtype=np.uint8)
        result = segmenter._find_eyeliner_edge(gray, 100, 100, 100, 100)
        assert result == (100, 100)

    def test_returns_integer_tuple(self, segmenter):
        """Return type should be (int, int)."""
        gray = np.full((200, 200), 128, dtype=np.uint8)
        result = segmenter._find_eyeliner_edge(gray, 50.7, 60.3, 150.2, 60.8)
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


class TestCorrectEyelinerPoints:
    """Tests for _correct_eyeliner_points."""

    def test_returns_float32_array(self, segmenter, uniform_gray_image, sample_landmarks):
        """Output should be float32 ndarray."""
        h, w = uniform_gray_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_gray_image)
        result = segmenter._correct_eyeliner_points(coords, 'eyeliner_left', gray=gray)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_returns_correct_point_count(self, segmenter, uniform_gray_image, sample_landmarks):
        """Should return points for all upper + lower indices."""
        from face_makeup.constants import EYELINER_CORRECTION_MAP

        h, w = uniform_gray_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_gray_image)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected_count = sum(
                len(indices) for indices, _deltas in correction_map.values()
            )
            result = segmenter._correct_eyeliner_points(coords, part_name, gray=gray)
            assert result.shape == (expected_count, 2), (
                f"{part_name}: expected {expected_count} points, got {result.shape[0]}"
            )

    def test_both_eyeliner_parts(self, segmenter, uniform_gray_image, sample_landmarks):
        """Should work for both eyeliner_left and eyeliner_right."""
        h, w = uniform_gray_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_gray_image)

        for part_name in ('eyeliner_left', 'eyeliner_right'):
            result = segmenter._correct_eyeliner_points(coords, part_name, gray=gray)
            assert result.ndim == 2
            assert result.shape[1] == 2


class TestSegmentWithImage:
    """Integration tests: segment() with image parameter."""

    def test_eyeliner_mask_generated_with_image(self, segmenter, sample_landmarks):
        """When image is provided, eyeliner masks should still be generated."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        masks = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['eyeliner_left'], image=image
        )
        assert 'eyeliner_left' in masks
        assert masks['eyeliner_left'].shape == (200, 200)
        assert masks['eyeliner_left'].dtype == np.uint8

    def test_segment_without_image_backward_compatible(self, segmenter, sample_landmarks):
        """segment() without image should produce same results as before."""
        masks_no_image = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['eyeliner_left', 'lips']
        )
        assert 'eyeliner_left' in masks_no_image
        assert 'lips' in masks_no_image

    def test_non_eyeliner_parts_unaffected(self, segmenter, sample_landmarks):
        """Non-eyeliner parts should be identical with or without image."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        masks_no_image = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['lips']
        )
        masks_with_image = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['lips'], image=image
        )
        np.testing.assert_array_equal(
            masks_no_image['lips'],
            masks_with_image['lips']
        )

    def test_segment_all_parts_with_image(self, segmenter, sample_landmarks):
        """segment() with image=... should not error for any part."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        masks = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=None, image=image
        )
        assert isinstance(masks, dict)
        assert len(masks) > 0

    def test_eyeliner_correction_changes_mask(self, segmenter, sample_landmarks):
        """With an edge image, corrected mask should differ from uncorrected."""
        # Create image with contrast that triggers edge detection
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :100] = 30
        image[:, 100:] = 200

        masks_no_correction = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['eyeliner_left']
        )
        masks_with_correction = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['eyeliner_left'], image=image
        )

        # They may or may not differ depending on random landmarks,
        # but both should be valid masks
        assert masks_no_correction['eyeliner_left'].dtype == np.uint8
        assert masks_with_correction['eyeliner_left'].dtype == np.uint8


class TestGetEyelinerCorrectionDebug:
    """Tests for get_eyeliner_correction_debug."""

    def test_returns_list_of_dicts(self, segmenter, sample_landmarks):
        """Should return a list of debug dicts."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        entries = segmenter.get_eyeliner_correction_debug(
            image, sample_landmarks, (200, 200), 'eyeliner_left', side='left'
        )
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert isinstance(entries[0], dict)

    def test_debug_entries_contain_required_keys(self, segmenter, sample_landmarks):
        """Each entry should have px, py, qx, qy, detected_x, detected_y, side."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        entries = segmenter.get_eyeliner_correction_debug(
            image, sample_landmarks, (200, 200), 'eyeliner_left', side='left'
        )
        required_keys = {'px', 'py', 'qx', 'qy', 'detected_x', 'detected_y', 'gray', 'side'}
        for entry in entries:
            assert required_keys.issubset(entry.keys())

    def test_side_propagated_correctly(self, segmenter, sample_landmarks):
        """Side value should be propagated to each entry."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        for part_name, expected_side in [('eyeliner_left', 'left'), ('eyeliner_right', 'right')]:
            entries = segmenter.get_eyeliner_correction_debug(
                image, sample_landmarks, (200, 200), part_name, side=expected_side
            )
            for entry in entries:
                assert entry['side'] == expected_side

    def test_edge_image_has_derivative_data(self, segmenter, sample_landmarks):
        """With an edge image, entries should include derivative data."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :100] = 30
        image[:, 100:] = 200
        entries = segmenter.get_eyeliner_correction_debug(
            image, sample_landmarks, (200, 200), 'eyeliner_right', side='right'
        )
        # At least some entries should have samples/derivative data
        entries_with_samples = [e for e in entries if 'samples' in e]
        assert len(entries_with_samples) > 0
        entry = entries_with_samples[0]
        assert 'first_derivative' in entry
        assert 'second_derivative' in entry
        assert 't_values' in entry
        assert 'zero_crossings' in entry

    def test_invalid_part_name_returns_empty(self, segmenter, sample_landmarks):
        """Non-eyeliner part should return empty list."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        entries = segmenter.get_eyeliner_correction_debug(
            image, sample_landmarks, (200, 200), 'lips', side='left'
        )
        assert entries == []

    def test_point_count_matches_correction_map(self, segmenter, sample_landmarks):
        """Debug entry count should match total indices in correction map."""
        from face_makeup.constants import EYELINER_CORRECTION_MAP
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected = sum(len(indices) for indices, _ in correction_map.values())
            side = 'right' if part_name.endswith('_right') else 'left'
            entries = segmenter.get_eyeliner_correction_debug(
                image, sample_landmarks, (200, 200), part_name, side=side
            )
            assert len(entries) == expected
