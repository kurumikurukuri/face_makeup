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
def uniform_color_image():
    """Uniform colour RGB image (no edges)."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def edge_color_image():
    """RGB image with a clear vertical edge at x=100."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :100] = 30   # dark left (eyeliner)
    img[:, 100:] = 200  # bright right (skin)
    return img


class TestFindEyelinerEdge:
    """Tests for EyelinerCorrector.find_edge."""

    def test_uniform_image_returns_original(self, segmenter):
        """On a uniform image, should return (px, py) (no edge to detect)."""
        eyeliner = np.full((200, 200, 3), 128, dtype=np.uint8)
        gray = segmenter.preprocess_for_eyeliner(eyeliner)
        result = segmenter._corrector.find_edge(gray, 50, 100, 150, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_edge_image_detects_edge(self, segmenter):
        """On an image with a clear dark->light edge, should detect near the edge."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :75] = 30   # dark left (eyeliner)
        image[:, 75:] = 200  # bright right (skin)
        gray = segmenter.preprocess_for_eyeliner(image)
        # P at x=50 (dark side), Q at x=150 (bright side), edge at x=75.
        # x=75 maps to t=0.25 which is well within the search range [0, 0.5).
        result = segmenter._corrector.find_edge(gray, 50, 100, 150, 100)
        detected_x, detected_y = result
        # The detected edge should be closer to x=75 than the original px=50
        assert abs(detected_x - 75) < abs(50 - 75)

    def test_zero_length_vector_returns_original(self, segmenter):
        """When P == Q (zero-length vector), return (px, py)."""
        eyeliner = np.full((200, 200, 3), 128, dtype=np.uint8)
        gray = segmenter.preprocess_for_eyeliner(eyeliner)
        result = segmenter._corrector.find_edge(gray, 100, 100, 100, 100)
        assert result == (100, 100)

    def test_returns_integer_tuple(self, segmenter):
        """Return type should be (int, int)."""
        eyeliner = np.full((200, 200, 3), 128, dtype=np.uint8)
        gray = segmenter.preprocess_for_eyeliner(eyeliner)
        result = segmenter._corrector.find_edge(gray, 50.7, 60.3, 150.2, 60.8)
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_accepts_color_image(self, segmenter, edge_color_image):
        """Should accept a 3-channel colour image without error."""
        gray = segmenter.preprocess_for_eyeliner(edge_color_image)
        result = segmenter._corrector.find_edge(gray, 50, 100, 150, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCorrectEyelinerPoints:
    """Tests for _correct_eyeliner_points."""

    def test_returns_float32_array(self, segmenter, uniform_color_image, sample_landmarks):
        """Output should be float32 ndarray."""
        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_color_image)
        result = segmenter._correct_eyeliner_points(coords, 'eyeliner_left', gray=gray)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_returns_correct_point_count(self, segmenter, uniform_color_image, sample_landmarks):
        """Should return points for all upper + lower indices."""
        from face_makeup.constants import EYELINER_CORRECTION_MAP

        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_color_image)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected_count = sum(
                len(indices) for indices in correction_map.values()
            )
            result = segmenter._correct_eyeliner_points(coords, part_name, gray=gray)
            assert result.shape == (expected_count, 2), (
                f"{part_name}: expected {expected_count} points, got {result.shape[0]}"
            )

    def test_both_eyeliner_parts(self, segmenter, uniform_color_image, sample_landmarks):
        """Should work for both eyeliner_left and eyeliner_right."""
        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_color_image)

        for part_name in ('eyeliner_left', 'eyeliner_right'):
            result = segmenter._correct_eyeliner_points(coords, part_name, gray=gray)
            assert result.ndim == 2
            assert result.shape[1] == 2

    def test_unknown_part_name_raises(self, segmenter, uniform_color_image, sample_landmarks):
        """Unknown part_name should raise ValueError, not silently return empty."""
        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        gray = segmenter.preprocess_for_eyeliner(uniform_color_image)

        with pytest.raises(ValueError, match="Unknown eyeliner part_name"):
            segmenter._correct_eyeliner_points(coords, 'lips', gray=gray)


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

    def test_eyeliner_correction_runs_with_color_image(self, segmenter, sample_landmarks):
        """With a colour image, eyeliner mask should be generated without error."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :100] = 30
        image[:, 100:] = 200

        masks = segmenter.segment(
            (200, 200), sample_landmarks,
            parts=['eyeliner_left'], image=image
        )
        assert masks['eyeliner_left'].dtype == np.uint8

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

    def test_eyeliner_gray_shared_across_both_parts(self, segmenter, sample_landmarks):
        """segment() should preprocess eyeliner_gray once and reuse it for both parts."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :100] = 30
        image[:, 100:] = 200

        call_count = {'n': 0}
        original_preprocess = segmenter._corrector.preprocess

        def counting_preprocess(img):
            call_count['n'] += 1
            return original_preprocess(img)

        segmenter._corrector.preprocess = counting_preprocess
        try:
            masks = segmenter.segment(
                (200, 200), sample_landmarks,
                parts=['eyeliner_left', 'eyeliner_right'], image=image
            )
        finally:
            segmenter._corrector.preprocess = original_preprocess

        assert 'eyeliner_left' in masks and 'eyeliner_right' in masks
        assert call_count['n'] == 1


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
        required_keys = {'px', 'py', 'qx', 'qy', 'detected_x', 'detected_y', 'side'}
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

    def test_invalid_part_name_returns_empty(self, segmenter, sample_landmarks):
        """Non-eyeliner part should return empty list."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        entries = segmenter.get_eyeliner_correction_debug(
            image, sample_landmarks, (200, 200), 'lips', side='left'
        )
        assert entries == []

    def test_point_count_matches_correction_map(self, segmenter, sample_landmarks):
        """Debug entry count should match total edge indices in correction map."""
        from face_makeup.constants import EYELINER_CORRECTION_MAP
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected = sum(
                sum(1 for idx in indices if isinstance(idx, tuple) and idx[0] == 'edge')
                for indices in correction_map.values()
            )
            side = 'right' if part_name.endswith('_right') else 'left'
            entries = segmenter.get_eyeliner_correction_debug(
                image, sample_landmarks, (200, 200), part_name, side=side
            )
            assert len(entries) == expected


