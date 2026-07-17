"""
Tests for rule-based eyeliner edge correction (2nd-derivative zero-crossing).
"""
import pytest
import numpy as np

from face_makeup.core.eyeliner_corrector import EyelinerCorrector
from face_makeup.core.types import FaceLandmarks
from face_makeup.core.landmark_resolver import extend_landmarks


@pytest.fixture
def corrector():
    """EyelinerCorrector instance."""
    return EyelinerCorrector()


@pytest.fixture
def sample_landmarks():
    """Sample face landmarks (478 points, normalized 0-1, extended with custom)."""
    rng = np.random.RandomState(42)
    points = rng.rand(478, 2).astype(np.float32)
    return extend_landmarks(FaceLandmarks(points=points))


@pytest.fixture
def uniform_color_image():
    """Uniform colour RGB image (no edges)."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def edge_color_image():
    """RGB image with a clear vertical edge at x=90 (dark left, bright right)."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :90] = 30    # dark left (eyeliner)
    img[:, 90:] = 200   # bright right (skin)
    return img


@pytest.fixture
def uniform_gray():
    """Uniform grayscale image (no edge)."""
    return np.full((200, 200), 128, dtype=np.uint8)


@pytest.fixture
def edge_gray():
    """Grayscale image with a dark->light transition at x=90."""
    gray = np.zeros((200, 200), dtype=np.uint8)
    gray[:, :90] = 30
    gray[:, 90:] = 200
    return gray


class TestFindEyelinerEdge:
    """Tests for EyelinerCorrector.find_edge (operates on a grayscale image)."""

    def test_uniform_image_returns_original(self, corrector, uniform_gray):
        """On a uniform image, should return (px, py) (no edge to detect)."""
        result = corrector.find_edge(uniform_gray, 50, 100, 150, 100)
        assert result == (50, 100)

    def test_zero_length_vector_returns_original(self, corrector, uniform_gray):
        """When P == Q (zero-length vector), return (px, py)."""
        result = corrector.find_edge(uniform_gray, 100, 100, 100, 100)
        assert result == (100, 100)

    def test_returns_integer_tuple(self, corrector, uniform_gray):
        """Return type should be (int, int)."""
        result = corrector.find_edge(uniform_gray, 50.7, 60.3, 150.2, 60.8)
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_detects_dark_to_light_transition(self, corrector, edge_gray):
        """Sampling P(dark)->Q(bright) should detect the edge near x=90."""
        result = corrector.find_edge(edge_gray, 50, 100, 150, 100)
        detected_x, detected_y = result
        assert 80 <= detected_x <= 100, f"detected_x={detected_x} not near edge x=90"
        assert abs(detected_y - 100) <= 2

    def test_debug_info_contains_derivative_keys(self, corrector, edge_gray):
        """Debug dict should carry the rule-based derivative analysis data."""
        _, debug = corrector.find_edge(
            edge_gray, 50, 100, 150, 100, return_debug_info=True
        )
        assert debug is not None
        for key in (
            'samples', 'first_derivative', 'second_derivative',
            't_values', 'zero_crossings', 'best_idx',
        ):
            assert key in debug

    def test_no_edge_returns_fallback_with_debug(self, corrector, uniform_gray):
        """Uniform image: fallback point but debug dict still populated, best_idx None."""
        result, debug = corrector.find_edge(
            uniform_gray, 50, 100, 150, 100, return_debug_info=True
        )
        assert result == (50, 100)
        assert debug is not None
        assert debug['best_idx'] is None

    def test_zero_length_debug_is_none(self, corrector, uniform_gray):
        """P == Q yields a None debug dict (no sampling performed)."""
        result, debug = corrector.find_edge(
            uniform_gray, 100, 100, 100, 100, return_debug_info=True
        )
        assert result == (100, 100)
        assert debug is None

    def test_subpixel_norm_skips_sampling(self, corrector, uniform_gray):
        """0 < norm < 1e-6 (sub-px landmark-rounding residue) short-circuits.

        The widened guard (norm < 1e-6) must bail before sampling, so the
        debug dict is None. A strict ``norm == 0`` guard would let norm=1e-7
        through, run the sampler, and return a populated (non-None) debug
        dict — so the ``debug is None`` assertion locks the widened range.
        """
        result, debug = corrector.find_edge(
            uniform_gray, 100.0, 100.0, 100.0 + 1e-7, 100.0,
            return_debug_info=True,
        )
        assert result == (100, 100)
        assert debug is None


class TestPreprocess:
    """Tests for EyelinerCorrector.preprocess."""

    def test_returns_2d_uint8(self, corrector, uniform_color_image):
        """RGB image should be reduced to a 2D uint8 grayscale image."""
        gray = corrector.preprocess(uniform_color_image)
        assert gray.ndim == 2
        assert gray.dtype == np.uint8
        assert gray.shape == uniform_color_image.shape[:2]


class TestCorrectEyelinerPoints:
    """Tests for EyelinerCorrector.correct_points (accepts an RGB image)."""

    def test_returns_float32_array(self, corrector, uniform_color_image, sample_landmarks):
        """Output should be float32 ndarray."""
        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        result = corrector.correct_points(coords, 'eyeliner_left', eyeliner=uniform_color_image)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_returns_correct_point_count(self, corrector, uniform_color_image, sample_landmarks):
        """Should return points for all upper + lower indices."""
        from face_makeup.constants import EYELINER_CORRECTION_MAP

        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected_count = sum(
                len(indices) for indices in correction_map.values()
            )
            result = corrector.correct_points(coords, part_name, eyeliner=uniform_color_image)
            assert result.shape == (expected_count, 2), (
                f"{part_name}: expected {expected_count} points, got {result.shape[0]}"
            )

    def test_both_eyeliner_parts(self, corrector, uniform_color_image, sample_landmarks):
        """Should work for both eyeliner_left and eyeliner_right."""
        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)

        for part_name in ('eyeliner_left', 'eyeliner_right'):
            result = corrector.correct_points(coords, part_name, eyeliner=uniform_color_image)
            assert result.ndim == 2
            assert result.shape[1] == 2

    def test_invalid_part_name_raises(self, corrector, uniform_color_image, sample_landmarks):
        """Non-eyeliner part should raise ValueError."""
        h, w = uniform_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        with pytest.raises(ValueError, match="Unknown part_name"):
            corrector.correct_points(coords, 'lips', eyeliner=uniform_color_image)


class TestGetEyelinerCorrectionDebug:
    """Tests for EyelinerCorrector.get_debug_entries."""

    def test_returns_list_of_dicts(self, corrector, sample_landmarks):
        """Should return a list of debug dicts."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        entries = corrector.get_debug_entries(
            image, sample_landmarks, (200, 200), 'eyeliner_left', side='left'
        )
        assert isinstance(entries, list)
        assert len(entries) > 0
        assert isinstance(entries[0], dict)

    def test_debug_entries_contain_required_keys(self, corrector, sample_landmarks):
        """Each entry should have px, py, qx, qy, detected_x, detected_y, side."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        entries = corrector.get_debug_entries(
            image, sample_landmarks, (200, 200), 'eyeliner_left', side='left'
        )
        required_keys = {'px', 'py', 'qx', 'qy', 'detected_x', 'detected_y', 'side'}
        for entry in entries:
            assert required_keys.issubset(entry.keys())

    def test_side_propagated_correctly(self, corrector, sample_landmarks):
        """Side value should be propagated to each entry."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        for part_name, expected_side in [('eyeliner_left', 'left'), ('eyeliner_right', 'right')]:
            entries = corrector.get_debug_entries(
                image, sample_landmarks, (200, 200), part_name, side=expected_side
            )
            for entry in entries:
                assert entry['side'] == expected_side

    def test_invalid_part_name_raises(self, corrector, sample_landmarks):
        """Non-eyeliner part should raise ValueError."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown part_name"):
            corrector.get_debug_entries(
                image, sample_landmarks, (200, 200), 'lips', side='left'
            )

    def test_point_count_matches_correction_map(self, corrector, sample_landmarks):
        """Debug entry count should match total edge indices in correction map."""
        from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS, EYELINER_CORRECTION_MAP
        image = np.full((200, 200, 3), 128, dtype=np.uint8)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected = sum(
                sum(1 for idx in indices if idx in EDGE_LANDMARK_DIRECTIONS)
                for indices in correction_map.values()
            )
            side = 'right' if part_name.endswith('_right') else 'left'
            entries = corrector.get_debug_entries(
                image, sample_landmarks, (200, 200), part_name, side=side
            )
            assert len(entries) == expected

    def test_reuses_cache_from_correct_points(self, corrector, edge_color_image, sample_landmarks):
        """get_debug_entries should reuse entries cached by correct_points (same image)."""
        h, w = edge_color_image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        corrector.correct_points(coords, 'eyeliner_left', eyeliner=edge_color_image)

        entries = corrector.get_debug_entries(
            image=edge_color_image, landmarks=sample_landmarks,
            image_shape=(h, w), part_name='eyeliner_left', side='left',
            eyeliner=edge_color_image,
        )
        assert len(entries) > 0
        # Cached entries are tagged with the source image on return.
        assert all(entry['eyeliner'] is edge_color_image for entry in entries)
