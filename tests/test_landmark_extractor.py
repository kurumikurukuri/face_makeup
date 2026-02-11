"""
Tests for LandmarkExtractor class.
"""
import numpy as np
import pytest

from src.face_makeup.core.landmark_extractor import LandmarkExtractor
from src.face_makeup.core.types import FaceLandmarks


class TestLandmarkExtractorInit:
    """Tests for LandmarkExtractor construction."""

    def test_default_params(self):
        extractor = LandmarkExtractor()
        assert extractor.face_mesh is not None

    def test_custom_params(self):
        extractor = LandmarkExtractor(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
        )
        assert extractor.face_mesh is not None


class TestLandmarkExtractorExtract:
    """Tests for LandmarkExtractor.extract() input validation and output."""

    @pytest.fixture
    def extractor(self):
        return LandmarkExtractor()

    def test_none_image_raises(self, extractor):
        with pytest.raises(ValueError, match="Invalid image"):
            extractor.extract(None)

    def test_empty_image_raises(self, extractor):
        with pytest.raises(ValueError, match="Invalid image"):
            extractor.extract(np.array([], dtype=np.uint8))

    def test_extract_returns_landmarks(self, extractor, test_image):
        landmarks = extractor.extract(test_image)
        assert isinstance(landmarks, FaceLandmarks)
        assert landmarks.points.shape[1] == 2
        assert landmarks.points.shape[0] == 478
        assert landmarks.points_3d.shape == (478, 3)

    def test_to_image_coords(self, extractor, test_image):
        landmarks = extractor.extract(test_image)
        h, w = test_image.shape[:2]
        coords = landmarks.to_image_coords(w, h)
        assert coords.shape == (478, 2)
        assert coords.dtype == np.int32
        assert coords[:, 0].max() <= w
        assert coords[:, 1].max() <= h


class TestLandmarkExtractorExtractAll:
    """Tests for LandmarkExtractor.extract_all() input validation and output."""

    @pytest.fixture
    def extractor(self):
        return LandmarkExtractor(max_num_faces=3)

    def test_none_image_raises(self, extractor):
        with pytest.raises(ValueError, match="Invalid image"):
            extractor.extract_all(None)

    def test_empty_image_raises(self, extractor):
        with pytest.raises(ValueError, match="Invalid image"):
            extractor.extract_all(np.array([], dtype=np.uint8))

    def test_extract_all_returns_list(self, extractor, test_image):
        results = extractor.extract_all(test_image)
        assert isinstance(results, list)
        assert len(results) >= 1
        for landmarks in results:
            assert isinstance(landmarks, FaceLandmarks)
            assert landmarks.points.shape[1] == 2


class TestLandmarkExtractorCleanup:
    """Tests for LandmarkExtractor resource cleanup."""

    def test_del_hasattr_guard(self):
        """__del__ safe when face_mesh attribute was never set."""
        extractor = LandmarkExtractor.__new__(LandmarkExtractor)
        extractor.__del__()
