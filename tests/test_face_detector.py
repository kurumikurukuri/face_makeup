"""
Tests for FaceDetector class.
"""
import glob

import cv2
import numpy as np
import pytest

from src.face_makeup.core.face_detector import FaceDetector
from src.face_makeup.core.types import FaceBBox


@pytest.fixture
def test_image():
    """Load first available test image as RGB."""
    image_files = sorted(glob.glob("tests/test_images/*.png"))
    assert len(image_files) > 0, "No test images found"
    image = cv2.imread(image_files[0])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class TestFaceDetectorInit:
    """Tests for FaceDetector construction."""

    def test_default_confidence(self):
        detector = FaceDetector()
        assert detector.min_detection_confidence == 0.5
        assert detector.detector is not None

    def test_custom_confidence(self):
        detector = FaceDetector(min_detection_confidence=0.8)
        assert detector.min_detection_confidence == 0.8


class TestFaceDetectorDetect:
    """Tests for FaceDetector.detect() input validation and output."""

    @pytest.fixture
    def detector(self):
        return FaceDetector()

    def test_none_image_raises(self, detector):
        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(None)

    def test_empty_image_raises(self, detector):
        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(np.array([], dtype=np.uint8))

    def test_no_face_in_blank_image_raises(self, detector):
        with pytest.raises(ValueError, match="No face detected"):
            detector.detect(np.zeros((256, 256, 3), dtype=np.uint8))

    def test_detect_returns_bboxes(self, detector, test_image):
        bboxes = detector.detect(test_image)
        assert isinstance(bboxes, list)
        assert len(bboxes) >= 1
        for bbox in bboxes:
            assert isinstance(bbox, FaceBBox)

    def test_bbox_attributes_absolute_coords(self, detector, test_image):
        bboxes = detector.detect(test_image)
        for bbox in bboxes:
            assert bbox.x >= 0
            assert bbox.y >= 0
            assert bbox.w > 0
            assert bbox.h > 0
            assert 0.0 <= bbox.score <= 1.0


class TestFaceDetectorCleanup:
    """Tests for FaceDetector resource cleanup."""

    def test_del_hasattr_guard(self):
        """__del__ safe when detector attribute was never set."""
        detector = FaceDetector.__new__(FaceDetector)
        detector.__del__()
