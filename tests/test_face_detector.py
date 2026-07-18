"""
Tests for FaceDetector class.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.face_makeup.core.face_detector import FaceDetector
from src.face_makeup.core.types import FaceBBox


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


class TestFaceDetectorBboxClamping:
    """Regression tests for boundary clamping of partial/edge-of-frame faces."""

    @staticmethod
    def _detect_with_fake_relative_bbox(xmin, ymin, width, height, image_shape):
        """Bypasses real MediaPipe inference so the relative_bounding_box
        values (including out-of-[0,1] ones MediaPipe can return for
        partial faces) are fully controlled."""
        detector = FaceDetector.__new__(FaceDetector)
        detector._closed = False

        fake_bbox = MagicMock(xmin=xmin, ymin=ymin, width=width, height=height)
        fake_detection = MagicMock(score=[0.9])
        fake_detection.location_data.relative_bounding_box = fake_bbox
        fake_results = MagicMock(detections=[fake_detection])

        detector.detector = MagicMock()
        detector.detector.process.return_value = fake_results

        image = np.zeros(image_shape, dtype=np.uint8)
        return detector.detect(image)[0]

    def test_negative_xmin_ymin_does_not_overstate_width_height(self):
        """A face partially outside the top-left of the frame can report a
        negative xmin/ymin. Width/height must be derived from the clamped
        right/bottom edge, not the raw relative size, or the reported bbox
        extends further into the frame than the actual detection."""
        bbox = self._detect_with_fake_relative_bbox(
            xmin=-0.1, ymin=-0.1, width=0.5, height=0.5,
            image_shape=(1000, 1000, 3),
        )
        assert bbox.x == 0
        assert bbox.y == 0
        # True right/bottom edge: (xmin + width) * dim = 0.4 * 1000 = 400.
        assert bbox.w == 400
        assert bbox.h == 400

    def test_bottom_right_overflow_still_clamped(self):
        """A face partially outside the bottom-right is clamped to the frame."""
        bbox = self._detect_with_fake_relative_bbox(
            xmin=0.9, ymin=0.9, width=0.3, height=0.3,
            image_shape=(1000, 1000, 3),
        )
        assert bbox.x == 900
        assert bbox.y == 900
        assert bbox.w == 100
        assert bbox.h == 100


class TestFaceDetectorCleanup:
    """Tests for FaceDetector resource cleanup."""

    def test_del_hasattr_guard(self):
        """__del__ safe when detector attribute was never set."""
        detector = FaceDetector.__new__(FaceDetector)
        detector.__del__()
