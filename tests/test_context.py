"""
Tests for ImageContext class.
"""
import pytest
from unittest import mock
import numpy as np
import cv2
import glob
import os
import tempfile

from face_makeup.processor import MakeupProcessor
from face_makeup.core.types import LipsParams, ColorParams


@pytest.fixture
def processor():
    """MakeupProcessor instance."""
    return MakeupProcessor()


@pytest.fixture
def test_image():
    """Load first test image."""
    image_files = sorted(glob.glob("tests/test_images/*.png"))
    assert len(image_files) > 0, "No test images found"
    image = cv2.imread(image_files[0])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture
def context(processor, test_image):
    """ImageContext with single face."""
    return processor.open(test_image)


class TestImageContextProperties:
    """Tests for ImageContext properties."""

    def test_original_image_returns_copy(self, context, test_image):
        """Test that original_image returns a copy."""
        original = context.original_image
        assert original.shape == test_image.shape
        # Modify returned image
        original[0, 0] = [255, 255, 255]
        # Original should be unchanged
        assert not np.array_equal(original, context.original_image)

    def test_faces_property(self, context):
        """Test faces property returns face list."""
        faces = context.faces
        assert isinstance(faces, list)
        assert len(faces) >= 1

    def test_face_count_property(self, context):
        """Test face_count property."""
        assert context.face_count == len(context.faces)

    def test_face_method_valid_index(self, context):
        """Test face() method with valid index."""
        face = context.face(0)
        assert face is not None
        assert face.face_id == 0

    def test_face_method_invalid_index(self, context):
        """Test face() method with invalid index."""
        with pytest.raises(IndexError):
            context.face(999)


class TestImageContextRender:
    """Tests for ImageContext.render() method."""

    def test_render_without_makeup(self, context, test_image):
        """Test render without any makeup applied."""
        result = context.render()
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        # Without makeup, result should be same as original
        assert np.array_equal(result, test_image)

    def test_render_with_makeup(self, context, test_image):
        """Test render with makeup applied."""
        lip_params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        context.faces[0].lip(lip_params)

        result = context.render()
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        # With makeup, result should be different
        assert not np.array_equal(result, test_image)

    def test_render_caching(self, context):
        """Test that render results are cached."""
        lip_params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        context.faces[0].lip(lip_params)

        result1 = context.render()
        result2 = context.render()

        # Should be equal (cached)
        assert np.array_equal(result1, result2)

    def test_render_cache_invalidation(self, context):
        """Test cache invalidation on makeup change."""
        lip_params1 = LipsParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=0.5
        )
        context.faces[0].lip(lip_params1)
        result1 = context.render()

        # Add more makeup (should invalidate cache)
        lip_params2 = LipsParams(
            color_params=ColorParams(target_rgb=(0, 255, 0)),
            opacity=0.5
        )
        context.faces[0].lip(lip_params2)
        result2 = context.render()

        # Results should be different
        assert not np.array_equal(result1, result2)


class TestImageContextSave:
    """Tests for ImageContext.save() method."""

    def test_save_creates_file(self, context):
        """Test that save creates output file."""
        lip_params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        context.faces[0].lip(lip_params)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            context.save(output_path)
            assert os.path.exists(output_path)

            # Verify saved image
            saved = cv2.imread(output_path)
            assert saved is not None
            assert saved.shape[:2] == context.original_image.shape[:2]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestImageContextShow:
    """Tests for ImageContext.show() method."""

    def test_show_calls_cv2_functions(self, context):
        """Test that show() calls cv2.imshow and cv2.waitKey."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey') as mock_waitkey:

            context.faces[0].lip(color=(255, 100, 100), opacity=0.5)
            context.show(window_name="test_window", wait_key=1)

            mock_imshow.assert_called_once()
            # Verify window name is passed
            assert mock_imshow.call_args[0][0] == "test_window"
            mock_waitkey.assert_called_once_with(1)

    def test_show_default_parameters(self, context):
        """Test show() with default parameters."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey') as mock_waitkey:

            context.show()

            mock_imshow.assert_called_once()
            # Default window name is "preview"
            assert mock_imshow.call_args[0][0] == "preview"
            # Default wait_key is 0
            mock_waitkey.assert_called_once_with(0)
