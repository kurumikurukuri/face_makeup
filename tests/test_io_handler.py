"""
Tests for IOHandler class.
"""
import pytest
import numpy as np
import tempfile
import os
from unittest import mock

from face_makeup.io_handler import IOHandler


@pytest.fixture
def handler():
    """Create an IOHandler instance."""
    return IOHandler()


@pytest.fixture
def test_rgb_image():
    """Create a test RGB image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 128  # Mid-gray


class TestIOHandlerSave:
    """Tests for IOHandler.save() method."""

    def test_save_creates_file(self, handler, test_rgb_image):
        """Test that save creates a file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            handler.save(test_rgb_image, output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_converts_rgb_to_bgr(self, handler, test_rgb_image):
        """Test that save converts RGB to BGR."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            # Create RGB image with specific colors
            rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = 255  # Red channel
            rgb_image[:, :, 1] = 128  # Green channel
            rgb_image[:, :, 2] = 64   # Blue channel

            handler.save(rgb_image, output_path)

            # Read back with OpenCV (BGR format)
            import cv2
            bgr_image = cv2.imread(output_path)

            # In BGR format, channels are swapped
            # Original RGB (255, 128, 64) becomes BGR (64, 128, 255)
            assert bgr_image[0, 0, 0] == 64   # Blue from original red
            assert bgr_image[0, 0, 1] == 128  # Green stays same
            assert bgr_image[0, 0, 2] == 255  # Red from original blue
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_with_different_formats(self, handler):
        """Test save with different file formats."""
        image = np.ones((20, 20, 3), dtype=np.uint8) * 200

        for suffix in ['.png', '.jpg', '.bmp']:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                output_path = f.name

            try:
                handler.save(image, output_path)
                assert os.path.exists(output_path)
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_save_preserves_image_dimensions(self, handler, test_rgb_image):
        """Test that save preserves image dimensions."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            original_shape = test_rgb_image.shape
            handler.save(test_rgb_image, output_path)

            import cv2
            saved_image = cv2.imread(output_path)
            assert saved_image.shape == original_shape
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestIOHandlerSaveValidation:
    """Tests for IOHandler.save() path validation."""

    def test_save_rejects_empty_path(self, handler, test_rgb_image):
        """Test that save raises ValueError for empty path."""
        with pytest.raises(ValueError, match="must not be empty"):
            handler.save(test_rgb_image, "")

    def test_save_rejects_whitespace_path(self, handler, test_rgb_image):
        """Test that save raises ValueError for whitespace-only path."""
        with pytest.raises(ValueError, match="must not be empty"):
            handler.save(test_rgb_image, "   ")

    def test_save_rejects_unsupported_extension(self, handler, test_rgb_image):
        """Test that save raises ValueError for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported image format"):
            handler.save(test_rgb_image, "output.gif")

    def test_save_rejects_missing_parent_directory(self, handler, test_rgb_image):
        """Test that save raises FileNotFoundError for missing parent dir."""
        with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
            handler.save(test_rgb_image, "nonexistent_subdir/output.png")


class TestIOHandlerPathTraversal:
    """Tests for output_base containment (path traversal protection)."""

    def test_save_rejects_relative_traversal(self, test_rgb_image):
        """Relative path escaping output_base via ../ is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            handler = IOHandler(output_base=tmp)
            with pytest.raises(ValueError, match="escapes output directory"):
                handler.save(test_rgb_image, "../escape.png")

    def test_save_relative_inside_output_base_succeeds(self, test_rgb_image):
        """A relative path inside output_base is written there."""
        with tempfile.TemporaryDirectory() as tmp:
            handler = IOHandler(output_base=tmp)
            handler.save(test_rgb_image, "out.png")
            assert os.path.isfile(os.path.join(tmp, "out.png"))

    def test_save_absolute_path_is_allowed(self, test_rgb_image):
        """Absolute paths are honoured as explicit intent (not contained)."""
        with tempfile.TemporaryDirectory() as output_base:
            # Create an absolute target in the system temp area, outside the
            # output_base directory.
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as abs_file:
                abs_path = abs_file.name
        try:
            handler = IOHandler(output_base=output_base)
            handler.save(test_rgb_image, abs_path)
            assert os.path.isfile(abs_path)
        finally:
            if os.path.exists(abs_path):
                os.unlink(abs_path)


class TestIOHandlerShow:
    """Tests for IOHandler.show() method."""

    def test_show_calls_cv2_functions(self, handler, test_rgb_image):
        """Test that show calls cv2.imshow and cv2.waitKey."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey') as mock_waitkey:

            handler.show(test_rgb_image, window_name="test", wait_key=1)

            mock_imshow.assert_called_once()
            mock_waitkey.assert_called_once_with(1)

    def test_show_default_window_name(self, handler, test_rgb_image):
        """Test show with default window name."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey'):

            handler.show(test_rgb_image)

            # Check that window name is "preview"
            call_args = mock_imshow.call_args
            assert call_args[0][0] == "preview"

    def test_show_default_wait_key(self, handler, test_rgb_image):
        """Test show with default wait_key."""
        with mock.patch('cv2.imshow'), \
             mock.patch('cv2.waitKey') as mock_waitkey:

            handler.show(test_rgb_image)

            # Check that wait_key is called with 0
            mock_waitkey.assert_called_once_with(0)

    def test_show_custom_parameters(self, handler, test_rgb_image):
        """Test show with custom window name and wait_key."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey') as mock_waitkey:

            handler.show(test_rgb_image, window_name="custom", wait_key=100)

            assert mock_imshow.call_args[0][0] == "custom"
            mock_waitkey.assert_called_once_with(100)

    def test_show_converts_rgb_to_bgr(self, handler, test_rgb_image):
        """Test that show converts RGB to BGR."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey'):

            # Create RGB image with specific colors
            rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)
            rgb_image[:, :, 0] = 255  # Red channel
            rgb_image[:, :, 1] = 128  # Green channel
            rgb_image[:, :, 2] = 64   # Blue channel

            handler.show(rgb_image)

            # Check that imshow received BGR version
            call_args = mock_imshow.call_args
            bgr_image = call_args[0][1]

            # After RGB->BGR conversion:
            # RGB (255, 128, 64) becomes BGR (64, 128, 255)
            assert bgr_image[0, 0, 0] == 64   # Blue from original red
            assert bgr_image[0, 0, 1] == 128  # Green stays same
            assert bgr_image[0, 0, 2] == 255  # Red from original blue

    def test_show_receives_copy(self, handler, test_rgb_image):
        """Test that show receives a copy of the image, not original."""
        with mock.patch('cv2.imshow') as mock_imshow, \
             mock.patch('cv2.waitKey'):

            handler.show(test_rgb_image)

            # Get the image passed to imshow
            call_args = mock_imshow.call_args
            displayed_image = call_args[0][1]

            # It should be different object (converted from RGB to BGR)
            assert displayed_image is not test_rgb_image
