"""
Tests for MakeupProcessor class.
"""
import pytest
import numpy as np
import cv2
import os

from face_makeup.processor import MakeupProcessor, ProcessorConfig


@pytest.fixture
def processor():
    """MakeupProcessor instance with default config."""
    return MakeupProcessor()


@pytest.fixture
def black_image():
    """Black image with no face."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


class TestMakeupProcessor:
    """Tests for MakeupProcessor initialization and configuration."""

    def test_init_default_config(self):
        """Test processor initialization with default config."""
        processor = MakeupProcessor()
        assert processor.config is not None
        assert processor.config.min_detection_confidence == 0.5
        assert processor.config.max_num_faces == 10

    def test_init_custom_config(self):
        """Test processor initialization with custom config."""
        config = ProcessorConfig(
            min_detection_confidence=0.7,
            max_num_faces=5
        )
        processor = MakeupProcessor(config=config)
        assert processor.config.min_detection_confidence == 0.7
        assert processor.config.max_num_faces == 5


class TestProcessorOpen:
    """Tests for MakeupProcessor.open() method."""

    def test_open_valid_image(self, processor, test_image):
        """Test opening a valid image with face."""
        ctx = processor.open(test_image)

        assert ctx is not None
        assert ctx.face_count >= 1
        assert ctx.original_image.shape == test_image.shape

    def test_open_returns_context_with_faces(self, processor, test_image):
        """Test that open returns ImageContext with Face objects."""
        ctx = processor.open(test_image)

        assert len(ctx.faces) >= 1
        face = ctx.faces[0]
        assert face.face_id == 0
        assert face.bbox is not None
        assert face.landmarks is not None

    def test_open_no_face_raises_error(self, processor, black_image):
        """Test that open raises error for image without face."""
        with pytest.raises(ValueError, match="No face detected"):
            processor.open(black_image)

    def test_open_invalid_image_raises_error(self, processor):
        """Test that open raises error for invalid image."""
        with pytest.raises(ValueError):
            processor.open(None)

    def test_open_wrong_dtype_raises_error(self, processor):
        """Test that open raises error for wrong dtype."""
        image = np.zeros((256, 256, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            processor.open(image)

    def test_open_wrong_shape_raises_error(self, processor):
        """Test that open raises error for wrong shape."""
        image = np.zeros((256, 256), dtype=np.uint8)  # Grayscale
        with pytest.raises(ValueError):
            processor.open(image)


class TestProcessorOpenPath:
    """Tests for MakeupProcessor.open_path() method."""

    def test_open_path_valid_file(self, processor, test_image_path):
        """Test opening a valid image file."""
        ctx = processor.open_path(test_image_path)
        assert ctx is not None
        assert ctx.face_count >= 1

    def test_open_path_nonexistent_file(self, processor):
        """Test that open_path raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            processor.open_path("nonexistent_image.jpg")

    def test_open_path_directory(self, processor):
        """Test that open_path raises ValueError for directory path."""
        with pytest.raises(ValueError, match="Path is not a file"):
            processor.open_path("tests/test_images")

    def test_open_path_empty_string(self, processor):
        """Test that open_path raises error for empty string."""
        # Empty string resolves to current directory, which is not a file
        with pytest.raises((FileNotFoundError, ValueError)):
            processor.open_path("")

    def test_open_path_relative_path(self, processor, test_image_path):
        """Test opening with relative path."""
        ctx = processor.open_path(test_image_path)
        assert ctx is not None

    def test_open_path_absolute_path(self, processor, test_image_path):
        """Test opening with absolute path."""
        abs_path = os.path.abspath(test_image_path)
        ctx = processor.open_path(abs_path)
        assert ctx is not None


class TestProcessorReuse:
    """Tests for processor reusability."""

    def test_processor_reuse_multiple_images(self, processor, test_images_dir):
        """Test that processor can process multiple images."""
        image_files = sorted(test_images_dir.glob("*.png")) + sorted(test_images_dir.glob("*.jpg"))

        success_count = 0
        for img_path in image_files[:3]:  # Test first 3 images
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                ctx = processor.open(image_rgb)
                assert ctx.face_count >= 1
                success_count += 1
            except ValueError:
                # Skip images without faces
                continue

        assert success_count > 0, "No images processed successfully"
