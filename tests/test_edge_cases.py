"""
Edge case tests for face makeup library.
"""
import pytest
import numpy as np

from face_makeup.processor import MakeupProcessor, ProcessorConfig
from face_makeup.core.types import ColorParams, LipsParams


@pytest.fixture
def processor():
    """MakeupProcessor instance."""
    return MakeupProcessor()


class TestImageEdgeCases:
    """Edge case tests for image input."""

    def test_very_dark_image(self, processor, test_image):
        """Test with very dark image (may fail face detection)."""
        # Make image very dark
        dark_image = (test_image * 0.1).astype(np.uint8)

        # May or may not detect face depending on image
        try:
            ctx = processor.open(dark_image)
            # If face detected, should still work
            if ctx.face_count > 0:
                ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
                result = ctx.render()
                assert result.shape == dark_image.shape
        except ValueError:
            pass  # No face detected is acceptable

    def test_very_bright_image(self, processor, test_image):
        """Test with very bright image."""
        # Make image very bright
        bright_image = np.clip(test_image * 1.8, 0, 255).astype(np.uint8)

        try:
            ctx = processor.open(bright_image)
            if ctx.face_count > 0:
                ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
                result = ctx.render()
                assert result.shape == bright_image.shape
        except ValueError:
            pass

    def test_high_contrast_image(self, processor, test_image):
        """Test with high contrast image."""
        # Increase contrast
        mean = np.mean(test_image)
        contrast_image = np.clip((test_image - mean) * 2 + mean, 0, 255).astype(np.uint8)

        try:
            ctx = processor.open(contrast_image)
            if ctx.face_count > 0:
                ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
                result = ctx.render()
                assert result.shape == contrast_image.shape
        except ValueError:
            pass


class TestColorEdgeCases:
    """Edge case tests for color values."""

    def test_boundary_color_black(self, processor, test_image):
        """Test with black color (0, 0, 0)."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(0, 0, 0), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_boundary_color_white(self, processor, test_image):
        """Test with white color (255, 255, 255)."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 255, 255), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_pure_red(self, processor, test_image):
        """Test with pure red color."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 0, 0), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_pure_green(self, processor, test_image):
        """Test with pure green color."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(0, 255, 0), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_pure_blue(self, processor, test_image):
        """Test with pure blue color."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(0, 0, 255), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape


class TestOpacityEdgeCases:
    """Edge case tests for opacity values."""

    def test_zero_opacity(self, processor, test_image):
        """Test with zero opacity."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.0)
        result = ctx.render()
        # Zero opacity should not change the image (or minimal change)
        assert result.shape == test_image.shape

    def test_full_opacity(self, processor, test_image):
        """Test with full opacity."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=1.0)
        result = ctx.render()
        assert result.shape == test_image.shape
        assert not np.array_equal(result, test_image)

    def test_very_low_opacity(self, processor, test_image):
        """Test with very low opacity."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.01)
        result = ctx.render()
        assert result.shape == test_image.shape


class TestMakeupLayerEdgeCases:
    """Edge case tests for makeup layers."""

    def test_multiple_same_layers(self, processor, test_image):
        """Test applying same makeup multiple times."""
        ctx = processor.open(test_image)
        for _ in range(5):
            ctx.faces[0].lip(color=(255, 100, 100), opacity=0.2)
        result = ctx.render()
        assert result.shape == test_image.shape
        assert len(ctx.faces[0].makeup_layers) == 5

    def test_conflicting_colors(self, processor, test_image):
        """Test applying conflicting colors to same area."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 0, 0), opacity=0.5)  # Red
        ctx.faces[0].lip(color=(0, 255, 0), opacity=0.5)  # Green
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_clear_and_verify_empty(self, processor, test_image):
        """Test that clear removes all layers."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        ctx.faces[0].cheek(color=(255, 180, 180), opacity=0.3)
        ctx.faces[0].eyelid(color=(180, 130, 150), opacity=0.3)

        # lip=1, cheek(both)=2, eyelid(both)=2 => total 5 layers
        assert len(ctx.faces[0].makeup_layers) == 5

        ctx.faces[0].clear()
        assert len(ctx.faces[0].makeup_layers) == 0

    def test_side_specific_layers(self, processor, test_image):
        """Test left and right side specific layers."""
        ctx = processor.open(test_image)
        ctx.faces[0].cheek(color=(255, 0, 0), opacity=0.3, side='left')
        ctx.faces[0].cheek(color=(0, 255, 0), opacity=0.3, side='right')
        result = ctx.render()
        assert result.shape == test_image.shape
        assert len(ctx.faces[0].makeup_layers) == 2


class TestCacheEdgeCases:
    """Edge case tests for render caching."""

    def test_cache_after_clear(self, processor, test_image):
        """Test cache invalidation after clear."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        result1 = ctx.render()

        ctx.faces[0].clear()
        result2 = ctx.render()

        # After clear, should be back to original
        assert np.array_equal(result2, test_image)

    def test_multiple_cache_invalidations(self, processor, test_image):
        """Test multiple cache invalidations."""
        ctx = processor.open(test_image)

        results = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for color in colors:
            ctx.faces[0].clear()
            ctx.faces[0].lip(color=color, opacity=0.5)
            results.append(ctx.render().copy())

        # All results should be different
        assert not np.array_equal(results[0], results[1])
        assert not np.array_equal(results[1], results[2])
        assert not np.array_equal(results[0], results[2])


class TestKwargEdgeCases:
    """Edge case tests for keyword arguments."""

    def test_kwargs_with_defaults(self, processor, test_image):
        """Test keyword arguments with default values."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_multiple_kwargs_calls(self, processor, test_image):
        """Test multiple makeup calls with keyword arguments."""
        ctx = processor.open(test_image)

        # Apply different makeup types
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        ctx.faces[0].cheek(color=(255, 180, 180), opacity=0.3)

        result = ctx.render()
        assert result.shape == test_image.shape


class TestProcessorConfigEdgeCases:
    """Edge case tests for processor configuration."""

    def test_low_confidence_threshold(self, test_image):
        """Test with very low confidence threshold."""
        config = ProcessorConfig(min_detection_confidence=0.1)
        processor = MakeupProcessor(config=config)

        ctx = processor.open(test_image)
        assert ctx is not None

    def test_high_confidence_threshold(self, test_image):
        """Test with high confidence threshold."""
        config = ProcessorConfig(min_detection_confidence=0.9)
        processor = MakeupProcessor(config=config)

        try:
            ctx = processor.open(test_image)
            # High threshold may reject some faces
        except ValueError:
            pass  # Acceptable if no face meets threshold

    def test_single_face_limit(self, test_image):
        """Test with max_num_faces=1."""
        config = ProcessorConfig(max_num_faces=1)
        processor = MakeupProcessor(config=config)

        ctx = processor.open(test_image)
        assert ctx.face_count <= 1


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_continue_after_invalid_color(self, processor, test_image):
        """Test that processor works after catching invalid color error."""
        ctx = processor.open(test_image)

        # Try invalid color
        try:
            ctx.faces[0].lip(color=(300, 0, 0), opacity=0.5)
        except ValueError:
            pass

        # Should still work with valid color
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape

    def test_continue_after_invalid_opacity(self, processor, test_image):
        """Test that processor works after catching invalid opacity error."""
        ctx = processor.open(test_image)

        # Try invalid opacity
        try:
            ctx.faces[0].lip(color=(255, 100, 100), opacity=2.0)
        except ValueError:
            pass

        # Should still work with valid opacity
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        result = ctx.render()
        assert result.shape == test_image.shape
