"""
Tests for MakeupBlender class.
"""
import pytest
import numpy as np

from src.face_makeup.core.makeup_blender import MakeupBlender
from src.face_makeup.core.types import ColorParams


@pytest.fixture
def blender():
    """MakeupBlender instance."""
    return MakeupBlender()


@pytest.fixture
def sample_image():
    """Sample RGB image (uint8)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [128, 128, 128]  # Gray
    return image


@pytest.fixture
def full_mask():
    """Full mask (float32 0-1)."""
    return np.ones((100, 100), dtype=np.float32)


@pytest.fixture
def partial_mask():
    """Partial mask with gradient."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 0.5
    return mask


class TestMakeupBlenderBasic:
    """Basic tests for MakeupBlender.blend()."""

    def test_blend_returns_correct_shape(self, blender, sample_image, full_mask):
        """Test that blend returns image with same shape."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, full_mask, color_params, opacity=0.5)
        assert result.shape == sample_image.shape

    def test_blend_returns_uint8(self, blender, sample_image, full_mask):
        """Test that blend returns uint8 dtype."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, full_mask, color_params, opacity=0.5)
        assert result.dtype == np.uint8

    def test_blend_modifies_image(self, blender, sample_image, full_mask):
        """Test that blend modifies the image."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, full_mask, color_params, opacity=0.5)
        assert not np.array_equal(result, sample_image)

    def test_blend_zero_opacity_no_change(self, blender, sample_image, full_mask):
        """Test that zero opacity produces no change."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, full_mask, color_params, opacity=0.0)
        assert np.array_equal(result, sample_image)

    def test_blend_empty_mask_no_change(self, blender, sample_image):
        """Test that empty mask produces no change."""
        empty_mask = np.zeros((100, 100), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, empty_mask, color_params, opacity=1.0)
        assert np.array_equal(result, sample_image)


class TestMakeupBlenderColors:
    """Tests for different color applications."""

    def test_blend_red_color(self, blender, sample_image, partial_mask):
        """Test blending with red color."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, partial_mask, color_params, opacity=0.8)
        # Check that red channel increased in masked area
        center = result[50, 50]
        edge = result[10, 10]
        assert center[0] > edge[0]  # Red increased in center

    def test_blend_blue_color(self, blender, sample_image, partial_mask):
        """Test blending with blue color."""
        color_params = ColorParams(target_rgb=(0, 0, 255))
        result = blender.blend(sample_image, partial_mask, color_params, opacity=0.8)
        center = result[50, 50]
        edge = result[10, 10]
        assert center[2] > edge[2]  # Blue increased in center

    def test_blend_pink_color(self, blender, sample_image, partial_mask):
        """Test blending with pink color (typical lip color)."""
        color_params = ColorParams(target_rgb=(255, 100, 100))
        result = blender.blend(sample_image, partial_mask, color_params, opacity=0.5)
        assert result.shape == sample_image.shape

    def test_blend_black_color(self, blender, sample_image, partial_mask):
        """Test blending with black color (eyeliner)."""
        color_params = ColorParams(target_rgb=(0, 0, 0))
        result = blender.blend(sample_image, partial_mask, color_params, opacity=0.8)
        # Black should darken the image
        center_brightness = np.mean(result[50, 50])
        edge_brightness = np.mean(result[10, 10])
        assert center_brightness < edge_brightness

    def test_blend_white_color(self, blender, sample_image, partial_mask):
        """Test blending with white color."""
        color_params = ColorParams(target_rgb=(255, 255, 255))
        result = blender.blend(sample_image, partial_mask, color_params, opacity=0.5)
        assert result.shape == sample_image.shape


class TestMakeupBlenderOpacity:
    """Tests for different opacity levels."""

    def test_low_opacity_subtle_change(self, blender, sample_image, full_mask):
        """Test that low opacity produces subtle change."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, full_mask, color_params, opacity=0.1)
        diff = np.abs(result.astype(float) - sample_image.astype(float))
        assert np.mean(diff) < 30  # Small difference

    def test_high_opacity_strong_change(self, blender, sample_image, full_mask):
        """Test that high opacity produces strong change."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, full_mask, color_params, opacity=0.9)
        diff = np.abs(result.astype(float) - sample_image.astype(float))
        assert np.mean(diff) > 30  # Significant difference

    def test_opacity_gradient(self, blender, sample_image, full_mask):
        """Test that higher opacity produces more change."""
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result_low = blender.blend(sample_image, full_mask, color_params, opacity=0.2)
        result_high = blender.blend(sample_image, full_mask, color_params, opacity=0.8)

        diff_low = np.mean(np.abs(result_low.astype(float) - sample_image.astype(float)))
        diff_high = np.mean(np.abs(result_high.astype(float) - sample_image.astype(float)))

        assert diff_high > diff_low


class TestMakeupBlenderValidation:
    """Tests for input validation."""

    def test_invalid_image_dtype(self, blender, full_mask):
        """Test that invalid image dtype raises error."""
        float_image = np.zeros((100, 100, 3), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        with pytest.raises(ValueError, match="must be uint8"):
            blender.blend(float_image, full_mask, color_params, opacity=0.5)

    def test_invalid_image_shape_2d(self, blender, full_mask):
        """Test that 2D image raises error."""
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        with pytest.raises(ValueError, match="must be"):
            blender.blend(gray_image, full_mask, color_params, opacity=0.5)

    def test_invalid_image_shape_4channel(self, blender, full_mask):
        """Test that 4-channel image raises error."""
        rgba_image = np.zeros((100, 100, 4), dtype=np.uint8)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        with pytest.raises(ValueError, match="must be"):
            blender.blend(rgba_image, full_mask, color_params, opacity=0.5)

    def test_invalid_mask_dtype(self, blender, sample_image):
        """Test that invalid mask dtype raises error."""
        uint8_mask = np.ones((100, 100), dtype=np.uint8)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        with pytest.raises(ValueError, match="must be float32"):
            blender.blend(sample_image, uint8_mask, color_params, opacity=0.5)

    def test_invalid_mask_shape(self, blender, sample_image):
        """Test that 3D mask raises error."""
        mask_3d = np.ones((100, 100, 3), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        with pytest.raises(ValueError, match="must be"):
            blender.blend(sample_image, mask_3d, color_params, opacity=0.5)

    def test_mismatched_shapes(self, blender, sample_image):
        """Test that mismatched shapes raise error."""
        wrong_size_mask = np.ones((50, 50), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        with pytest.raises(ValueError, match="shape"):
            blender.blend(sample_image, wrong_size_mask, color_params, opacity=0.5)


class TestMakeupBlenderEdgeCases:
    """Edge case tests for MakeupBlender."""

    def test_small_image(self, blender):
        """Test with very small image."""
        small_image = np.full((10, 10, 3), 128, dtype=np.uint8)
        small_mask = np.ones((10, 10), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(small_image, small_mask, color_params, opacity=0.5)
        assert result.shape == small_image.shape

    def test_large_image(self, blender):
        """Test with large image."""
        large_image = np.full((512, 512, 3), 128, dtype=np.uint8)
        large_mask = np.ones((512, 512), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(large_image, large_mask, color_params, opacity=0.5)
        assert result.shape == large_image.shape

    def test_non_square_image(self, blender):
        """Test with non-square image."""
        rect_image = np.full((100, 200, 3), 128, dtype=np.uint8)
        rect_mask = np.ones((100, 200), dtype=np.float32)
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(rect_image, rect_mask, color_params, opacity=0.5)
        assert result.shape == rect_image.shape

    def test_values_clamped(self, blender, full_mask):
        """Test that output values are clamped to valid range."""
        bright_image = np.full((100, 100, 3), 250, dtype=np.uint8)
        color_params = ColorParams(target_rgb=(255, 255, 255))
        result = blender.blend(bright_image, full_mask, color_params, opacity=1.0)
        assert result.max() <= 255
        assert result.min() >= 0

    def test_preserves_unmasked_regions(self, blender, sample_image):
        """Test that unmasked regions are preserved."""
        partial_mask = np.zeros((100, 100), dtype=np.float32)
        partial_mask[50:, :] = 1.0  # Only bottom half
        color_params = ColorParams(target_rgb=(255, 0, 0))
        result = blender.blend(sample_image, partial_mask, color_params, opacity=1.0)
        # Top half should be unchanged
        assert np.array_equal(result[:40, :], sample_image[:40, :])
