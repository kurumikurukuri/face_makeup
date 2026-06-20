"""
Tests for MakeupBlender class.
"""
import pytest
import numpy as np

from face_makeup.core.makeup_blender import MakeupBlender
from face_makeup.core.types import ColorParams, EyeIrisParams


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

    def test_uint8_mask_accepted(self, blender, sample_image):
        """Test that uint8 mask is accepted and converted to float32."""
        uint8_mask = np.ones((100, 100), dtype=np.uint8) * 255
        color_params = ColorParams(target_rgb=(255, 0, 0))
        # Should not raise - uint8 is now accepted and converted internally
        result = blender.blend(sample_image, uint8_mask, color_params, opacity=0.5)
        assert result.shape == sample_image.shape

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


class TestSoftenMask:
    """Tests for _soften_mask distance-based attenuation."""

    def test_returns_float32(self, blender):
        """Output is always float32."""
        mask = np.ones((50, 50), dtype=np.float32)
        result = blender._soften_mask(mask, blur_strength=0.5)
        assert result.dtype == np.float32

    def test_uint8_input_converted(self, blender):
        """uint8 0/255 mask is converted to float32 0.0-1.0."""
        mask = np.full((50, 50), 255, dtype=np.uint8)
        result = blender._soften_mask(mask, blur_strength=0.0)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0)

    def test_uint8_zero_mask(self, blender):
        """uint8 zero mask stays zero."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = blender._soften_mask(mask, blur_strength=0.5)
        np.testing.assert_allclose(result, 0.0)

    def test_zero_blur_strength_no_attenuation(self, blender):
        """blur_strength <= 0 returns the mask as-is (no attenuation)."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0
        result = blender._soften_mask(mask, blur_strength=0.0)
        np.testing.assert_array_equal(result, mask)

    def test_negative_blur_strength_no_attenuation(self, blender):
        """Negative blur_strength treated same as zero (early exit)."""
        mask = np.ones((50, 50), dtype=np.float32)
        result = blender._soften_mask(mask, blur_strength=-0.5)
        np.testing.assert_array_equal(result, mask)

    def test_empty_mask_returns_zeros(self, blender):
        """All-zero mask is returned unchanged regardless of blur_strength."""
        mask = np.zeros((50, 50), dtype=np.float32)
        result = blender._soften_mask(mask, blur_strength=0.5)
        np.testing.assert_allclose(result, 0.0)

    def test_center_pixel_stays_at_one(self, blender):
        """Center of a large filled mask stays at 1.0."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[10:90, 10:90] = 1.0
        result = blender._soften_mask(mask, blur_strength=0.3)
        assert result[50, 50] == pytest.approx(1.0, abs=1e-5)

    def test_boundary_pixel_equals_blur_strength(self, blender):
        """Boundary pixels approach blur_strength value."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        blur_strength = 0.4
        result = blender._soften_mask(mask, blur_strength=blur_strength)
        # Pixels at the edge of the mask (distance=1 from boundary)
        # should be close to blur_strength
        # Check a pixel just inside the boundary
        boundary_val = result[20, 50]
        assert boundary_val < 0.6  # Well below center value
        assert boundary_val >= blur_strength - 0.1  # Close to blur_strength

    def test_attenuation_monotonic_from_edge_to_center(self, blender):
        """Values increase monotonically from boundary toward center."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[10:90, 10:90] = 1.0
        result = blender._soften_mask(mask, blur_strength=0.2)
        # Sample along a horizontal line from edge to center
        row = 50
        edge = result[row, 11]   # just inside mask
        mid = result[row, 30]    # midway
        center = result[row, 50] # center
        assert edge <= mid <= center

    def test_higher_blur_strength_less_falloff(self, blender):
        """Higher blur_strength produces less falloff at edges."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        result_low = blender._soften_mask(mask, blur_strength=0.1)
        result_high = blender._soften_mask(mask, blur_strength=0.8)
        # Edge pixel with high blur_strength should be brighter
        edge_low = result_low[20, 50]
        edge_high = result_high[20, 50]
        assert edge_high > edge_low

    def test_blur_strength_one_no_falloff(self, blender):
        """blur_strength=1.0 means no falloff (attenuation is always 1)."""
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[10:70, 10:70] = 1.0
        result = blender._soften_mask(mask, blur_strength=1.0)
        # All masked pixels should remain 1.0
        np.testing.assert_allclose(result[10:70, 10:70], 1.0, atol=1e-5)
        # Unmasked pixels should remain 0.0
        np.testing.assert_allclose(result[0:10, :], 0.0)

    def test_unmasked_region_stays_zero(self, blender):
        """Pixels outside the mask remain 0 regardless of blur_strength."""
        mask = np.zeros((80, 80), dtype=np.float32)
        mask[30:50, 30:50] = 1.0
        result = blender._soften_mask(mask, blur_strength=0.5)
        assert result[0, 0] == 0.0
        assert result[79, 79] == 0.0
        np.testing.assert_allclose(result[0:30, :], 0.0)

    def test_output_shape_matches_input(self, blender):
        """Output shape matches input mask shape."""
        for shape in [(50, 50), (100, 200), (200, 100)]:
            mask = np.ones(shape, dtype=np.float32)
            result = blender._soften_mask(mask, blur_strength=0.5)
            assert result.shape == shape

    def test_single_pixel_mask(self, blender):
        """Single pixel mask: distanceTransform normalizes to 1.0."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[25, 25] = 1.0
        result = blender._soften_mask(mask, blur_strength=0.3)
        # Single pixel is the farthest point from any boundary,
        # so dist_norm=1.0 and attenuation=1.0
        assert result[25, 25] == pytest.approx(1.0, abs=1e-5)
        assert result[0, 0] == 0.0


class TestBlendEyeiris:
    """Tests for blend_eyeiris (covers uint8/float32 mask handling and eye_mask clipping)."""

    @pytest.fixture
    def iris_params(self):
        return EyeIrisParams(color_params=ColorParams(target_rgb=(0, 100, 200)), opacity=0.8)

    def test_uint8_mask_accepted(self, blender, sample_image, iris_params):
        """uint8 mask is accepted and converted internally."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        eye_mask = np.ones((100, 100), dtype=np.uint8) * 255

        result = blender.blend_eyeiris(sample_image, mask, eye_mask, iris_params)

        assert result.dtype == np.uint8
        assert result.shape == sample_image.shape

    def test_float32_mask_accepted(self, blender, sample_image, iris_params):
        """float32 mask is accepted without conversion."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 1.0
        eye_mask = np.ones((100, 100), dtype=np.float32)

        result = blender.blend_eyeiris(sample_image, mask, eye_mask, iris_params)

        assert result.dtype == np.uint8
        assert result.shape == sample_image.shape

    def test_color_applied_in_masked_region(self, blender, iris_params):
        """Color is applied inside mask; unmasked region is unchanged."""
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 1.0
        eye_mask = np.ones((100, 100), dtype=np.float32)

        result = blender.blend_eyeiris(image, mask, eye_mask, iris_params)

        # Outside mask: pixel unchanged
        np.testing.assert_array_equal(result[0, 0], image[0, 0])
        # Inside mask: pixel differs from original gray
        assert not np.array_equal(result[50, 50], image[50, 50])

    def test_eye_mask_clips_iris_mask(self, blender, iris_params):
        """Iris color should not bleed outside eye_mask even if eyeiris_mask covers it."""
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Iris mask (ellipse) overlaps eyelid skin outside the eye opening
        eyeiris_mask = np.zeros((100, 100), dtype=np.float32)
        eyeiris_mask[30:70, 30:70] = 1.0
        # Eye opening only covers the lower half of that region
        eye_mask = np.zeros((100, 100), dtype=np.float32)
        eye_mask[50:70, 30:70] = 1.0

        result = blender.blend_eyeiris(image, eyeiris_mask, eye_mask, iris_params)

        # Inside iris mask but outside eye_mask: unaffected (clipped)
        np.testing.assert_array_equal(result[35, 50], image[35, 50])
        # Inside both masks: color applied
        assert not np.array_equal(result[60, 50], image[60, 50])

    def test_none_eye_mask_raises(self, blender, sample_image, iris_params):
        """eye_mask is required; None should raise ValueError."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 1.0

        with pytest.raises(ValueError, match="eye_mask"):
            blender.blend_eyeiris(sample_image, mask, None, iris_params)
