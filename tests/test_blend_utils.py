"""
Tests for blend_utils.py: soften_mask, apply_target_color, standard_blend,
apply_hue_shift_blend.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.core.blend.blend_utils import (
    apply_hue_shift_blend,
    apply_target_color,
    soften_mask,
    standard_blend,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def solid_gray():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def solid_white():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def solid_black():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def center_mask():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    return mask


@pytest.fixture
def half_mask():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[:, :50] = 1.0
    return mask


@pytest.fixture
def zero_mask():
    return np.zeros((100, 100), dtype=np.float32)


# ── soften_mask ──────────────────────────────────────────────────────────────


class TestSoftenMask:
    """Tests for soften_mask()."""

    def test_zero_blur_returns_binary_float(self, center_mask):
        """blur_strength=0 returns float32 mask unchanged."""
        result = soften_mask(center_mask, 0.0)
        assert result.dtype == np.float32
        assert result.min() == 0.0
        assert result.max() == 1.0
        assert result[50, 50] == 1.0
        assert result[0, 0] == 0.0

    def test_blur_reduces_edge_values(self, center_mask):
        """blur_strength > 0 attenuates edge pixels."""
        result = soften_mask(center_mask, 0.5)
        edge_val = result[25, 25]
        center_val = result[50, 50]
        assert edge_val < center_val

    def test_uint8_mask_normalised(self):
        """uint8 mask (0/255) is normalised to float32 0-1."""
        uint8_mask = np.zeros((50, 50), dtype=np.uint8)
        uint8_mask[10:40, 10:40] = 255
        result = soften_mask(uint8_mask, 0.0)
        assert result.dtype == np.float32
        assert result.max() == 1.0
        assert result[25, 25] == 1.0

    def test_empty_binary_mask_early_return(self):
        """Mask ≤ 0.5 everywhere → binary.max()==0 → early return."""
        low_mask = np.full((50, 50), 0.3, dtype=np.float32)
        result = soften_mask(low_mask, 0.5)
        assert result.dtype == np.float32
        assert result.max() == pytest.approx(0.3, abs=1e-5)

    def test_all_zero_mask(self):
        """soften_mask on all-zero mask returns all-zero float32."""
        z = np.zeros((50, 50), dtype=np.float32)
        result = soften_mask(z, 0.5)
        assert result.max() == 0.0
        assert result.dtype == np.float32


# ── apply_target_color ───────────────────────────────────────────────────────


class TestApplyTargetColor:
    """Tests for apply_target_color()."""

    def test_chromatic_blend_changes_hue(self, solid_gray, center_mask):
        """Pure-red target on gray image should shift hue toward red."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (255, 0, 0), 1.0)
        center_pixel = result[50, 50]
        assert center_pixel[0] > center_pixel[1]
        assert center_pixel[0] > center_pixel[2]

    def test_achromatic_white_target_lightens(self, solid_gray, center_mask):
        """Achromatic white target (S≈0) lerps toward its own V, lightening."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (255, 255, 255), 1.0)
        center_val = result[50, 50].mean()
        assert center_val > rgb_f[50, 50].mean()

    def test_achromatic_black_target_darkens(self, solid_gray, center_mask):
        """Achromatic black target (S≈0) lerps toward its own V, darkening."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (0, 0, 0), 1.0)
        center_val = result[50, 50].mean()
        assert center_val < rgb_f[50, 50].mean()

    def test_achromatic_matching_gray_unchanged(self, solid_gray, center_mask):
        """Achromatic target matching source gray leaves it unchanged (not darkened)."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (128, 128, 128), 1.0)
        assert np.isclose(result[50, 50].mean(), rgb_f[50, 50].mean(), atol=1e-2)

    def test_zero_opacity_unchanged(self, solid_gray, center_mask):
        """opacity=0 returns input unchanged."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (255, 0, 0), 0.0)
        assert np.allclose(result, rgb_f)

    def test_masked_region_only_affected(self, solid_gray, center_mask):
        """Pixels outside mask unchanged."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (255, 0, 0), 1.0)
        assert np.allclose(result[0, 0], rgb_f[0, 0])

    def test_achromatic_target_triggers_vlerp_path(self, solid_gray, center_mask):
        """Target S < 0.05 → achromatic V-lerp path (darkens toward black target)."""
        rgb_f = solid_gray.astype(np.float32) / 255.0
        result = apply_target_color(rgb_f, center_mask, (0, 0, 0), 0.5)
        center_before = rgb_f[50, 50].mean()
        center_after = result[50, 50].mean()
        assert center_after < center_before

    def test_zero_mask(self):
        """Mask all zeros → output equals input."""
        rgb_f = np.full((50, 50, 3), 0.5, dtype=np.float32)
        z = np.zeros((50, 50), dtype=np.float32)
        result = apply_target_color(rgb_f, z, (255, 0, 0), 1.0)
        assert np.allclose(result, rgb_f)

    def test_full_opacity_full_mask(self):
        """opacity=1, mask all ones: every pixel affected."""
        rgb_f = np.full((50, 50, 3), 0.5, dtype=np.float32)
        full = np.ones((50, 50), dtype=np.float32)
        result = apply_target_color(rgb_f, full, (255, 0, 0), 1.0)
        assert not np.allclose(result, rgb_f)


# ── standard_blend ───────────────────────────────────────────────────────────


class TestStandardBlend:
    """Tests for standard_blend()."""

    def test_output_uint8(self, solid_gray, center_mask):
        result = standard_blend(solid_gray, center_mask, (255, 0, 0), 0.5, 0.0)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self, solid_gray, center_mask):
        result = standard_blend(solid_gray, center_mask, (255, 0, 0), 0.5, 0.0)
        assert result.shape == solid_gray.shape

    def test_opacity_zero_unchanged(self, solid_gray, center_mask):
        result = standard_blend(solid_gray, center_mask, (255, 0, 0), 0.0, 0.0)
        assert np.array_equal(result, solid_gray)

    def test_red_target_increases_r_channel(self, solid_gray, center_mask):
        before_r = solid_gray[50, 50, 0]
        result = standard_blend(solid_gray, center_mask, (255, 0, 0), 1.0, 0.0)
        after_r = result[50, 50, 0]
        assert after_r > before_r

    def test_black_target_darkens(self, solid_gray, center_mask):
        before = solid_gray[50, 50].mean()
        result = standard_blend(solid_gray, center_mask, (0, 0, 0), 1.0, 0.0)
        after = result[50, 50].mean()
        assert after < before

    def test_unmasked_region_unchanged(self, solid_gray, center_mask):
        result = standard_blend(solid_gray, center_mask, (255, 0, 0), 1.0, 0.0)
        assert np.array_equal(result[0:10, 0:10], solid_gray[0:10, 0:10])


# ── apply_hue_shift_blend ───────────────────────────────────────────────────


class TestApplyHueShiftBlend:
    """Tests for apply_hue_shift_blend()."""

    def test_output_uint8(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (100, 150, 255), 0.5, 0.0)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (100, 150, 255), 0.5, 0.0)
        assert result.shape == solid_gray.shape

    def test_opacity_zero_unchanged(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (255, 0, 0), 0.0, 0.0)
        assert np.array_equal(result, solid_gray)

    def test_achromatic_black_target_darkens(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (0, 0, 0), 1.0, 0.0)
        assert result[50, 50].mean() < solid_gray[50, 50].mean()

    def test_achromatic_white_target_lightens(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (255, 255, 255), 1.0, 0.0)
        assert result[50, 50].mean() > solid_gray[50, 50].mean()

    def test_achromatic_matching_gray_unchanged(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (128, 128, 128), 1.0, 0.0)
        assert abs(int(result[50, 50].mean()) - int(solid_gray[50, 50].mean())) <= 1

    def test_unmasked_region_unchanged(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (255, 0, 0), 1.0, 0.0)
        assert np.array_equal(result[0:10, 0:10], solid_gray[0:10, 0:10])

    def test_chromatic_via_small_mask(self, solid_gray):
        """Small mask with chromatic target exercises shifted path."""
        small_mask = np.zeros((100, 100), dtype=np.float32)
        small_mask[45:55, 45:55] = 1.0
        result = apply_hue_shift_blend(solid_gray, small_mask, (255, 0, 0), 0.5, 0.0)
        assert not np.array_equal(result, solid_gray)

    def test_zero_mask_unchanged(self, solid_gray):
        z = np.zeros((100, 100), dtype=np.float32)
        result = apply_hue_shift_blend(solid_gray, z, (255, 0, 0), 1.0, 0.0)
        assert np.array_equal(result, solid_gray)

    def test_achromatic_without_crash(self, solid_gray, center_mask):
        result = apply_hue_shift_blend(solid_gray, center_mask, (128, 128, 128), 0.5, 0.0)
        assert result.shape == solid_gray.shape


# ── Mask edge cases ─────────────────────────────────────────────────────────


class TestStandardBlendEdgeCases:
    """Edge cases for standard_blend with degenerate masks and images."""

    def test_all_zero_mask_unchanged(self, solid_gray):
        z = np.zeros((100, 100), dtype=np.float32)
        result = standard_blend(solid_gray, z, (255, 0, 0), 1.0, 0.0)
        assert np.array_equal(result, solid_gray)

    def test_uint8_mask(self, solid_gray):
        """uint8 (0/255) mask works with standard_blend."""
        uint8_mask = np.zeros((100, 100), dtype=np.uint8)
        uint8_mask[25:75, 25:75] = 255
        result = standard_blend(solid_gray, uint8_mask, (255, 0, 0), 0.5, 0.0)
        assert result.shape == solid_gray.shape

    def test_all_black_image(self):
        """All-zero image: standard_blend should work."""
        black = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.float32)
        result = standard_blend(black, mask, (255, 0, 0), 0.5, 0.0)
        assert result.shape == black.shape
        assert result.dtype == np.uint8

    def test_all_white_image(self):
        """All-255 image: standard_blend should work without overflow."""
        white = np.full((50, 50, 3), 255, dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.float32)
        result = standard_blend(white, mask, (255, 0, 0), 0.5, 0.0)
        assert result.shape == white.shape
        assert result.max() <= 255

    def test_single_pixel_image(self):
        """Single-pixel image: standard_blend must not crash."""
        tiny = np.full((1, 1, 3), 128, dtype=np.uint8)
        mask = np.ones((1, 1), dtype=np.float32)
        result = standard_blend(tiny, mask, (255, 0, 0), 0.5, 0.0)
        assert result.shape == tiny.shape
