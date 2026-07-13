"""
Tests for eye iris pass: EyeIrisMaskStrategy and EyeIrisBlendStrategy.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.constants import (
    EYE_IRIS_LEFT_INDICES,
    LEFT_EYE_INDICES,
)
from face_makeup.core.passes.eyeiris import (
    EyeIrisBlendStrategy,
    EyeIrisMaskStrategy,
)
from face_makeup.core.types import (
    ColorParams,
    EyeIrisParams,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def solid_gray():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def center_mask():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    return mask


def _build_iris_coords() -> np.ndarray:
    """Build coords array with iris landmark points in a circle + eye contour."""
    all_indices = set(EYE_IRIS_LEFT_INDICES) | set(LEFT_EYE_INDICES)
    max_idx = max(all_indices)
    coords = np.zeros((max_idx + 1, 2), dtype=np.float32)

    # Iris points: small circle around (50, 50)
    for i, idx in enumerate(EYE_IRIS_LEFT_INDICES):
        angle = 2 * np.pi * i / len(EYE_IRIS_LEFT_INDICES)
        coords[idx] = [50.0 + 8.0 * np.cos(angle), 50.0 + 8.0 * np.sin(angle)]

    # Eye contour: larger polygon around (50, 50)
    for i, idx in enumerate(LEFT_EYE_INDICES):
        angle = 2 * np.pi * i / len(LEFT_EYE_INDICES)
        coords[idx] = [50.0 + 20.0 * np.cos(angle), 50.0 + 12.0 * np.sin(angle)]

    return coords


# ── EyeIrisMaskStrategy ──────────────────────────────────────────────────────


class TestEyeIrisMaskStrategy:
    """Tests for EyeIrisMaskStrategy mask generation."""

    def test_generate_fewer_than_five_points(self):
        """Fewer than 5 iris points → early return zero mask (fitEllipse requires ≥5)."""
        # Create a strategy with only 4 indices — len(points) < 5 triggers early return
        strategy = EyeIrisMaskStrategy(
            list(EYE_IRIS_LEFT_INDICES[:4]),
        )
        coords = _build_iris_coords()
        mask = strategy.generate(coords, (100, 100))
        assert mask.max() == 0

    def test_generate_with_valid_points(self):
        """5 iris points → ellipse mask (clip is handled by renderer)."""
        coords = _build_iris_coords()
        strategy = EyeIrisMaskStrategy(
            list(EYE_IRIS_LEFT_INDICES),
        )
        mask = strategy.generate(coords, (100, 100))
        # Mask strategy now bakes the fade in and returns float32 0-1.
        assert mask.dtype == np.float32
        assert mask.shape == (100, 100)
        assert 0.0 < mask.max() <= 1.0

    def test_generate_pure_ellipse(self):
        """Without clip_landmarks in the pass, the mask strategy returns a
        pure (unclipped) iris ellipse — clipping is the renderer's job."""
        coords = _build_iris_coords()
        strategy = EyeIrisMaskStrategy(list(EYE_IRIS_LEFT_INDICES))
        mask = strategy.generate(coords, (100, 100))
        assert mask.dtype == np.float32
        assert mask.shape == (100, 100)
        assert 0.0 < mask.max() <= 1.0


# ── EyeIrisBlendStrategy ─────────────────────────────────────────────────────


class TestEyeIrisBlendStrategy:
    """Tests for EyeIrisBlendStrategy blend application."""

    def test_apply_hue_shift_true(self, solid_gray, center_mask):
        """hue_shift=True uses hue-shift blend path."""
        strategy = EyeIrisBlendStrategy()
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(100, 150, 255)),
            opacity=0.8,
            softness=0.0,
            hue_shift=True,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape
        assert result.dtype == np.uint8

    def test_apply_hue_shift_false(self, solid_gray, center_mask):
        """hue_shift=False uses standard_blend path."""
        strategy = EyeIrisBlendStrategy()
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(100, 150, 255)),
            opacity=0.8,
            softness=0.0,
            hue_shift=False,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape
        assert result.dtype == np.uint8

    def test_apply_opacity_zero_unchanged(self, solid_gray, center_mask):
        """opacity=0 → input unchanged."""
        strategy = EyeIrisBlendStrategy()
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(100, 150, 255)),
            opacity=0.0,
            softness=0.0,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_apply_outside_mask_unchanged(self, solid_gray, center_mask):
        """Pixels outside mask region unchanged."""
        strategy = EyeIrisBlendStrategy()
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(100, 150, 255)),
            opacity=1.0,
            softness=0.0,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result[0:10, 0:10], solid_gray[0:10, 0:10])

    def test_apply_hue_shift_with_achromatic_target(self, solid_gray, center_mask):
        """Achromatic target with hue_shift=True triggers darken fallback."""
        strategy = EyeIrisBlendStrategy()
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(128, 128, 128)),
            opacity=0.5,
            softness=0.0,
            hue_shift=True,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape
        assert result.dtype == np.uint8
