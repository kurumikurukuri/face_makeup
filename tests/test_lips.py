"""
Tests for lips pass: LipMaskStrategy and LipBlendStrategy.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.constants import LOWER_LIP_INDICES, UPPER_LIP_INDICES
from face_makeup.core.passes.lips import LipBlendStrategy, LipMaskStrategy
from face_makeup.core.types import ColorParams, LipsParams


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def solid_gray():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def center_mask():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    return mask


def _build_lips_coords() -> np.ndarray:
    """Build a coords array covering all upper and lower lip indices."""
    indices_list = list(UPPER_LIP_INDICES) + list(LOWER_LIP_INDICES)
    max_idx = max(indices_list)
    coords = np.zeros((max_idx + 1, 2), dtype=np.float32)
    for i, idx in enumerate(sorted(set(indices_list))):
        # Arrange points in an ellipse-like shape centered at (50, 50)
        angle = 2 * np.pi * i / len(set(indices_list))
        coords[idx] = [50.0 + 20.0 * np.cos(angle), 50.0 + 10.0 * np.sin(angle)]
    return coords


# ── LipMaskStrategy ──────────────────────────────────────────────────────────


class TestLipMaskStrategy:
    """Tests for LipMaskStrategy mask generation."""

    def test_generate_returns_soft_mask(self):
        """Valid upper and lower lip points produce a non-empty soft mask.

        The mask strategy now bakes the fade in and returns float32 0-1, so
        the pipeline can apply exclusions to the softened mask before blending.
        """
        coords = _build_lips_coords()
        strategy = LipMaskStrategy(softness=0.5)
        mask = strategy.generate(coords, (100, 100))
        assert mask.dtype == np.float32
        assert mask.shape == (100, 100)
        assert 0.0 < mask.max() <= 1.0

    def test_generate_with_all_zero_coords(self):
        """All landmark coords at (0,0) still produces valid (but degenerate) mask."""
        max_idx = max(max(UPPER_LIP_INDICES), max(LOWER_LIP_INDICES))
        coords = np.zeros((max_idx + 1, 2), dtype=np.float32)
        strategy = LipMaskStrategy(softness=0.5)
        mask = strategy.generate(coords, (100, 100))
        # fillPoly with all coords at origin may produce small artifacts
        # near (0,0) with sub-pixel fixed-point rendering, but won't
        # crash and returns correct shape/dtype.
        assert mask.dtype == np.float32
        assert mask.shape == (100, 100)


# ── LipBlendStrategy ─────────────────────────────────────────────────────────


class TestLipBlendStrategy:
    """Tests for LipBlendStrategy blend application."""

    def test_apply_returns_same_shape_and_dtype(self, solid_gray, center_mask):
        """Output has same shape and dtype as input."""
        strategy = LipBlendStrategy()
        params = LipsParams(
            color_params=ColorParams(target_rgb=(200, 50, 80)),
            opacity=0.8,
            softness=0.3,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape
        assert result.dtype == np.uint8

    def test_apply_zero_mask_no_change(self, solid_gray):
        """All-zero mask → input unchanged."""
        strategy = LipBlendStrategy()
        params = LipsParams(
            color_params=ColorParams(target_rgb=(200, 50, 80)),
            opacity=0.8,
            softness=0.3,
        )
        zero_mask = np.zeros((100, 100), dtype=np.float32)
        result = strategy.apply(solid_gray, zero_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_apply_nonzero_mask_changes_image(self, solid_gray, center_mask):
        """Non-zero mask → output differs from input."""
        strategy = LipBlendStrategy()
        params = LipsParams(
            color_params=ColorParams(target_rgb=(200, 50, 80)),
            opacity=1.0,
            softness=0.0,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert not np.array_equal(result, solid_gray)

    def test_apply_opacity_zero_unchanged(self, solid_gray, center_mask):
        """opacity=0 → input unchanged."""
        strategy = LipBlendStrategy()
        params = LipsParams(
            color_params=ColorParams(target_rgb=(200, 50, 80)),
            opacity=0.0,
            softness=0.3,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_apply_outside_mask_unchanged(self, solid_gray, center_mask):
        """Pixels outside mask region unchanged."""
        strategy = LipBlendStrategy()
        params = LipsParams(
            color_params=ColorParams(target_rgb=(200, 50, 80)),
            opacity=1.0,
            softness=0.0,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result[0:10, 0:10], solid_gray[0:10, 0:10])
