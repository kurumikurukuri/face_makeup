"""
Tests for eyeliner pass: EyelinerMaskStrategy and EyelinerBlendStrategy.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.constants import (
    EYELINER_LEFT_INDICES,
    EYELINER_RIGHT_INDICES,
)
from face_makeup.core.passes.eyeliner import (
    EyelinerBlendStrategy,
    EyelinerMaskStrategy,
)
from face_makeup.core.types import (
    ColorParams,
    EyeLinerParams,
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


def _build_eyeliner_coords(indices_dict: dict) -> np.ndarray:
    """Build a coords array covering all landmark indices referenced by *indices_dict*.

    Each index receives a coordinate inside (64, 64) on a 128×128 canvas so the
    generated mask stays non-empty.
    """
    all_indices = set()
    for idxs in indices_dict.values():
        all_indices.update(idxs)
    max_idx = max(all_indices)
    coords = np.zeros((max_idx + 1, 2), dtype=np.float32)
    for i, idx in enumerate(sorted(all_indices)):
        coords[idx] = [64.0 + i % 32, 64.0 + i // 32]
    return coords


# ── EyelinerMaskStrategy ─────────────────────────────────────────────────────


class TestEyelinerMaskStrategy:
    """Tests for EyelinerMaskStrategy mask generation."""

    def test_generate_simple_path_left(self):
        """generate with image=None forces _generate_simple path (left eye)."""
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        strategy = EyelinerMaskStrategy("eyeliner_left")
        mask = strategy.generate(coords, (128, 128), image=None)
        assert mask.shape == (128, 128)
        assert mask.max() == 255

    def test_generate_simple_path_right(self):
        """generate with image=None forces _generate_simple path (right eye)."""
        coords = _build_eyeliner_coords(EYELINER_RIGHT_INDICES)
        strategy = EyelinerMaskStrategy("eyeliner_right")
        mask = strategy.generate(coords, (128, 128), image=None)
        assert mask.shape == (128, 128)
        assert mask.max() == 255

    def test_generate_corrected_path(self):
        """generate with image forces _generate_corrected path."""
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        strategy = EyelinerMaskStrategy("eyeliner_left")
        mask = strategy.generate(coords, img.shape[:2], image=img)
        assert mask.shape == img.shape[:2]
        assert mask.max() == 255


# ── EyelinerBlendStrategy ────────────────────────────────────────────────────


class TestEyelinerBlendStrategy:
    """Tests for EyelinerBlendStrategy blend application."""

    def test_thickness_1_no_dilation(self, solid_gray, center_mask):
        """thickness=1: no dilation — mask unchanged."""
        strategy = EyelinerBlendStrategy()
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=1,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape

    def test_thickness_5_dilates(self, solid_gray, center_mask):
        """thickness=5 dilates the mask → more pixels affected."""
        strategy = EyelinerBlendStrategy()
        params_thin = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=1,
        )
        result_thin = strategy.apply(solid_gray, center_mask.copy(), params_thin)
        params_thick = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=5,
        )
        result_thick = strategy.apply(solid_gray, center_mask.copy(), params_thick)
        diff_thin = np.abs(result_thin.astype(float) - solid_gray.astype(float)).sum()
        diff_thick = np.abs(result_thick.astype(float) - solid_gray.astype(float)).sum()
        assert diff_thick > diff_thin

    def test_zero_opacity_unchanged(self, solid_gray, center_mask):
        """opacity=0 returns image unchanged."""
        strategy = EyelinerBlendStrategy()
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=0.0,
            thickness=3,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_large_thickness_no_crash(self, solid_gray, center_mask):
        """Very large thickness must not crash."""
        strategy = EyelinerBlendStrategy()
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=0.5,
            thickness=50,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape
