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
    EyelinerPass,
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

    def test_generate_left(self):
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        strategy = EyelinerMaskStrategy("eyeliner_left")
        mask = strategy.generate(coords, (128, 128), image=None)
        assert mask.shape == (128, 128)
        # Mask strategy now returns the final soft mask (float32 0-1).
        assert 0.0 < mask.max() <= 1.0

    def test_generate_right(self):
        coords = _build_eyeliner_coords(EYELINER_RIGHT_INDICES)
        strategy = EyelinerMaskStrategy("eyeliner_right")
        mask = strategy.generate(coords, (128, 128), image=None)
        assert mask.shape == (128, 128)
        assert 0.0 < mask.max() <= 1.0

    def test_image_does_not_change_mask(self):
        """Correction happens before the pass (renderer); the mask only draws coords."""
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        strategy = EyelinerMaskStrategy("eyeliner_left")
        np.testing.assert_array_equal(
            strategy.generate(coords, img.shape[:2], image=img),
            strategy.generate(coords, img.shape[:2], image=None),
        )

    def test_draws_given_edge_points(self):
        """Moving an edge landmark row moves the drawn line with it."""
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        strategy = EyelinerMaskStrategy("eyeliner_left")
        moved = coords.copy()
        moved[EYELINER_LEFT_INDICES["upper"][3]] += (0.0, -20.0)
        assert not np.array_equal(
            strategy.generate(coords, (128, 128)),
            strategy.generate(moved, (128, 128)),
        )

    def test_keeps_subpixel_positions(self):
        """Float (corrected) coords are drawn with sub-pixel precision."""
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        strategy = EyelinerMaskStrategy("eyeliner_left")
        shifted = coords + np.float32(0.25)
        assert not np.array_equal(
            strategy.generate(coords, (128, 128)),
            strategy.generate(shifted, (128, 128)),
        )

    def test_uses_lash_line(self):
        assert EyelinerMaskStrategy.uses_lash_line is True

    def test_thickness_dilates_mask(self):
        """Larger thickness dilates the eyeliner geometry → more covered pixels.

        Thickness dilation now lives in the mask stage (before the fade) so the
        widened band is softened as one piece.
        """
        coords = _build_eyeliner_coords(EYELINER_LEFT_INDICES)
        thin = EyelinerMaskStrategy(
            "eyeliner_left", thickness=1
        ).generate(coords, (128, 128), image=None)
        thick = EyelinerMaskStrategy(
            "eyeliner_left", thickness=5
        ).generate(coords, (128, 128), image=None)
        assert np.count_nonzero(thick) > np.count_nonzero(thin)


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


# ── EyelinerPass ─────────────────────────────────────────────────────────────


class TestEyelinerPass:
    """Tests for EyelinerPass construction and wiring."""

    @pytest.fixture
    def params(self):
        return EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=0.9,
            softness=0.2,
            thickness=3,
        )

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_pass_type(self, params, side):
        p = EyelinerPass(params, side)
        assert p.pass_type == f"eyeliner_{side}"

    def test_priority(self, params):
        p = EyelinerPass(params, "left")
        assert p.priority == 30

    def test_has_mask_strategy(self, params):
        p = EyelinerPass(params, "left")
        assert isinstance(p.mask_strategy, EyelinerMaskStrategy)

    def test_has_blend_strategy(self, params):
        p = EyelinerPass(params, "left")
        assert isinstance(p.blend_strategy, EyelinerBlendStrategy)

    def test_mask_strategy_wires_params(self, params):
        """EyelinerPass must thread thickness, softness, AND the per-part
        feather/soften calibration into the strategy — previously only
        thickness and softness were passed."""
        p = EyelinerPass(params, "left")
        assert p.mask_strategy._thickness == 3
        assert p.mask_strategy._softness == 0.2
        assert p.mask_strategy._feather_scale == EyeLinerParams._FEATHER_SCALE
        assert p.mask_strategy._soften_floor == EyeLinerParams._SOFTEN_FLOOR
