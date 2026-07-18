"""
Unit tests for the tear bag (涙袋) highlight + shadow makeup pass.

Covers the lerp geometry of ``_lerp_outer`` (including its envelope-length
contract), the screen-blend lightening of the highlight, and the thickness
dilation of the shadow mask.

``_lerp_outer`` is exercised directly because its envelope-length ValueError
cannot be reached through the public Strategy classes — they bind the
fixed-length ``TEAR_BAG_ENVELOPE`` constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.core.passes.tearbag import (
    TearBagHighlightBlendStrategy,
    TearBagHighlightMaskStrategy,
    TearBagHighlightPass,
    TearBagShadowBlendStrategy,
    TearBagShadowMaskStrategy,
    TearBagShadowPass,
    _lerp_outer,
)
from face_makeup.core.types import (
    ColorParams,
    TearBagHighlightParams,
    TearBagShadowParams,
)


# ── shared geometry ───────────────────────────────────────────────────────────


def _flat_line_coords() -> np.ndarray:
    """7 ridge points on y=10 (x=0..6) + 7 lash points on y=20 (x=6..0).

    Ridge occupies coords indices 0..6; lash occupies 7..13 and is stored
    outer→inner (x descending) to mirror the real landmark order, so the
    internal ``lash_pts[::-1]`` reversal aligns it inner→outer with the ridge.
    """
    ridge = np.array([[i, 10.0] for i in range(7)], dtype=np.float32)
    lash = np.array([[6 - i, 20.0] for i in range(7)], dtype=np.float32)
    return np.vstack([ridge, lash])


_RIDGE = [0, 1, 2, 3, 4, 5, 6]
_LASH = [7, 8, 9, 10, 11, 12, 13]


# ── _lerp_outer ───────────────────────────────────────────────────────────────


class TestLerpOuter:
    def test_envelope_length_mismatch_raises_value_error(self):
        """envelope length != ridge point count must raise ValueError."""
        coords = _flat_line_coords()
        with pytest.raises(ValueError, match="envelope length"):
            _lerp_outer(coords, _RIDGE, _LASH, size=1.0, envelope=(0.0, 1.0))

    def test_size_zero_collapses_outer_onto_lash_line(self):
        """size=0 collapses the outer edge onto the (reversed) lash line."""
        coords = _flat_line_coords()
        outer, _lash = _lerp_outer(coords, _RIDGE, _LASH, size=0.0)
        assert np.allclose(outer[:, 1], 20.0)
        assert np.allclose(outer[:, 0], np.arange(7))

    def test_size_one_full_envelope_keeps_ridge(self):
        """size=1 with an all-ones envelope keeps the full ridge."""
        coords = _flat_line_coords()
        outer, _lash = _lerp_outer(
            coords, _RIDGE, _LASH, size=1.0, envelope=(1.0,) * 7
        )
        assert np.allclose(outer, coords[_RIDGE])

    def test_default_envelope_keeps_outer_between_lash_and_ridge(self):
        """With the real tapered envelope, outer y stays strictly between lines."""
        coords = _flat_line_coords()
        outer, _lash = _lerp_outer(coords, _RIDGE, _LASH, size=1.0)
        assert np.all(outer[:, 1] > 10.0)
        assert np.all(outer[:, 1] < 20.0)


# ── highlight screen blend ────────────────────────────────────────────────────


class TestHighlightScreenBlend:
    def test_white_target_always_lightens(self):
        """Screen blend toward white must brighten a black region, never darken."""
        strategy = TearBagHighlightBlendStrategy()
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.float32)
        params = TearBagHighlightParams(
            color_params=ColorParams(target_rgb=(255, 255, 255)),
            opacity=1.0,
        )
        result = strategy.apply(image, mask, params)
        assert result.dtype == np.uint8
        assert result.shape == image.shape
        assert result.mean() > image.mean()
        assert result.max() > 0

    def test_returns_distinct_array(self):
        """Blend must return a fresh ndarray, not the input by reference."""
        strategy = TearBagHighlightBlendStrategy()
        image = np.full((30, 30, 3), 50, dtype=np.uint8)
        mask = np.ones((30, 30), dtype=np.float32)
        params = TearBagHighlightParams(
            color_params=ColorParams(target_rgb=(255, 255, 255)),
            opacity=1.0,
        )
        result = strategy.apply(image, mask, params)
        assert result is not image


# ── highlight mask geometry ───────────────────────────────────────────────────


class TestHighlightMaskGenerate:
    def test_fills_area_between_ridge_and_lash_lines(self):
        """A point strictly between the ridge (y=10) and lash (y=20) lines is
        filled; a point above the ridge line is not."""
        coords = _flat_line_coords()
        strategy = TearBagHighlightMaskStrategy(
            _RIDGE, _LASH, size=1.0, envelope=(1.0,) * 7, softness=0.0,
        )
        mask = strategy.generate(coords, (30, 10))
        assert mask.shape == (30, 10)
        assert mask.dtype == np.float32
        assert mask[15, 3] == 1.0
        assert mask[2, 3] == 0.0

    def test_size_zero_produces_degenerate_near_zero_area_mask(self):
        """size=0 folds the outer edge onto the lash line, producing a
        zero-height polygon — filled area collapses to a thin scanline
        instead of the full ridge-to-lash band."""
        coords = _flat_line_coords()
        image_shape = (30, 10)
        full = TearBagHighlightMaskStrategy(
            _RIDGE, _LASH, size=1.0, envelope=(1.0,) * 7, softness=0.0,
        )
        degenerate = TearBagHighlightMaskStrategy(
            _RIDGE, _LASH, size=0.0, softness=0.0,
        )
        full_mask = full.generate(coords, image_shape)
        degenerate_mask = degenerate.generate(coords, image_shape)
        assert degenerate_mask.sum() < full_mask.sum() * 0.2

    def test_fewer_than_three_points_returns_empty_mask(self):
        """A single-point ridge/lash pair can't form a polygon (< 3 points),
        so generate() takes the early-return branch."""
        coords = _flat_line_coords()
        strategy = TearBagHighlightMaskStrategy(
            [0], [7], size=1.0, envelope=(1.0,), softness=0.0,
        )
        mask = strategy.generate(coords, (30, 10))
        assert mask.shape == (30, 10)
        assert mask.sum() == 0.0


# ── shadow blend ──────────────────────────────────────────────────────────────


class TestShadowBlendStrategy:
    def test_applies_matte_color_over_masked_region(self):
        """TearBagShadowBlendStrategy delegates to blend_matte: a fully
        masked region moves toward the target colour."""
        strategy = TearBagShadowBlendStrategy()
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        params = TearBagShadowParams(
            color_params=ColorParams(target_rgb=(150, 110, 90)),
            opacity=1.0,
        )
        result = strategy.apply(image, mask, params)
        assert result.dtype == np.uint8
        assert result.shape == image.shape
        assert result is not image
        assert result.mean() > image.mean()


# ── pass construction ─────────────────────────────────────────────────────────


class TestPassConstruction:
    def test_highlight_pass_wires_strategies_and_priority(self):
        params = TearBagHighlightParams(
            color_params=ColorParams(target_rgb=(255, 255, 255)),
            opacity=0.3,
        )
        makeup_pass = TearBagHighlightPass(params, "tear_bag_highlight_left")
        assert makeup_pass.pass_type == "tear_bag_highlight_left"
        assert makeup_pass.priority == 22
        assert isinstance(makeup_pass.mask_strategy, TearBagHighlightMaskStrategy)
        assert isinstance(makeup_pass.blend_strategy, TearBagHighlightBlendStrategy)

    def test_shadow_pass_wires_strategies_and_priority(self):
        params = TearBagShadowParams(
            color_params=ColorParams(target_rgb=(150, 110, 90)),
            opacity=0.4,
        )
        makeup_pass = TearBagShadowPass(params, "tear_bag_shadow_right")
        assert makeup_pass.pass_type == "tear_bag_shadow_right"
        assert makeup_pass.priority == 27
        assert isinstance(makeup_pass.mask_strategy, TearBagShadowMaskStrategy)
        assert isinstance(makeup_pass.blend_strategy, TearBagShadowBlendStrategy)


# ── shadow mask dilation ──────────────────────────────────────────────────────


class TestShadowMaskDilation:
    @pytest.fixture
    def horizontal_line_coords(self):
        # 7 collinear points on y=100; ridge and lash share indices so
        # _lerp_outer stays on a single horizontal line regardless of size.
        return np.array(
            [[60 + i * 10, 100.0] for i in range(7)], dtype=np.float32
        )

    def test_thicker_line_covers_more_pixels(self, horizontal_line_coords):
        """A larger line_thickness dilates the polyline into a wider mask."""
        thin = TearBagShadowMaskStrategy(
            _RIDGE, _RIDGE, line_thickness=1, softness=0.0,
        )
        thick = TearBagShadowMaskStrategy(
            _RIDGE, _RIDGE, line_thickness=6, softness=0.0,
        )
        image_shape = (200, 200)
        mask_thin = thin.generate(horizontal_line_coords, image_shape)
        mask_thick = thick.generate(horizontal_line_coords, image_shape)
        assert mask_thick.sum() > mask_thin.sum()

    def test_mask_shape_matches_image(self, horizontal_line_coords):
        """Shadow mask must match the requested image shape."""
        strategy = TearBagShadowMaskStrategy(
            _RIDGE, _RIDGE, line_thickness=2, softness=0.0,
        )
        mask = strategy.generate(horizontal_line_coords, (200, 200))
        assert mask.shape == (200, 200)
        assert mask.dtype == np.float32
