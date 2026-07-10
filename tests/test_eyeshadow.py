"""
Tests for eye shadow pass: EyeShadowMaskStrategy and EyeShadowBlendStrategy.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from face_makeup.constants import (
    EYELID_CREASE_LEFT_INDICES,
    EYELID_LASH_LEFT_INDICES,
    EYELID_LEFT_INDICES,
)
from face_makeup.core.passes.eyeshadow import (
    EyeShadowBlendStrategy,
    EyeShadowMaskStrategy,
)
from face_makeup.core.types import (
    ColorParams,
    EyeShadowParams,
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


def _build_band_coords(crease: list[int], lash: list[int]) -> np.ndarray:
    """Build coords with the crease on an upper line and the lash on a lower line.

    Crease indices (outer→inner) run right-to-left on y=30; lash indices
    (inner→outer) run left-to-right on y=60, so the closed polygon is a
    horizontal band whose height scales with ``size``.
    """
    max_idx = max(max(crease), max(lash))
    coords = np.zeros((max_idx + 1, 2), dtype=np.float32)
    for idx, x in zip(crease, np.linspace(80.0, 20.0, len(crease))):
        coords[idx] = [x, 30.0]
    for idx, x in zip(lash, np.linspace(20.0, 80.0, len(lash))):
        coords[idx] = [x, 60.0]
    return coords


def _mask_area(size: float) -> int:
    crease = list(EYELID_CREASE_LEFT_INDICES)
    lash = list(EYELID_LASH_LEFT_INDICES)
    coords = _build_band_coords(crease, lash)
    strategy = EyeShadowMaskStrategy(crease, lash, size=size)
    return int(np.count_nonzero(strategy.generate(coords, (100, 100))))


# ── EyeShadowMaskStrategy ────────────────────────────────────────────────────


class TestEyeShadowMaskStrategy:
    """Tests for EyeShadowMaskStrategy mask generation."""

    def test_generate_with_valid_indices(self):
        """Valid crease/lash indices produce non-zero mask."""
        crease = list(EYELID_CREASE_LEFT_INDICES)
        lash = list(EYELID_LASH_LEFT_INDICES)
        coords = _build_band_coords(crease, lash)
        strategy = EyeShadowMaskStrategy(crease, lash)
        mask = strategy.generate(coords, (100, 100))
        # Mask strategy now bakes the fade in and returns float32 0-1.
        assert mask.dtype == np.float32
        assert mask.shape == (100, 100)
        assert 0.0 < mask.max() <= 1.0

    def test_generate_fewer_than_three_points(self):
        """Fewer than 3 polygon points → zero mask."""
        crease = [EYELID_CREASE_LEFT_INDICES[0]]
        lash = [EYELID_LASH_LEFT_INDICES[0]]
        coords = np.zeros((max(crease + lash) + 1, 2), dtype=np.float32)
        coords[crease[0]] = [50.0, 30.0]
        coords[lash[0]] = [50.0, 60.0]
        strategy = EyeShadowMaskStrategy(crease, lash)
        mask = strategy.generate(coords, (100, 100))
        assert mask.max() == 0


class TestEyeShadowMaskSize:
    """Tests for the size parameter of EyeShadowMaskStrategy."""

    def test_size_area_monotonic(self):
        """Mask area grows monotonically with size."""
        areas = [_mask_area(s) for s in (0.0, 0.25, 0.5, 1.0)]
        assert areas[0] < areas[1] < areas[2] < areas[3]

    def test_size_zero_collapses_onto_lash_line(self):
        """size=0 degenerates the band onto the lash line (near-zero area)."""
        assert _mask_area(0.0) < 0.05 * _mask_area(1.0)

    def test_size_one_matches_legacy_full_polygon(self):
        """size=1 reproduces the legacy concatenated-indices polygon (IoU)."""
        coords = _build_band_coords(
            list(EYELID_CREASE_LEFT_INDICES), list(EYELID_LASH_LEFT_INDICES)
        )
        legacy = np.zeros((100, 100), dtype=np.uint8)
        shift = 4
        points = (coords[EYELID_LEFT_INDICES] * (1 << shift)).astype(np.int32)
        cv2.fillPoly(legacy, [points], 255, shift=shift)

        strategy = EyeShadowMaskStrategy(
            list(EYELID_CREASE_LEFT_INDICES),
            list(EYELID_LASH_LEFT_INDICES),
            size=1.0,
        )
        new = strategy.generate(coords, (100, 100))

        intersection = np.count_nonzero((legacy > 0) & (new > 0))
        union = np.count_nonzero((legacy > 0) | (new > 0))
        assert intersection / union > 0.99

    def test_default_size_is_full(self):
        """Omitting size behaves like size=1.0."""
        crease = list(EYELID_CREASE_LEFT_INDICES)
        lash = list(EYELID_LASH_LEFT_INDICES)
        coords = _build_band_coords(crease, lash)
        default_mask = EyeShadowMaskStrategy(crease, lash).generate(coords, (100, 100))
        full_mask = EyeShadowMaskStrategy(crease, lash, size=1.0).generate(coords, (100, 100))
        assert np.array_equal(default_mask, full_mask)


# ── EyeShadowParams size validation ──────────────────────────────────────────


class TestEyeShadowParamsSizeValidation:
    """Tests for EyeShadowParams size validation at the dataclass level."""

    @staticmethod
    def _params(**kwargs) -> EyeShadowParams:
        return EyeShadowParams(
            color_params=ColorParams(target_rgb=(80, 60, 200)),
            opacity=0.5,
            **kwargs,
        )

    def test_size_default_is_one(self):
        assert self._params().size == 1.0

    def test_size_too_high(self):
        with pytest.raises(ValueError, match="size must be"):
            self._params(size=1.5)

    def test_size_negative(self):
        with pytest.raises(ValueError, match="size must be"):
            self._params(size=-0.1)

    def test_size_string(self):
        with pytest.raises(ValueError, match="size must be"):
            self._params(size="0.5")

    def test_size_boundaries_valid(self):
        assert self._params(size=0.0).size == 0.0
        assert self._params(size=1.0).size == 1.0


# ── EyeShadowBlendStrategy ───────────────────────────────────────────────────


class TestEyeShadowBlendStrategy:
    """Tests for EyeShadowBlendStrategy blend application."""

    def test_apply_output_contract(self, solid_gray, center_mask):
        """Output has same shape and dtype as input."""
        strategy = EyeShadowBlendStrategy()
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(80, 60, 200)),
            opacity=0.8,
            blur_strength=0.3,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert result.shape == solid_gray.shape
        assert result.dtype == np.uint8

    def test_apply_zero_mask_no_change(self, solid_gray):
        """All-zero mask → input unchanged."""
        strategy = EyeShadowBlendStrategy()
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(80, 60, 200)),
            opacity=0.8,
            blur_strength=0.3,
        )
        zero_mask = np.zeros((100, 100), dtype=np.float32)
        result = strategy.apply(solid_gray, zero_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_apply_opacity_zero_unchanged(self, solid_gray, center_mask):
        """opacity=0 → input unchanged."""
        strategy = EyeShadowBlendStrategy()
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(80, 60, 200)),
            opacity=0.0,
            blur_strength=0.3,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_apply_outside_mask_unchanged(self, solid_gray, center_mask):
        """Pixels outside mask region unchanged."""
        strategy = EyeShadowBlendStrategy()
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(80, 60, 200)),
            opacity=1.0,
            blur_strength=0.0,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result[0:10, 0:10], solid_gray[0:10, 0:10])
