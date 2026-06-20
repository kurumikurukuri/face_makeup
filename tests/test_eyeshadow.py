"""
Tests for eye shadow pass: EyeShadowMaskStrategy and EyeShadowBlendStrategy.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.constants import (
    EYELID_EDGE_LEFT_INDICES,
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


def _build_eyelid_coords(indices: list[int]) -> np.ndarray:
    """Build a coords array with a polygon shape for the given landmark indices."""
    max_idx = max(indices)
    coords = np.zeros((max_idx + 1, 2), dtype=np.float32)
    for i, idx in enumerate(sorted(set(indices))):
        angle = 2 * np.pi * i / len(set(indices))
        coords[idx] = [50.0 + 20.0 * np.cos(angle), 50.0 + 10.0 * np.sin(angle)]
    return coords


# ── EyeShadowMaskStrategy ────────────────────────────────────────────────────


class TestEyeShadowMaskStrategy:
    """Tests for EyeShadowMaskStrategy mask generation."""

    def test_generate_with_valid_indices(self):
        """Valid polygon indices produce non-zero mask."""
        indices = list(EYELID_LEFT_INDICES)
        coords = _build_eyelid_coords(indices)
        strategy = EyeShadowMaskStrategy(indices)
        mask = strategy.generate(coords, (100, 100))
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)
        assert mask.max() == 255

    def test_generate_fewer_than_three_points(self):
        """Fewer than 3 points → zero mask."""
        indices = [EYELID_LEFT_INDICES[0], EYELID_LEFT_INDICES[1]]
        coords = np.zeros((max(indices) + 1, 2), dtype=np.float32)
        coords[indices[0]] = [50.0, 50.0]
        coords[indices[1]] = [55.0, 50.0]
        strategy = EyeShadowMaskStrategy(indices)
        mask = strategy.generate(coords, (100, 100))
        assert mask.max() == 0

    def test_generate_eyelid_edge_indices(self):
        """Eyelid edge indices also produce valid mask."""
        indices = list(EYELID_EDGE_LEFT_INDICES)
        coords = _build_eyelid_coords(indices)
        strategy = EyeShadowMaskStrategy(indices)
        mask = strategy.generate(coords, (100, 100))
        assert mask.shape == (100, 100)
        assert mask.max() == 255


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
