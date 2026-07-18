"""
Tests for cheek pass: CheekMaskStrategy.generate() (production entry point),
CheekBlendStrategy, and CheekPass construction.

test_ellipse_mask_debug_info.py already covers create_cheek_mask's debug_info
dict; this file exercises the actual generate() path (convex hull ∩ 3D ellipse
→ soften) that the renderer calls, which had no coverage before.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_makeup.constants import CHEEK_LEFT_INDICES, CHEEK_RIGHT_INDICES
from face_makeup.core.passes.cheek import (
    CheekBlendStrategy,
    CheekMaskStrategy,
    CheekPass,
)
from face_makeup.core.types import CheekParams, ColorParams, FaceLandmarks

_IMAGE_SIZE = 100


def _build_cheek_coords(indices: list[int]) -> np.ndarray:
    """Pixel-space coords with the given cheek indices on a near-full-image
    circle, so the convex hull covers most of the frame regardless of where
    a randomly-generated 3D ellipse projects to.

    int32, matching the contract of FaceLandmarks.to_image_coords() — the
    only producer of coords in the real pipeline (renderer.py). cheek.py's
    cv2.convexHull/fillConvexPoly calls assume already-integer points, unlike
    the sub-pixel float→int32 casts used by lips/eyeliner/eyeshadow.
    """
    max_idx = max(indices)
    coords = np.zeros((max_idx + 1, 2), dtype=np.int32)
    center = _IMAGE_SIZE / 2
    radius = _IMAGE_SIZE * 0.45
    angles = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
    for idx, angle in zip(indices, angles):
        coords[idx] = [
            round(center + radius * np.cos(angle)),
            round(center + radius * np.sin(angle)),
        ]
    return coords


@pytest.fixture
def cheek_params():
    return CheekParams(
        color_params=ColorParams(target_rgb=(255, 150, 150)),
        opacity=0.5,
        center=(0.0, 0.0),
        radius=(0.5, 0.5),
    )


@pytest.fixture
def landmarks_3d():
    """Deterministic 3D landmarks so the tangent-plane ellipse lands
    reproducibly within the image bounds (seeded — avoids flaky overlap
    with the convex hull across runs)."""
    rng = np.random.default_rng(42)
    num_landmarks = 478
    points_2d = rng.random((num_landmarks, 2)).astype(np.float32) * 0.5 + 0.25
    points_3d = rng.random((num_landmarks, 3)).astype(np.float32) * 0.5 + 0.25
    return FaceLandmarks(points=points_2d, points_3d=points_3d)


@pytest.fixture
def landmarks_2d_only():
    num_landmarks = 478
    rng = np.random.default_rng(42)
    points_2d = rng.random((num_landmarks, 2)).astype(np.float32) * 0.5 + 0.25
    return FaceLandmarks(points=points_2d, points_3d=None)


# ── CheekMaskStrategy.generate (production entry point) ────────────────────


class TestCheekMaskStrategyGenerate:
    """Tests for the real generate() path: convex hull ∩ ellipse → soften."""

    @pytest.mark.parametrize("side,indices", [
        ("left", CHEEK_LEFT_INDICES),
        ("right", CHEEK_RIGHT_INDICES),
    ])
    def test_generate_3d_path_returns_valid_soft_mask(
        self, cheek_params, landmarks_3d, side, indices
    ):
        """3D path (landmarks with points_3d) produces a valid float32 soft mask."""
        coords = _build_cheek_coords(list(indices))
        strategy = CheekMaskStrategy(list(indices), side, cheek_params)
        mask = strategy.generate(
            coords, (_IMAGE_SIZE, _IMAGE_SIZE), landmarks=landmarks_3d
        )
        assert mask.shape == (_IMAGE_SIZE, _IMAGE_SIZE)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.max() > 0.0, "hull ∩ ellipse should be non-empty for these fixtures"

    def test_generate_2d_fallback_path_returns_valid_soft_mask(
        self, cheek_params, landmarks_2d_only
    ):
        """2D fallback (points_3d is None) produces a valid float32 soft mask."""
        indices = list(CHEEK_LEFT_INDICES)
        coords = _build_cheek_coords(indices)
        strategy = CheekMaskStrategy(indices, "left", cheek_params)
        mask = strategy.generate(
            coords, (_IMAGE_SIZE, _IMAGE_SIZE), landmarks=landmarks_2d_only
        )
        assert mask.shape == (_IMAGE_SIZE, _IMAGE_SIZE)
        assert mask.dtype == np.float32
        assert mask.max() > 0.0

    def test_generate_no_landmarks_uses_2d_fallback(self, cheek_params):
        """landmarks=None also takes the 2D fallback path (no 3D data available)."""
        indices = list(CHEEK_LEFT_INDICES)
        coords = _build_cheek_coords(indices)
        strategy = CheekMaskStrategy(indices, "left", cheek_params)
        mask = strategy.generate(coords, (_IMAGE_SIZE, _IMAGE_SIZE), landmarks=None)
        assert mask.shape == (_IMAGE_SIZE, _IMAGE_SIZE)
        assert mask.max() > 0.0

    def test_generate_too_few_points_returns_soft_empty_mask(self, cheek_params):
        """Fewer than 3 hull points → early-return soft (all-zero-ish) mask,
        not a crash."""
        indices = list(CHEEK_LEFT_INDICES)
        coords = np.zeros((max(indices) + 1, 2), dtype=np.float32)
        strategy = CheekMaskStrategy(indices[:2], "left", cheek_params)
        mask = strategy.generate(coords, (_IMAGE_SIZE, _IMAGE_SIZE), landmarks=None)
        assert mask.shape == (_IMAGE_SIZE, _IMAGE_SIZE)
        assert mask.dtype == np.float32
        assert mask.max() == 0.0

    def test_softness_widens_feather_band(self, cheek_params, landmarks_3d):
        """Higher softness → more distinct gray levels near the mask edge."""
        indices = list(CHEEK_LEFT_INDICES)
        coords = _build_cheek_coords(indices)

        params_sharp = CheekParams(
            color_params=ColorParams(target_rgb=(255, 150, 150)),
            opacity=0.5, softness=0.0,
        )
        params_soft = CheekParams(
            color_params=ColorParams(target_rgb=(255, 150, 150)),
            opacity=0.5, softness=1.0,
        )
        sharp = CheekMaskStrategy(indices, "left", params_sharp).generate(
            coords, (_IMAGE_SIZE, _IMAGE_SIZE), landmarks=landmarks_3d
        )
        soft = CheekMaskStrategy(indices, "left", params_soft).generate(
            coords, (_IMAGE_SIZE, _IMAGE_SIZE), landmarks=landmarks_3d
        )
        sharp_full = np.sum(sharp > 0.99)
        soft_full = np.sum(soft > 0.99)
        assert soft_full <= sharp_full


# ── CheekBlendStrategy ───────────────────────────────────────────────────


class TestCheekBlendStrategy:
    """Tests for CheekBlendStrategy.apply — pure matte colour over a ready mask."""

    @pytest.fixture
    def strategy(self):
        return CheekBlendStrategy()

    @pytest.fixture
    def solid_gray(self):
        return np.full((_IMAGE_SIZE, _IMAGE_SIZE, 3), 128, dtype=np.uint8)

    @pytest.fixture
    def center_mask(self):
        mask = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
        mask[25:75, 25:75] = 1.0
        return mask

    def test_apply_returns_same_shape_and_dtype(self, strategy, solid_gray, center_mask, cheek_params):
        result = strategy.apply(solid_gray, center_mask, cheek_params)
        assert result.shape == solid_gray.shape
        assert result.dtype == np.uint8

    def test_apply_zero_mask_no_change(self, strategy, solid_gray, cheek_params):
        zero_mask = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
        result = strategy.apply(solid_gray, zero_mask, cheek_params)
        assert np.array_equal(result, solid_gray)

    def test_apply_zero_opacity_no_change(self, strategy, solid_gray, center_mask):
        params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 150, 150)),
            opacity=0.0,
        )
        result = strategy.apply(solid_gray, center_mask, params)
        assert np.array_equal(result, solid_gray)

    def test_apply_nonzero_mask_changes_image(self, strategy, solid_gray, center_mask, cheek_params):
        result = strategy.apply(solid_gray, center_mask, cheek_params)
        assert not np.array_equal(result, solid_gray)


# ── CheekPass ────────────────────────────────────────────────────────────


class TestCheekPass:
    """Tests for CheekPass construction and wiring."""

    @pytest.mark.parametrize("side,expected_indices", [
        ("left", CHEEK_LEFT_INDICES),
        ("right", CHEEK_RIGHT_INDICES),
    ])
    def test_pass_type(self, cheek_params, side, expected_indices):
        p = CheekPass(cheek_params, side)
        assert p.pass_type == f"cheek_{side}"

    def test_priority(self, cheek_params):
        p = CheekPass(cheek_params, "left")
        assert p.priority == 15

    def test_has_mask_strategy(self, cheek_params):
        p = CheekPass(cheek_params, "left")
        assert isinstance(p.mask_strategy, CheekMaskStrategy)

    def test_has_blend_strategy(self, cheek_params):
        p = CheekPass(cheek_params, "left")
        assert isinstance(p.blend_strategy, CheekBlendStrategy)

    def test_left_side_selects_left_indices(self, cheek_params):
        p = CheekPass(cheek_params, "left")
        assert p.mask_strategy._indices == list(CHEEK_LEFT_INDICES)
        assert p.mask_strategy._cheek_side == "left"

    def test_right_side_selects_right_indices(self, cheek_params):
        p = CheekPass(cheek_params, "right")
        assert p.mask_strategy._indices == list(CHEEK_RIGHT_INDICES)
        assert p.mask_strategy._cheek_side == "right"

    def test_mask_strategy_stores_params(self, cheek_params):
        """CheekPass wires params through so the strategy can read
        _FEATHER_SCALE / _SOFTEN_FLOOR and geometry fields at generate time."""
        p = CheekPass(cheek_params, "left")
        assert p.mask_strategy._params is cheek_params
