"""
Tear bag (涙袋) makeup pass — highlight fill + shadow line on the lower eyelid.

  - highlight: filled polygon over the ridge + lower lash line, brightened via
    screen blend. The ``size`` parameter (0.0–1.0) lerps the outer polygon
    edge between the lash line (0.0, no tear bag) and the natural ridge (1.0).
  - shadow:    open polyline tracing the lerped outer edge, darkened via
    standard blend, dilated by the shadow thickness.

Both passes share the same lerp formula so highlight and shadow always track
together as ``size`` changes.

The highlight uses a screen blend (always lightens) rather than the shared
standard_blend, whose achromatic path darkens and would invert a white or
pearl highlight. The change is local to this module — no shared blend code
is modified.
"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import cv2
import numpy as np

from face_makeup.constants import (
    TEAR_BAG_SHADOW_LEFT_INDICES,
    TEAR_BAG_SHADOW_RIGHT_INDICES,
    TEAR_BAG_ROOT_LEFT_INDICES,
    TEAR_BAG_ROOT_RIGHT_INDICES,
    TEAR_BAG_ENVELOPE,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import soften_mask, standard_blend
from face_makeup.core.types import TearBagHighlightParams, TearBagShadowParams

if TYPE_CHECKING:
    from face_makeup.core.types import FaceLandmarks, PartParams


# Maps part_name → (ridge_indices, lash_indices).
# ridge: inner→outer, 9 pts.
# lash:  outer→inner, 9 pts (EYELINER_LOWER order).
_RIDGE_LASH_MAP: dict[str, tuple[list[int], list[int]]] = {
    "tear_bag_highlight_left":  (list(TEAR_BAG_SHADOW_LEFT_INDICES),  list(TEAR_BAG_ROOT_LEFT_INDICES)),
    "tear_bag_highlight_right": (list(TEAR_BAG_SHADOW_RIGHT_INDICES), list(TEAR_BAG_ROOT_RIGHT_INDICES)),
    "tear_bag_shadow_left":     (list(TEAR_BAG_SHADOW_LEFT_INDICES),  list(TEAR_BAG_ROOT_LEFT_INDICES)),
    "tear_bag_shadow_right":    (list(TEAR_BAG_SHADOW_RIGHT_INDICES), list(TEAR_BAG_ROOT_RIGHT_INDICES)),
}


def _shifted_points(points: np.ndarray, shift: int) -> np.ndarray:
    """Scale float landmark points to fixed-point ints for sub-pixel drawing."""
    scale = 1 << shift
    return (points * scale).astype(np.int32)


def _lerp_outer(
    coords: np.ndarray,
    ridge_indices: list[int],
    lash_indices: list[int],
    size: float,
    envelope: Sequence[float] = TEAR_BAG_ENVELOPE,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the lerped outer edge and the fixed lash inner edge.

    Both outputs are float32 pixel coordinates ready for drawing. ``size``
    scales a per-point envelope that blends each ridge point toward the lash
    line, so 0.0 collapses the tear bag onto the lash line and 1.0 keeps the
    full ridge (tapered narrower at the corners).

    Returns:
        outer_pts: lerped outer boundary (inner→outer, len = len(ridge_indices))
        lash_pts:  lash inner boundary  (outer→inner, len = len(lash_indices))
    """
    ridge_pts = coords[ridge_indices].astype(np.float32)   # inner→outer
    lash_pts  = coords[lash_indices].astype(np.float32)    # outer→inner
    lash_lerp = lash_pts[::-1]                              # inner→outer (same dir as ridge)

    n = len(ridge_pts)
    m = len(lash_lerp)
    if m != n:
        t_src = np.linspace(0.0, 1.0, m, dtype=np.float32)
        t_dst = np.linspace(0.0, 1.0, n, dtype=np.float32)
        lash_lerp = np.column_stack([
            np.interp(t_dst, t_src, lash_lerp[:, c]) for c in range(2)
        ]).astype(np.float32)

    # Per-point ridge weight (inner→outer): tapers the tear bag narrower at the
    # corners and widest at the centre. Length must equal len(ridge_pts).
    #←内側　外側→
    env = np.asarray(envelope, dtype=np.float32)
    assert len(env) == n, "envelope length must match ridge point count"
    weights = (size * env)[:, np.newaxis]            # (n, 1) → broadcasts over (n, 2)

    outer_pts = (1.0 - weights) * lash_lerp + weights * ridge_pts
    return outer_pts, lash_pts


class TearBagHighlightMaskStrategy(MaskStrategy):
    """Filled-polygon mask over the tear-bag bulge.

    The polygon spans from the lerped outer edge (ridge at size=1, lash at
    size=0) to the fixed lash inner edge.
    """

    def __init__(
        self,
        ridge_indices: list[int],
        lash_indices: list[int],
        size: float = 1.0,
        envelope: Sequence[float] = TEAR_BAG_ENVELOPE,
    ):
        self._ridge_indices = ridge_indices
        self._lash_indices = lash_indices
        self._size = size
        self._envelope = envelope

    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: FaceLandmarks | None = None,
        segmenter: object | None = None,
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        outer_pts, lash_pts = _lerp_outer(
            coords, self._ridge_indices, self._lash_indices, self._size,
            envelope=self._envelope,
        )
        # outer (inner→outer) + lash (outer→inner) forms a closed polygon
        poly = np.vstack([outer_pts, lash_pts])
        if len(poly) < 3:
            return mask

        shift = 4
        cv2.fillPoly(mask, [_shifted_points(poly, shift)], 255, shift=shift)
        return mask


class TearBagShadowMaskStrategy(MaskStrategy):
    """Open-polyline mask tracing the lerped outer edge."""

    def __init__(
        self,
        ridge_indices: list[int],
        lash_indices: list[int],
        size: float = 1.0,
        envelope: Sequence[float] = TEAR_BAG_ENVELOPE,
    ):
        self._ridge_indices = ridge_indices
        self._lash_indices = lash_indices
        self._size = size
        self._envelope = envelope

    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: FaceLandmarks | None = None,
        segmenter: object | None = None,
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        outer_pts, _ = _lerp_outer(
            coords, self._ridge_indices, self._lash_indices, self._size,
            envelope=self._envelope,
        )
        if len(outer_pts) < 2:
            return mask

        thickness = max(1, min(h, w) // 512)
        shift = 4
        cv2.polylines(
            mask, [_shifted_points(outer_pts, shift)], isClosed=False,
            color=255, thickness=thickness,
            lineType=cv2.LINE_AA, shift=shift,
        )
        return mask


class TearBagHighlightBlendStrategy(BlendStrategy):
    """Screen blend toward a light target colour — brightens the tear bag.

    ``result = 1 - (1 - base)(1 - target)`` always lightens per channel, so a
    white or pearl highlight reads correctly. The pipeline feeds the blend an
    RGB image; ``target_rgb`` is applied channel-wise in the same order.
    """

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: PartParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> np.ndarray:
        soft_mask = soften_mask(mask, params.blur_strength)
        image_f = image.astype(np.float32) / 255.0
        target = np.array(params.color_params.target_rgb, dtype=np.float32) / 255.0

        screened = 1.0 - (1.0 - image_f) * (1.0 - target)
        blend_weight = soft_mask[:, :, np.newaxis] * params.opacity
        result = image_f * (1.0 - blend_weight) + screened * blend_weight
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)


class TearBagShadowBlendStrategy(BlendStrategy):
    """Standard HSV-aware blend with thickness dilation for the shadow line."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: TearBagShadowParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> np.ndarray:
        effective_mask = mask
        if params.thickness > 1:
            kernel = np.ones((params.thickness, params.thickness), np.uint8)
            mask_uint8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            effective_mask = mask_dilated.astype(np.float32) / 255.0

        return standard_blend(
            image, effective_mask,
            params.color_params.target_rgb,
            params.opacity,
            params.blur_strength,
        )


class TearBagHighlightPass(MakeupPass):
    """Tear bag highlight — filled polygon with screen-blend lightening."""

    def __init__(self, params: TearBagHighlightParams, part_name: str):
        ridge_indices, lash_indices = _RIDGE_LASH_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=22,
            params=params,
            landmark_indices=ridge_indices + lash_indices,
            mask_strategy=TearBagHighlightMaskStrategy(ridge_indices, lash_indices, params.size),
            blend_strategy=TearBagHighlightBlendStrategy(),
        )


class TearBagShadowPass(MakeupPass):
    """Tear bag shadow — lerped polyline with thickness dilation."""

    def __init__(self, params: TearBagShadowParams, part_name: str):
        ridge_indices, lash_indices = _RIDGE_LASH_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=27,
            params=params,
            landmark_indices=ridge_indices + lash_indices,
            mask_strategy=TearBagShadowMaskStrategy(ridge_indices, lash_indices, params.size),
            blend_strategy=TearBagShadowBlendStrategy(),
        )
