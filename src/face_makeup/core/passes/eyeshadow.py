"""
Eye shadow makeup pass — size-scaled polygon mask + HSV-aware blend.

The mask fills the band between the lash line and an upper boundary lerped
point-by-point from the lash line (size=0) toward the crease line (size=1),
mirroring the tear-bag ``_lerp_outer`` approach. A small size paints a thin
band along the lash line; 1.0 covers the whole lid. Layered looks (e.g. a
dark rim under a lighter full lid) are built by adding multiple passes with
different sizes.

Both boundary polylines are smoothed with a Catmull-Rom spline before the
polygon is filled, so the sparse crease landmarks (3 points on the
center/inner zones) do not leave a polygonal kink on the mask edge. The
spline reproduces the seam vertices exactly, keeping adjacent zones tiled
without gaps at any size.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from face_makeup.constants import (
    EYELID_CENTER_LEFT_INDICES,
    EYELID_CENTER_RIGHT_INDICES,
    EYELID_CREASE_LEFT_INDICES,
    EYELID_CREASE_RIGHT_INDICES,
    EYELID_INNER_CENTER_LEFT_INDICES,
    EYELID_INNER_CENTER_RIGHT_INDICES,
    EYELID_INNER_LEFT_INDICES,
    EYELID_INNER_RIGHT_INDICES,
    EYELID_LASH_LEFT_INDICES,
    EYELID_LASH_RIGHT_INDICES,
    EYELID_OUTER_CENTER_LEFT_INDICES,
    EYELID_OUTER_CENTER_RIGHT_INDICES,
    EYELID_OUTER_LEFT_INDICES,
    EYELID_OUTER_RIGHT_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import apply_glitter, blend_matte, soften_mask
from face_makeup.core.types import EyeShadowParams

if TYPE_CHECKING:
    from face_makeup.core.types import FaceLandmarks

# OpenCV sub-pixel precision bits for anti-aliased fillPoly.
_SUBPIXEL_BITS = 4

# Catmull-Rom interpolation points generated per original polyline segment.
# Higher = smoother edge but more fillPoly vertices; 10 is visually smooth
# without measurable cost on typical face-image resolutions.
_SPLINE_SAMPLES_PER_SEGMENT = 10


def _catmull_rom_resample(
    points: np.ndarray,
    samples_per_segment: int = _SPLINE_SAMPLES_PER_SEGMENT,
) -> np.ndarray:
    """Resample a polyline through Catmull-Rom spline interpolation.

    Both endpoints are reproduced exactly, so adjacent zones that share seam
    vertices still produce identical seam endpoints — no gaps open between
    zone masks at any ``size``. Interior vertices become smooth curve points,
    removing the polygonal kinks that sparse crease landmarks (3 points on
    the center/inner zones) otherwise leave on the mask's upper edge.

    For ``N < 3`` the input has no interior to smooth and is returned as-is.

    Args:
        points: (N, 2) float32 pixel coordinates.
        samples_per_segment: Interpolation points emitted per original segment.

    Returns:
        (M, 2) float32 with ``M = 2 + (N - 1) * samples_per_segment``.
    """
    n = len(points)
    if n < 3:
        return points.astype(np.float32, copy=True)

    # Mirror-reflect the endpoints so the boundary tangents follow the local
    # polyline direction instead of clamping the curve flat at the seams.
    extended = np.vstack([
        2.0 * points[0] - points[1],
        points,
        2.0 * points[-1] - points[-2],
    ])

    resampled = [points[0]]
    for i in range(len(extended) - 3):
        p0, p1, p2, p3 = extended[i], extended[i + 1], extended[i + 2], extended[i + 3]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            resampled.append(pt)
    resampled.append(points[-1])
    return np.asarray(resampled, dtype=np.float32)


def _lerp_upper(
    coords: np.ndarray,
    crease_indices: list[int],
    lash_indices: list[int],
    size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the lerped upper boundary and the fixed lash lower edge.

    Both outputs are float32 pixel coordinates ready for drawing. ``size``
    blends each upper point from the lash line (0.0) toward the crease line
    (1.0), so 0.0 collapses the band onto the lash line and 1.0 keeps the
    full crease-to-lash band.

    The crease polyline is resampled to the lash point count, so adjacent
    zones that share their seam vertices produce identical seam endpoints
    at any size (no gaps between zone masks).

    Returns:
        upper_pts: lerped upper boundary (inner→outer, len = len(lash_indices))
        lash_pts:  lash lower boundary  (inner→outer, len = len(lash_indices))
    """
    crease_pts = coords[crease_indices].astype(np.float32)  # outer→inner
    lash_pts = coords[lash_indices].astype(np.float32)      # inner→outer
    crease_lerp = crease_pts[::-1]                           # inner→outer (same dir as lash)

    n = len(lash_pts)
    m = len(crease_lerp)
    if m != n:
        t_src = np.linspace(0.0, 1.0, m, dtype=np.float32)
        t_dst = np.linspace(0.0, 1.0, n, dtype=np.float32)
        crease_lerp = np.column_stack([
            np.interp(t_dst, t_src, crease_lerp[:, c]) for c in range(2)
        ]).astype(np.float32)

    upper_pts = (1.0 - size) * lash_pts + size * crease_lerp
    return upper_pts, lash_pts


class EyeShadowMaskStrategy(MaskStrategy):
    """Filled-polygon mask between the lash line and the size-lerped upper boundary."""

    def __init__(
        self,
        crease_indices: list[int],
        lash_indices: list[int],
        size: float = 1.0,
        blur_strength: float = 0.5,
    ):
        self._crease_indices = list(crease_indices)
        self._lash_indices = list(lash_indices)
        self._size = size
        self._blur_strength = blur_strength

    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: object | None = None,
        segmenter: object | None = None,
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(self._crease_indices) < 1 or len(self._lash_indices) < 2:
            return soften_mask(mask, self._blur_strength)

        upper_pts, lash_pts = _lerp_upper(
            coords, self._crease_indices, self._lash_indices, self._size
        )
        # Smooth both boundary polylines with Catmull-Rom before filling, so
        # the sparse crease landmarks do not leave a polygonal kink on the
        # upper edge. Endpoints are preserved → seams still match neighbours.
        upper_smooth = _catmull_rom_resample(upper_pts)
        lash_smooth = _catmull_rom_resample(lash_pts)
        # upper (inner→outer) + lash (outer→inner) forms a closed polygon
        poly = np.vstack([upper_smooth, lash_smooth[::-1]])

        shift = _SUBPIXEL_BITS
        scale = 1 << shift
        points_shifted = (poly * scale).astype(np.int32)
        cv2.fillPoly(mask, [points_shifted], 255, shift=shift)

        return soften_mask(mask, self._blur_strength)


class EyeShadowBlendStrategy(BlendStrategy):
    """Pure matte colour over a ready soft mask, with optional glitter overlay."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: EyeShadowParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> np.ndarray:
        result = blend_matte(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
        )
        if params.glitter > 0:
            rng = np.random.default_rng(params.seed) if params.seed is not None else None
            result = apply_glitter(result, mask, params.glitter, rng=rng)
        return result


# Maps part_name → (crease_indices outer→inner, lash_indices inner→outer).
_INDICES_MAP: dict[str, tuple[list[int], list[int]]] = {
    "eyelid_left": (EYELID_CREASE_LEFT_INDICES, EYELID_LASH_LEFT_INDICES),
    "eyelid_right": (EYELID_CREASE_RIGHT_INDICES, EYELID_LASH_RIGHT_INDICES),
    # Horizontal zones (outer/center/inner).
    "eyelid_outer_left": EYELID_OUTER_LEFT_INDICES,
    "eyelid_center_left": EYELID_CENTER_LEFT_INDICES,
    "eyelid_inner_left": EYELID_INNER_LEFT_INDICES,
    "eyelid_outer_right": EYELID_OUTER_RIGHT_INDICES,
    "eyelid_center_right": EYELID_CENTER_RIGHT_INDICES,
    "eyelid_inner_right": EYELID_INNER_RIGHT_INDICES,
    # Extended zones (outer+center, inner+center).
    "eyelid_inner_center_left": EYELID_INNER_CENTER_LEFT_INDICES,
    "eyelid_inner_center_right": EYELID_INNER_CENTER_RIGHT_INDICES,
    "eyelid_outer_center_left": EYELID_OUTER_CENTER_LEFT_INDICES,
    "eyelid_outer_center_right": EYELID_OUTER_CENTER_RIGHT_INDICES,
}


class EyeShadowPass(MakeupPass):
    """Eye shadow makeup pass with size-scaled polygon mask."""

    def __init__(self, params: EyeShadowParams, part_name: str):
        crease_indices, lash_indices = _INDICES_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=25,
            params=params,
            landmark_indices=list(crease_indices) + list(lash_indices),
            mask_strategy=EyeShadowMaskStrategy(
                list(crease_indices), list(lash_indices), params.size,
                params.blur_strength,
            ),
            blend_strategy=EyeShadowBlendStrategy(),
        )
