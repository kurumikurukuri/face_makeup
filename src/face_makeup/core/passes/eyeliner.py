"""
Eyeliner makeup pass — curve mask along the (corrected) lash line.

The lash-line correction itself runs once per face in the renderer
(:mod:`face_makeup.core.landmark_correction`); this pass draws the edge
landmarks it is given.
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.constants import (
    EYELINER_LEFT_INDICES,
    EYELINER_RIGHT_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import (
    base_stroke_thickness,
    blend_matte,
    feather_px_from_softness,
    soften_mask,
)
from face_makeup.core.types import EyeLinerParams

# OpenCV sub-pixel precision bits for anti-aliased polylines.
_SUBPIXEL_BITS = 4

_INDICES_MAP = {
    "eyeliner_left": EYELINER_LEFT_INDICES,
    "eyeliner_right": EYELINER_RIGHT_INDICES,
}


class EyelinerMaskStrategy(MaskStrategy):
    """Curve mask for eyeliner along the upper and lower lash lines."""

    uses_lash_line = True

    def __init__(
        self,
        part_name: str,
        thickness: int = 2,
        softness: float = 0.5,
        feather_scale: float = 0.04,
        soften_floor: float = 0.1,
    ):
        self._part_name = part_name
        self._indices_dict = _INDICES_MAP[part_name]
        self._thickness = thickness
        self._softness = softness
        self._feather_scale = feather_scale
        self._soften_floor = soften_floor

    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: object | None = None,
    ) -> np.ndarray:
        """Return the soft eyeliner mask drawn through *coords* (sub-pixel)."""
        feather_px = feather_px_from_softness(
            self._softness, coords, image_shape,
            feather_scale=self._feather_scale,
        )
        mask = self._draw_polylines(coords, image_shape)

        # Thickness dilation belongs to geometry, so it must run before the
        # fade — otherwise the added band would have a hard (unfaded) edge.
        if self._thickness > 1:
            kernel = np.ones((self._thickness, self._thickness), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return soften_mask(mask, feather_px, soften_floor=self._soften_floor)

    def _draw_polylines(
        self, coords: np.ndarray, image_shape: tuple[int, int]
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        thickness = base_stroke_thickness((h, w))
        shift = _SUBPIXEL_BITS
        scale = 1 << shift

        for indices in self._indices_dict.values():
            points = coords[indices]
            if len(points) < 2:
                continue
            points_shifted = (points * scale).astype(np.int32)
            part_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(
                part_mask, [points_shifted], isClosed=False,
                color=255, thickness=thickness,
                lineType=cv2.LINE_AA, shift=shift,
            )
            mask = cv2.bitwise_or(mask, part_mask)

        return mask


class EyelinerBlendStrategy(BlendStrategy):
    """Pure matte colour over a ready soft (already thickness-dilated) mask."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: EyeLinerParams,
        landmarks: object | None = None,
    ) -> np.ndarray:
        return blend_matte(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
        )


class EyelinerPass(MakeupPass):
    """Eyeliner makeup pass with curve-based mask and thickness dilation."""

    def __init__(self, params: EyeLinerParams, side: str):
        part_name = f"eyeliner_{side}"
        super().__init__(
            pass_type=part_name,
            priority=30,
            params=params,
            mask_strategy=EyelinerMaskStrategy(
                part_name, params.thickness, params.softness,
                feather_scale=params._FEATHER_SCALE,
                soften_floor=params._SOFTEN_FLOOR,
            ),
            blend_strategy=EyelinerBlendStrategy(),
        )
