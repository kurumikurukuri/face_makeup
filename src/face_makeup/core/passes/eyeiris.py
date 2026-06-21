"""
Eye iris makeup pass — ellipse mask + hue-shift or standard blend.
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.constants import (
    EYE_IRIS_LEFT_INDICES,
    EYE_IRIS_RIGHT_INDICES,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import apply_hue_shift_blend, standard_blend
from face_makeup.core.types import EyeIrisParams


_EYE_CONTOUR_MAP = {
    "eyeiris_left": LEFT_EYE_IDX,
    "eyeiris_right": RIGHT_EYE_IDX,
}


_INDICES_MAP = {
    "eyeiris_left": EYE_IRIS_LEFT_INDICES,
    "eyeiris_right": EYE_IRIS_RIGHT_INDICES,
}


class EyeIrisMaskStrategy(MaskStrategy):
    """Ellipse mask fitted to iris landmark points, clipped to eye opening."""

    def __init__(self, indices: list[int], eye_contour_indices: list[int]):
        self._indices = indices
        self._eye_contour_indices = eye_contour_indices

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

        points = coords[self._indices]
        if len(points) < 5:
            return mask

        # 1. Iris ellipse from 5 landmark points
        iris_mask = np.zeros((h, w), dtype=np.uint8)
        ellipse = cv2.fitEllipse(points)
        cv2.ellipse(iris_mask, ellipse, color=255, thickness=-1)

        # 2. Clip to eye opening contour to prevent colour bleeding onto
        #    eyelid skin when blinking, at extreme gaze angles, or under
        #    landmark noise.
        contour_pts = coords[self._eye_contour_indices]
        if len(contour_pts) >= 3:
            eye_opening = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(eye_opening, [contour_pts.astype(np.int32)], color=255)
            mask = cv2.bitwise_and(iris_mask, eye_opening)
        else:
            # Fallback: not enough contour points — use unclipped ellipse.
            mask = iris_mask

        return mask


class EyeIrisBlendStrategy(BlendStrategy):
    """Dispatches to hue-shift or standard blend based on params."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: EyeIrisParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: object | None = None,
    ) -> np.ndarray:
        if params.hue_shift:
            return apply_hue_shift_blend(
                image, mask,
                params.color_params.target_rgb,
                params.opacity,
                params.blur_strength,
            )
        return standard_blend(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
            params.blur_strength,
        )


class EyeIrisPass(MakeupPass):
    """Eye iris (color contact lens) makeup pass."""

    def __init__(self, params: EyeIrisParams, side: str):
        part_name = f"eyeiris_{side}"
        indices = EYE_IRIS_LEFT_INDICES if side == "left" else EYE_IRIS_RIGHT_INDICES
        contour_indices = _EYE_CONTOUR_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=35,
            params=params,
            landmark_indices=list(indices),
            mask_strategy=EyeIrisMaskStrategy(list(indices), list(contour_indices)),
            blend_strategy=EyeIrisBlendStrategy(),
        )
