"""
Eye iris makeup pass — ellipse mask + hue-shift or standard blend.
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.constants import (
    EYE_IRIS_LEFT_INDICES,
    EYE_IRIS_RIGHT_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import blend_hue_shift, blend_matte, soften_mask
from face_makeup.core.types import EyeIrisParams




class EyeIrisMaskStrategy(MaskStrategy):
    """Ellipse mask fitted to iris landmark points.

    Clipping to the eye opening is applied by the renderer via
    ``EyeIrisPass.clip_landmarks`` — the mask strategy itself only
    produces the iris ellipse.
    """

    def __init__(
        self,
        indices: list[int],
        blur_strength: float = 0.5,
    ):
        self._indices = indices
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

        points = coords[self._indices]
        if len(points) < 5:
            return soften_mask(mask, self._blur_strength)

        ellipse = cv2.fitEllipse(points)
        cv2.ellipse(mask, ellipse, color=255, thickness=-1)

        return soften_mask(mask, self._blur_strength)


class EyeIrisBlendStrategy(BlendStrategy):
    """Dispatches to hue-shift or matte colour over a ready soft mask."""

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
            return blend_hue_shift(
                image, mask,
                params.color_params.target_rgb,
                params.opacity,
            )
        return blend_matte(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
        )


class EyeIrisPass(MakeupPass):
    """Eye iris (color contact lens) makeup pass.

    The iris ellipse is clipped to the eye opening contour by the
    renderer via ``clip_landmarks``, preventing colour bleed onto eyelid
    skin when blinking, at extreme gaze angles, or under landmark noise.
    """

    def __init__(self, params: EyeIrisParams, side: str):
        part_name = f"eyeiris_{side}"
        indices = EYE_IRIS_LEFT_INDICES if side == "left" else EYE_IRIS_RIGHT_INDICES
        contour_indices = (
            LEFT_EYE_INDICES if side == "left" else RIGHT_EYE_INDICES
        )
        super().__init__(
            pass_type=part_name,
            priority=35,
            params=params,
            landmark_indices=list(indices),
            mask_strategy=EyeIrisMaskStrategy(
                list(indices), params.blur_strength,
            ),
            blend_strategy=EyeIrisBlendStrategy(),
            clip_landmarks=[list(contour_indices)],
        )
