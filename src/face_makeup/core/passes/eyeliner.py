"""
Eyeliner makeup pass — curve mask with optional rule-based edge correction.
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.constants import (
    EYELINER_CORRECTION_MAP,
    EYELINER_LEFT_INDICES,
    EYELINER_RIGHT_INDICES,
)
from face_makeup.core.eyeliner_corrector import EyelinerCorrector
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import blend_matte, soften_mask
from face_makeup.core.types import EyeLinerParams

# OpenCV sub-pixel precision bits for anti-aliased polylines.
_SUBPIXEL_BITS = 4

_INDICES_MAP = {
    "eyeliner_left": EYELINER_LEFT_INDICES,
    "eyeliner_right": EYELINER_RIGHT_INDICES,
}


class EyelinerMaskStrategy(MaskStrategy):
    """Curve mask for eyeliner, optionally corrected via rule-based edge detection."""

    def __init__(
        self,
        part_name: str,
        thickness: int = 2,
        blur_strength: float = 0.5,
    ):
        self._part_name = part_name
        self._indices_dict = _INDICES_MAP[part_name]
        self._correction_map = EYELINER_CORRECTION_MAP.get(part_name, {})
        self._corrector = EyelinerCorrector()
        self._thickness = thickness
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

        if image is not None and self._correction_map:
            mask = self._generate_corrected(coords, image_shape, image)
        else:
            mask = self._generate_simple(coords, image_shape)

        # Thickness dilation belongs to geometry, so it must run before the
        # fade — otherwise the added band would have a hard (unfaded) edge.
        if self._thickness > 1:
            kernel = np.ones((self._thickness, self._thickness), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return soften_mask(mask, self._blur_strength)

    def _generate_simple(
        self, coords: np.ndarray, image_shape: tuple[int, int]
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        thickness = max(1, min(h, w) // 512)
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

    def _generate_corrected(
        self, coords: np.ndarray, image_shape: tuple[int, int], image: np.ndarray
    ) -> np.ndarray:
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        thickness = max(1, min(h, w) // 512)
        shift = _SUBPIXEL_BITS
        scale = 1 << shift

        corrected_flat = self._corrector.correct_points(
            coords, self._part_name, eyeliner=image
        )

        offset = 0
        for sub_indices in self._correction_map.values():
            n = len(sub_indices)
            sub_points = corrected_flat[offset:offset + n]
            offset += n
            if len(sub_points) < 2:
                continue
            points_shifted = (sub_points * scale).astype(np.int32)
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
        blender: object | None = None,
        segmenter: object | None = None,
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
        indices = EYELINER_LEFT_INDICES if side == "left" else EYELINER_RIGHT_INDICES
        super().__init__(
            pass_type=part_name,
            priority=30,
            params=params,
            landmark_indices=indices,
            mask_strategy=EyelinerMaskStrategy(
                part_name, params.thickness, params.blur_strength
            ),
            blend_strategy=EyelinerBlendStrategy(),
        )
