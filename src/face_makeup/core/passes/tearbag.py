"""
Tear bag (涙袋) makeup pass — highlight fill + shadow line on the lower eyelid.

  - highlight: filled polygon over the ridge + lower lash line
    (TEAR_BAG_HIGHLIGHT_*_INDICES), brightened via screen blend.
  - shadow:    open polyline tracing just the ridge (TEAR_BAG_SHADOW_*_INDICES),
    darkened via standard blend, dilated by the shadow thickness.

The highlight uses a screen blend (always lightens) rather than the shared
standard_blend, whose achromatic path darkens and would invert a white or
pearl highlight. The change is local to this module — no shared blend code
is modified.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from face_makeup.constants import (
    TEAR_BAG_HIGHLIGHT_LEFT_INDICES,
    TEAR_BAG_HIGHLIGHT_RIGHT_INDICES,
    TEAR_BAG_SHADOW_LEFT_INDICES,
    TEAR_BAG_SHADOW_RIGHT_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import soften_mask, standard_blend
from face_makeup.core.types import EyeLinerParams, PartParams

if TYPE_CHECKING:
    from face_makeup.core.types import FaceLandmarks


_INDICES_MAP = {
    "tear_bag_highlight_left": TEAR_BAG_HIGHLIGHT_LEFT_INDICES,
    "tear_bag_highlight_right": TEAR_BAG_HIGHLIGHT_RIGHT_INDICES,
    "tear_bag_shadow_left": TEAR_BAG_SHADOW_LEFT_INDICES,
    "tear_bag_shadow_right": TEAR_BAG_SHADOW_RIGHT_INDICES,
}


def _shifted_points(points: np.ndarray, shift: int) -> np.ndarray:
    """Scale float landmark points to fixed-point ints for sub-pixel drawing."""
    scale = 1 << shift
    return (points * scale).astype(np.int32)


class TearBagHighlightMaskStrategy(MaskStrategy):
    """Filled-polygon mask over the tear-bag bulge (cv2.fillPoly)."""

    def __init__(self, indices: list[int]):
        self._indices = indices

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

        points = coords[self._indices]
        if len(points) < 3:
            return mask

        shift = 4
        cv2.fillPoly(mask, [_shifted_points(points, shift)], 255, shift=shift)
        return mask


class TearBagShadowMaskStrategy(MaskStrategy):
    """Open-polyline mask tracing the tear-bag ridge (cv2.polylines)."""

    def __init__(self, indices: list[int]):
        self._indices = indices

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

        points = coords[self._indices]
        if len(points) < 2:
            return mask

        thickness = max(1, min(h, w) // 512)
        shift = 4
        cv2.polylines(
            mask, [_shifted_points(points, shift)], isClosed=False,
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
        params: EyeLinerParams,
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

    def __init__(self, params: PartParams, part_name: str):
        indices = _INDICES_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=22,
            params=params,
            landmark_indices=list(indices),
            mask_strategy=TearBagHighlightMaskStrategy(list(indices)),
            blend_strategy=TearBagHighlightBlendStrategy(),
        )


class TearBagShadowPass(MakeupPass):
    """Tear bag shadow — open polyline with thickness dilation."""

    def __init__(self, params: EyeLinerParams, part_name: str):
        indices = _INDICES_MAP[part_name]
        super().__init__(
            pass_type=part_name,
            priority=27,
            params=params,
            landmark_indices=list(indices),
            mask_strategy=TearBagShadowMaskStrategy(list(indices)),
            blend_strategy=TearBagShadowBlendStrategy(),
        )
