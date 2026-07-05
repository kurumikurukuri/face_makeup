"""
Foundation makeup pass — full-face skin tone overlay.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from face_makeup.constants import (
    EYELINER_LOWER_LEFT_INDICES,
    EYELINER_LOWER_RIGHT_INDICES,
    EYELINER_UPPER_LEFT_INDICES,
    EYELINER_UPPER_RIGHT_INDICES,
    FACE_OVAL_INDICES,
    LIPS_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.types import PartParams

# Regions always excluded from the foundation mask.
# Eyelid: eye shadow area.  Eyeliner upper+lower: eye interior.
_EXCLUSION_REGIONS = [
    EYELINER_UPPER_LEFT_INDICES + EYELINER_LOWER_LEFT_INDICES,
    EYELINER_UPPER_RIGHT_INDICES + EYELINER_LOWER_RIGHT_INDICES,
    LIPS_INDICES["upper"],
    LIPS_INDICES["lower"],
]

# Every pass_type string Face._eyelid_part() can produce (zone x side), so
# foundation excludes eyelid makeup regardless of which zone the caller
# picked. The renderer only subtracts exclusion_passes entries that are
# actually present in a given render (see renderer.py), so listing a
# zone/side combination the caller didn't add is a harmless no-op.
_EYELID_EXCLUSION_PASSES = [
    f"eyelid_{side}" for side in ("left", "right")
] + [
    f"eyelid_{zone}_{side}"
    for zone in ("outer", "center", "inner")
    for side in ("left", "right")
]


class FoundationMaskStrategy(MaskStrategy):
    """Face oval convex hull mask with Gaussian blur edge softening.

    Always subtracts the eye interior (eyeliner upper+lower region) and
    lip areas from the mask, so foundation never bleeds into the eyes or
    lips, regardless of which other passes are applied. Eyelid exclusion
    is handled separately at the pipeline level via
    ``FoundationPass.exclusion_passes`` — it only takes effect when the
    eyelid passes are part of the same render.
    """

    def __init__(self, blur_kernel: tuple[int, int] = (31, 31)):
        self._blur_kernel = blur_kernel

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

        # Base mask: face oval convex hull
        oval_points = coords[FACE_OVAL_INDICES]
        hull = cv2.convexHull(oval_points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        # Subtract eye and lip regions so foundation never covers them.
        # Filter to valid indices so tests with small coordinate arrays work.
        n_coords = len(coords)
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        shift = 4
        scale = 1 << shift
        for indices in _EXCLUSION_REGIONS:
            valid = [i for i in indices if i < n_coords]
            if len(valid) < 3:
                continue
            points = coords[valid]
            points_shifted = (points * scale).astype(np.int32)
            cv2.fillPoly(exclusion_mask, [points_shifted], 255, shift=shift)

        mask = cv2.subtract(mask, exclusion_mask)
        mask = cv2.GaussianBlur(mask, self._blur_kernel, 0)
        return mask


class FoundationBlendStrategy(BlendStrategy):
    """Linear alpha blend between original image and target color."""

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: PartParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: object | None = None,
    ) -> np.ndarray:
        target_color = np.array(params.color_params.target_rgb, dtype=np.float32) / 255.0
        alpha = mask[:, :, np.newaxis] * params.opacity
        image_float = image.astype(np.float32) / 255.0
        result = image_float * (1.0 - alpha) + target_color * alpha
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def blur_kernel_from_strength(blur_strength: float) -> tuple[int, int]:
    """Map a 0.0-1.0 blur_strength to an odd-sized Gaussian kernel.

    0.5 (the ``PartParams`` default) reproduces the original fixed
    (31, 31) kernel, so existing callers see no behaviour change.
    """
    size = max(1, round(blur_strength * 60))
    if size % 2 == 0:
        size += 1
    return (size, size)


class FoundationPass(MakeupPass):
    """Full-face foundation with skin-tone overlay.

    The mask strategy always carves out the eye interior (eyeliner
    region) and lip areas so foundation never bleeds into eyes or lips.
    Eyelid regions, in any zone, are additionally excluded via
    ``exclusion_passes``, but only take effect when the corresponding
    pass is present in the same render.
    """

    def __init__(
        self,
        params: PartParams,
        blur_kernel: tuple[int, int] | None = None,
    ):
        if blur_kernel is None:
            blur_kernel = blur_kernel_from_strength(params.blur_strength)
        super().__init__(
            pass_type="foundation",
            priority=0,
            params=params,
            landmark_indices=list(FACE_OVAL_INDICES),
            mask_strategy=FoundationMaskStrategy(blur_kernel=blur_kernel),
            blend_strategy=FoundationBlendStrategy(),
            exclusion_passes=_EYELID_EXCLUSION_PASSES + ["lips"],
        )
