"""
Foundation makeup pass — full-face skin tone overlay.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from face_makeup.constants import (
    EYELID_LEFT_INDICES,
    EYELID_RIGHT_INDICES,
    EYELINER_LOWER_LEFT_INDICES,
    EYELINER_LOWER_RIGHT_INDICES,
    EYELINER_UPPER_LEFT_INDICES,
    EYELINER_UPPER_RIGHT_INDICES,
    FACE_OVAL_INDICES,
    LIPS_INDICES,
)
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.types import PartParams

# Landmark groups always excluded from the foundation mask by the renderer
# via ``exclusion_landmarks`` (soft * (1 - exclusion) = AND NOT).  Each
# entry is a list of landmark indices whose polygon is subtracted from the
# foundation soft mask after the fade, so the face-oval distance field is
# never corrupted by exclusion holes.
#
# Groups:
#   0-1   Eye interiors (eyeliner upper+lower) — prevent foundation bleed
#         into the eyes regardless of which other passes are applied.
#   2-3   Full eyelid regions (crease + lash) — keep foundation off the
#         eyelids; always excluded, not conditional on eyeshadow presence.
#   4-5   Upper / lower lip — keep foundation off the lips; always excluded,
#         not conditional on a lips pass being present.
_EXCLUSION_REGIONS: List[List[int]] = [
    EYELINER_UPPER_LEFT_INDICES + EYELINER_LOWER_LEFT_INDICES,
    EYELINER_UPPER_RIGHT_INDICES + EYELINER_LOWER_RIGHT_INDICES,
    list(EYELID_LEFT_INDICES),
    list(EYELID_RIGHT_INDICES),
    list(LIPS_INDICES["upper"]),
    list(LIPS_INDICES["lower"]),
]


class FoundationMaskStrategy(MaskStrategy):
    """Face oval convex hull mask with Gaussian blur edge softening.

    Exclusions (eye interiors, eyelids, lips) are applied by the renderer
    via ``FoundationPass.exclusion_landmarks`` — the mask strategy itself
    only defines the face-oval shape.
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

    Eye interiors, full eyelid regions, and lips are always carved out
    via ``exclusion_landmarks`` (landmark polygons subtracted by the
    renderer after the fade), regardless of which other passes are
    present in the same render.
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
            exclusion_landmarks=_EXCLUSION_REGIONS,
        )
