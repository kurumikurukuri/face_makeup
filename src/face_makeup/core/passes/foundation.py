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
from face_makeup.core.blend.blend_utils import feather_px_from_softness, soften_mask
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
    """Face oval convex hull mask with distance-based edge softening.

    Generates the base face-oval geometry with soft edges.  Exclusion of
    eye and lip regions is applied by the renderer via ``exclusion_landmarks``
    so that the distance-based fade is computed on the full geometry before
    any holes are carved.
    """

    def __init__(self, softness: float = 0.5):
        self._softness = softness

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

        # Distance-based feather using the shared soften_mask path — consistent
        # edge profile with every other pass.
        feather_px = feather_px_from_softness(self._softness, coords, image_shape)
        return soften_mask(mask, feather_px)


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


class FoundationPass(MakeupPass):
    """Full-face foundation with skin-tone overlay.

    Eye interiors, full eyelid regions, and lips are always carved out
    by the renderer via ``exclusion_landmarks`` after the fade, regardless
    of which other passes are present in the same render.
    """

    def __init__(self, params: PartParams):
        super().__init__(
            pass_type="foundation",
            priority=0,
            params=params,
            landmark_indices=list(FACE_OVAL_INDICES),
            mask_strategy=FoundationMaskStrategy(softness=params.softness),
            blend_strategy=FoundationBlendStrategy(),
            exclusion_landmarks=_EXCLUSION_REGIONS,
        )
