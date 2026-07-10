"""
Core pipeline abstractions: MakeupPass, MaskStrategy, BlendStrategy.

A MakeupPass bundles a mask generation strategy, a blend strategy,
parameters, priority, and landmark-based clip/exclusion declarations
into one composable unit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from face_makeup.core.types import FaceLandmarks, PartParams


class MaskStrategy(ABC):
    """Generate a (H, W) soft mask with the fade baked in.

    Subclasses own the full geometry plus its edge softening and return the
    final soft mask — float32 in [0, 1] — so the renderer can apply
    exclusions to the softened mask before blending.

    Subclasses may override to accept optional context parameters
    (image, landmarks, segmenter) for advanced mask generation
    such as rule-based edge correction or 3D projection.
    """

    @abstractmethod
    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: FaceLandmarks | None = None,
        segmenter: object | None = None,
    ) -> np.ndarray:
        """Return (H, W) soft mask, float32 in [0, 1] (fade already applied)."""
        ...


class BlendStrategy(ABC):
    """Apply color blending to a masked region of an image.

    Subclasses may override to accept optional context parameters
    (blender, segmenter, landmarks) for advanced blending such as
    3D-projected cheek blending or hue-shift iris blending.
    """

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: PartParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB blended result (new array, input unchanged)."""
        ...


@dataclass
class MakeupPass:
    """A single makeup operation: mask + blend + priority.

    Attributes:
        pass_type: Unique identifier for this pass (e.g. "foundation", "lips").
        priority: Render order; lower values are drawn first (under other layers).
        params: Color, opacity, blur_strength, and part-specific fields.
        landmark_indices: MediaPipe landmark indices for mask generation.
        mask_strategy: Strategy that returns the final soft mask (float32 0-1,
            geometry + fade baked in).
        blend_strategy: Strategy that applies pure colour to the ready soft mask.
        exclusion_landmarks: Landmark-index groups whose feathered polygons are
            subtracted from this pass's soft mask (``soft * (1 - exclusion)``).
            Always applied, regardless of which other passes are present.
            E.g. eye interiors, eyelids, and lips for foundation.
        clip_landmarks: Landmark-index groups whose feathered polygons clip
            (AND) the mask — the result is restricted to inside these regions.
            E.g. eye opening contour for eye iris (colour contact lens).
    """

    pass_type: str
    priority: int
    params: PartParams
    landmark_indices: List[int] | Dict[str, List]
    mask_strategy: Optional[MaskStrategy] = None
    blend_strategy: Optional[BlendStrategy] = None
    exclusion_landmarks: List[List[int]] = field(default_factory=list)
    clip_landmarks: List[List[int]] = field(default_factory=list)
