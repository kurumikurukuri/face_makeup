"""
Core pipeline abstractions: MakeupPass, MaskStrategy, BlendStrategy.

A MakeupPass bundles a mask generation strategy, a blend strategy,
parameters, priority, and dependency declarations into one composable unit.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from face_makeup.core.types import FaceLandmarks, PartParams


class MaskStrategy(ABC):
    """Generate a (H, W) uint8 mask from landmark coordinates.

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
        """Return (H, W) uint8 mask (0 or 255)."""
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
        """Return (H, W, 3) uint8 BGR blended result (new array, input unchanged)."""
        ...


@dataclass
class MakeupPass:
    """A single makeup operation: mask + blend + priority.

    Attributes:
        pass_type: Unique identifier for this pass (e.g. "foundation", "lips").
        priority: Render order; lower values are drawn first (under other layers).
        mask_strategy: Strategy that generates a uint8 mask from landmarks.
        blend_strategy: Strategy that applies color to the masked region.
        params: Color, opacity, blur_strength, and part-specific fields.
        landmark_indices: MediaPipe landmark indices for mask generation.
        exclusion_passes: pass_type names whose masks are subtracted from this
            pass's mask during Phase 1 of the two-phase pipeline.
    """

    pass_type: str
    priority: int
    params: PartParams
    landmark_indices: List[int] | Dict[str, List]
    mask_strategy: Optional[MaskStrategy] = None
    blend_strategy: Optional[BlendStrategy] = None
    exclusion_passes: List[str] = field(default_factory=list)


def topological_sort(passes: List[MakeupPass]) -> List[MakeupPass]:
    """Sort passes so that exclusion sources precede dependent passes.

    Passes with no exclusions come first.  Each subsequent pass is only
    emitted once all of its ``exclusion_passes`` have been produced.

    Exclusion references to pass_types that don't exist in the input are
    silently ignored (optional exclusion — if the source isn't applied,
    there's nothing to subtract).

    Raises:
        ValueError: If the exclusion graph contains an unresolvable cycle.
    """
    pass_types = {p.pass_type for p in passes}
    produced: set[str] = set()
    ordered: List[MakeupPass] = []
    remaining = list(passes)

    while remaining:
        for p in list(remaining):
            # Only consider exclusions that reference existing passes
            required = [e for e in p.exclusion_passes if e in pass_types]
            if all(e in produced for e in required):
                ordered.append(p)
                produced.add(p.pass_type)
                remaining.remove(p)
                break
        else:
            remaining_types = [p.pass_type for p in remaining]
            raise ValueError(
                f"Unresolvable exclusion dependency in: {remaining_types}"
            )

    return ordered
