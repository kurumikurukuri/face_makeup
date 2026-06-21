"""
Data structure definitions
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FaceBBox:
    """Face detection result"""
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass
class FaceLandmarks:
    """Landmarks (normalized coordinates)"""
    points: np.ndarray  # shape: (N, 2), dtype: float32, range: 0..1. N depends on max custom landmark ID (e.g. 1708 when max ID = 1707).
    points_3d: Optional[np.ndarray] = None  # shape: (478, 3), dtype: float32, normalized 3D coordinates

    def to_image_coords(self, img_width: int, img_height: int) -> np.ndarray:
        """Convert to image coordinate system"""
        coords = self.points * np.array([img_width, img_height], dtype=np.float32)
        return np.round(coords).astype(np.int32)


@dataclass
class ColorParams:
    """Color correction parameters"""
    target_rgb: Optional[Tuple[int, int, int]] = None  # Directly specified in RGB


# Parameters by facial part
@dataclass
class PartParams:
    """Base class shared by all facial parts"""
    color_params: ColorParams
    opacity: float  # 0..1
    blur_strength: float = 0.5  # Edge attenuation (0..1). Boundary retains this fraction of opacity; center stays 1.0.

    def __post_init__(self):
        """Validate common parameter bounds."""
        if not isinstance(self.opacity, (int, float)) or not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be 0.0-1.0, got {self.opacity}")
        if not isinstance(self.blur_strength, (int, float)) or not 0.0 <= self.blur_strength <= 1.0:
            raise ValueError(f"blur_strength must be 0.0-1.0, got {self.blur_strength}")


@dataclass
class LipsParams(PartParams):
    """Parameters for lips

    Additional parameters such as gloss effects or contour enhancement
    can be defined in the future.
    Currently, only the parameters from the base class are used.
    """
    pass


@dataclass
class CheekParams(PartParams):
    """Parameters for cheeks.

    Parameters for generating an elliptical mask.
    Fadeout effect is achieved via blur_strength (inherited from PartParams).

    Attributes:
        center: Ellipse center offset in range [-1.0, 1.0] for each axis.
        radius: Ellipse radius in range (0.0, 1.0] for each axis.
    """
    center: Tuple[float, float] = (0.0, 0.0)
    radius: Tuple[float, float] = (0.5, 0.5)

    def __post_init__(self):
        """Validate parameter bounds."""
        super().__post_init__()
        if not all(-1.0 <= c <= 1.0 for c in self.center):
            raise ValueError(f"center values must be in range [-1.0, 1.0], got {self.center}")
        if not all(0.0 < r <= 1.0 for r in self.radius):
            raise ValueError(f"radius values must be in range (0.0, 1.0], got {self.radius}")


@dataclass
class EyeShadowParams(PartParams):
    """Parameters for eyeshadow.

    Attributes:
        glitter: Glitter (lamé) effect intensity 0.0-1.0.
            0.0 = matte only (current behaviour, default).
            Higher values increase glitter particle density/visibility.
        seed: Optional integer seed for deterministic glitter particle
            generation. When None (default), each call uses a fresh
            unpredictable RNG; pass an integer for reproducible output
            (useful for tests and comparison shots).
    """

    glitter: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        """Validate parameter bounds."""
        super().__post_init__()
        if not isinstance(self.glitter, (int, float)) or not 0.0 <= self.glitter <= 1.0:
            raise ValueError(f"glitter must be 0.0-1.0, got {self.glitter}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be None or int, got {type(self.seed).__name__}")


@dataclass
class EyeLinerParams(PartParams):
    """Parameters for eyeliner
    Specifies the thickness.
    """
    thickness: int = 2

    def __post_init__(self):
        """Validate parameter bounds."""
        super().__post_init__()
        if not isinstance(self.thickness, int) or self.thickness < 1:
            raise ValueError(f"thickness must be a positive integer, got {self.thickness}")


@dataclass
class EyeIrisParams(PartParams):
    """Parameters for eye iris (color contact lens effect).

    Used to apply colored contact lens effect to the iris area.

    Attributes:
        hue_shift: If True, use hue-shift blend (cut→shift→paste).
            Preserves original S/V channels; only H is shifted to target hue.
            If False (default), use legacy HSV replace blend.
    """
    hue_shift: bool = False
