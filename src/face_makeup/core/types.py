"""
Data structure definitions
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from typing import Type


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
    points: np.ndarray  # shape: (478, 2), dtype: float32, range: 0..1
    points_3d: Optional[np.ndarray] = None  # shape: (478, 3), dtype: float32, normalized 3D coordinates

    def to_image_coords(self, img_width: int, img_height: int) -> np.ndarray:
        """Convert to image coordinate system"""
        coords = self.points.copy()
        coords[:, 0] *= img_width
        coords[:, 1] *= img_height
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
    """Parameters for eyeshadow

    Gradient functionality is planned to be added in the future.
    Currently, only the parameters from the base class are used.
    """
    pass


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
    Currently uses base class parameters only.
    """
    pass


# PartParams subclass -> priority mapping
# Higher values are drawn on top (higher priority)
PARAMS_PRIORITY: Dict["Type[PartParams]", int] = {
    CheekParams: 0,        # Bottom layer
    LipsParams: 10,
    EyeShadowParams: 20,   # eyelid, eyelid_edge
    EyeLinerParams: 30,
    EyeIrisParams: 40,     # Top layer
}