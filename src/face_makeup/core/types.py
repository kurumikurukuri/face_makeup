"""
Data structure definitions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
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
    points_3d: np.ndarray | None = None  # shape: (478, 3), dtype: float32, normalized 3D coordinates

    def to_image_coords(self, img_width: int, img_height: int) -> np.ndarray:
        """Convert to image coordinate system"""
        coords = self.points * np.array([img_width, img_height], dtype=np.float32)
        return np.round(coords).astype(np.int32)


@dataclass
class ColorParams:
    """Color correction parameters"""
    target_rgb: tuple[int, int, int] | None = None  # Directly specified in RGB


# Parameters by facial part
@dataclass
class PartParams:
    """Base class shared by all facial parts"""
    color_params: ColorParams
    opacity: float  # 0..1
    softness: float = 0.5  # Feather width as a face-size ratio (0..1); larger = softer/wider edge. Mapped to an absolute pixel feather via the inter-ocular span.

    # Internal feather calibration — NOT dataclass fields, so they never appear
    # in __init__/__repr__/__eq__ and stay hidden from the user API. Each part
    # overrides them on its own Params subclass; the values here are the shared
    # fallback (consumed as defaults by blend_utils.feather_px_from_softness /
    # soften_mask when a pass does not override).
    _FEATHER_SCALE: ClassVar[float] = 0.04
    _SOFTEN_FLOOR: ClassVar[float] = 0.1

    def __post_init__(self):
        """Validate common parameter bounds."""
        if not isinstance(self.opacity, (int, float)) or not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"opacity must be 0.0-1.0, got {self.opacity}")
        if not isinstance(self.softness, (int, float)) or not 0.0 <= self.softness <= 1.0:
            raise ValueError(f"softness must be 0.0-1.0, got {self.softness}")


@dataclass
class FoundationParams(PartParams):
    """Parameters for foundation (full-face skin tone overlay).

    Currently uses only the base class parameters (color, opacity, softness).
    Defined as a dedicated subclass for consistency with other parts
    and to allow foundation-specific calibration in the future.
    """
    pass


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
    Fadeout effect is achieved via softness (inherited from PartParams).

    Attributes:
        center: Ellipse center offset in range [-1.0, 1.0] for each axis.
        radius: Ellipse radius in range (0.0, 1.0] for each axis.
    """
    # Calibrated via scripts/param_sweep.py. Wider feather than the shared
    # fallback so the broad cheek plateau fades softly.
    _FEATHER_SCALE: ClassVar[float] = 0.15
    _SOFTEN_FLOOR: ClassVar[float] = 0.05

    softness: float = 0.6
    center: tuple[float, float] = (0.0, 0.0)
    radius: tuple[float, float] = (0.5, 0.5)

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
        size: Band height 0.0-1.0. The upper boundary is lerped from the
            lash line (0.0) toward the crease line (1.0), so a small size
            paints a thin band along the lash line and 1.0 (default)
            covers the whole lid.
        glitter: Glitter (lamé) effect intensity 0.0-1.0.
            0.0 = matte only (current behaviour, default).
            Higher values increase glitter particle density/visibility.
        seed: Optional integer seed for deterministic glitter particle
            generation. When None (default), each call uses a fresh
            unpredictable RNG; pass an integer for reproducible output
            (useful for tests and comparison shots).
    """
    # Calibrated via scripts/param_sweep.py. Narrower feather than cheek so
    # the thin lid band keeps a crisp lash-line edge.
    _FEATHER_SCALE: ClassVar[float] = 0.10
    _SOFTEN_FLOOR: ClassVar[float] = 0.05

    softness: float = 0.4
    size: float = 1.0
    glitter: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        """Validate parameter bounds."""
        super().__post_init__()
        if not isinstance(self.size, (int, float)) or not 0.0 <= self.size <= 1.0:
            raise ValueError(f"size must be 0.0-1.0, got {self.size}")
        if not isinstance(self.glitter, (int, float)) or not 0.0 <= self.glitter <= 1.0:
            raise ValueError(f"glitter must be 0.0-1.0, got {self.glitter}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be None or int, got {type(self.seed).__name__}")


@dataclass
class EyeLinerParams(PartParams):
    """Parameters for eyeliner
    Specifies the thickness.
    """
    # Per-part feather calibration (same as PartParams defaults — the thin
    # liner stroke keeps the standard feather width and soften floor).
    _FEATHER_SCALE: ClassVar[float] = 0.04
    _SOFTEN_FLOOR: ClassVar[float] = 0.1

    thickness: int = 2

    def __post_init__(self):
        """Validate parameter bounds."""
        super().__post_init__()
        if not isinstance(self.thickness, int) or self.thickness < 1:
            raise ValueError(f"thickness must be a positive integer, got {self.thickness}")


@dataclass
class TearBagHighlightParams(PartParams):
    """Parameters for tear bag highlight.

    Attributes:
        size: Tear bag size 0.0-1.0. 0.0 = no tear bag (outer edge collapsed
              onto lash line), 1.0 = full natural ridge (default).
        glitter: Glitter (lamé) effect intensity 0.0-1.0.
            0.0 = matte highlight only (default).
            Higher values add silver-white sparkle over the highlight.
        seed: Optional integer seed for deterministic glitter particle
            generation. When None (default), each call uses a fresh
            unpredictable RNG; pass an integer for reproducible output
            (useful for tests and comparison shots).
    """
    # Per-part feather calibration (same as PartParams defaults — tear bag
    # geometry is similar in scale to foundation/lips, so 0.04 / 0.1 works well).
    _FEATHER_SCALE: ClassVar[float] = 0.04
    _SOFTEN_FLOOR: ClassVar[float] = 0.1

    size: float = 1.0
    glitter: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.size, (int, float)) or not 0.0 <= self.size <= 1.0:
            raise ValueError(f"size must be 0.0-1.0, got {self.size}")
        if not isinstance(self.glitter, (int, float)) or not 0.0 <= self.glitter <= 1.0:
            raise ValueError(f"glitter must be 0.0-1.0, got {self.glitter}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be None or int, got {type(self.seed).__name__}")


@dataclass
class TearBagShadowParams(PartParams):
    """Parameters for tear bag shadow.

    Attributes:
        thickness: Line thickness in pixels (>= 1).
        size: Shadow line position 0.0-1.0. Mirrors TearBagHighlightParams.size
              so highlight and shadow track together.
    """
    # Per-part feather calibration (same as PartParams defaults).
    _FEATHER_SCALE: ClassVar[float] = 0.04
    _SOFTEN_FLOOR: ClassVar[float] = 0.1

    thickness: int = 2
    size: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.thickness, int) or self.thickness < 1:
            raise ValueError(f"thickness must be a positive integer, got {self.thickness}")
        if not isinstance(self.size, (int, float)) or not 0.0 <= self.size <= 1.0:
            raise ValueError(f"size must be 0.0-1.0, got {self.size}")


@dataclass
class EyeIrisParams(PartParams):
    """Parameters for eye iris (color contact lens effect).

    Used to apply colored contact lens effect to the iris area.

    Attributes:
        hue_shift: If True, use hue-shift blend (cut→shift→paste).
            Preserves original S/V channels; only H is shifted to target hue.
            If False (default), use legacy HSV replace blend.
    """
    # Calibrated via scripts/param_sweep.py (eyeiris_minimal_fade preset).
    # A contact lens has a physically defined edge, unlike a blended cosmetic
    # such as blush — the feather only needs to anti-alias the raster edge
    # and absorb landmark jitter, not fade the colour into the sclera. Hence
    # a narrow feather_scale, a high soften_floor (near-opaque at the edge),
    # and a lower default softness than the shared PartParams fallback.
    _FEATHER_SCALE: ClassVar[float] = 0.02
    _SOFTEN_FLOOR: ClassVar[float] = 0.95

    softness: float = 0.1
    hue_shift: bool = False
