"""
Face class for makeup application.

Represents a single detected face with its bounding box, landmarks,
and makeup layers. Provides a fluent API for applying makeup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np

from face_makeup.core.types import (
    FaceBBox,
    FaceLandmarks,
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams,
    ColorParams,
    PartParams,
    EyeIrisParams
)

if TYPE_CHECKING:
    from face_makeup.context import ImageContext

logger = logging.getLogger(__name__)


@dataclass
class MakeupLayer:
    """Base class for makeup layers."""
    part_name: str
    params: PartParams
    mask: Optional[np.ndarray] = None  # Soft mask (float32 0-1)


@dataclass
class LipLayer(MakeupLayer):
    """Lip makeup layer."""
    pass


@dataclass
class CheekLayer(MakeupLayer):
    """Cheek makeup layer."""
    cheek_side: str = field(default='left')  # 'left' or 'right'


@dataclass
class EyeShadowLayer(MakeupLayer):
    """Eye shadow makeup layer."""
    pass


@dataclass
class EyeLinerLayer(MakeupLayer):
    """Eye liner makeup layer."""
    pass

@dataclass
class EyeIrisLayer(MakeupLayer):
    """Eye iris makeup layer."""
    pass




class Face:
    """
    Single face data and operations.

    Does NOT hold image reference - uses bbox to locate in ImageContext.
    Provides fluent API for makeup application.

    Example:
        >>> face = ctx.faces[0]
        >>> face.lip(lip_params).cheek(cheek_params).eyelid(eye_params)
        >>> result = ctx.render()
    """

    @staticmethod
    def _validate_color(color: Tuple[int, int, int]) -> None:
        """
        Validate RGB color tuple.

        Args:
            color: RGB color tuple (R, G, B).

        Raises:
            ValueError: If color is not a valid RGB tuple.
        """
        if not isinstance(color, tuple) or len(color) != 3:
            raise ValueError(f"Color must be (R, G, B) tuple, got {color}")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError(f"Color values must be integers 0-255, got {color}")

    @staticmethod
    def _validate_opacity(opacity: float) -> None:
        """
        Validate opacity value.

        Args:
            opacity: Opacity value 0.0-1.0.

        Raises:
            ValueError: If opacity is not in valid range.
        """
        if not isinstance(opacity, (int, float)) or not 0.0 <= opacity <= 1.0:
            raise ValueError(f"Opacity must be 0.0-1.0, got {opacity}")

    _VALID_SIDES = ('left', 'right', 'both')

    @staticmethod
    def _validate_side(side: str) -> None:
        """
        Validate side parameter.

        Args:
            side: Must be 'left', 'right', or 'both'.

        Raises:
            ValueError: If side is not a valid value.
        """
        if side not in Face._VALID_SIDES:
            raise ValueError(
                f"side must be one of {Face._VALID_SIDES}, got '{side}'"
            )

    @staticmethod
    def _validate_blur_strength(blur_strength: float) -> None:
        """
        Validate blur_strength value.

        Args:
            blur_strength: Blur strength value 0.0-1.0.

        Raises:
            ValueError: If blur_strength is not in valid range.
        """
        if not isinstance(blur_strength, (int, float)) or not 0.0 <= blur_strength <= 1.0:
            raise ValueError(f"blur_strength must be 0.0-1.0, got {blur_strength}")

    def __init__(
        self,
        face_id: int,
        bbox: FaceBBox,
        landmarks: FaceLandmarks,
        context: Optional[ImageContext] = None
    ):
        """
        Initialize face.

        Args:
            face_id: Unique identifier within image.
            bbox: Face bounding box.
            landmarks: Facial landmarks.
            context: Parent ImageContext (weak reference for cache invalidation).
        """
        self._face_id = face_id
        self._bbox = bbox
        self._landmarks = landmarks
        self._context = context
        self._makeup_layers: List[MakeupLayer] = []

    @property
    def face_id(self) -> int:
        """Get face identifier."""
        return self._face_id

    @property
    def bbox(self) -> FaceBBox:
        """Get face bounding box."""
        return self._bbox

    @property
    def landmarks(self) -> FaceLandmarks:
        """Get facial landmarks."""
        return self._landmarks

    @property
    def makeup_layers(self) -> List[MakeupLayer]:
        """Get list of makeup layers (copy)."""
        return self._makeup_layers.copy()

    def lip(
        self,
        params: Optional[LipsParams] = None,
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 0.5,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add lip makeup.

        Args:
            params: Lip parameters (alias).
            color: RGB color tuple, e.g. (255, 0, 0) for red.
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.lip(color=(255, 100, 100), opacity=0.5)
            >>> face.lip(LipsParams(...))  # alias
        """
        if params is None:
            if color is None:
                raise ValueError("Either 'params' or 'color' must be provided")
            self._validate_color(color)
            self._validate_opacity(opacity)
            self._validate_blur_strength(blur_strength)
            params = LipsParams(
                color_params=ColorParams(target_rgb=color),
                opacity=opacity,
                blur_strength=blur_strength
            )
        self._makeup_layers.append(LipLayer(
            part_name='lips',
            params=params
        ))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added lip layer")
        return self

    def cheek(
        self,
        params: Optional[CheekParams] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 0.3,
        center: Tuple[float, float] = (0.0, 0.0),
        radius: Tuple[float, float] = (0.5, 0.5),
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add cheek makeup.

        Args:
            params: Cheek parameters (alias).
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (255, 180, 180) for pink.
            opacity: Makeup strength 0.0-1.0.
            center: Ellipse center position (x, y) in normalized coords.
            radius: Ellipse radius (x, y) in normalized coords.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.cheek(color=(255, 180, 180), opacity=0.3)
            >>> face.cheek(CheekParams(...))  # alias
        """
        self._validate_side(side)
        if params is None:
            if color is None:
                raise ValueError("Either 'params' or 'color' must be provided")
            self._validate_color(color)
            self._validate_opacity(opacity)
            self._validate_blur_strength(blur_strength)
            params = CheekParams(
                color_params=ColorParams(target_rgb=color),
                opacity=opacity,
                blur_strength=blur_strength,
                center=center,
                radius=radius
            )
        if side in ('left', 'both'):
            self._makeup_layers.append(CheekLayer(
                part_name='cheek_left',
                params=params,
                cheek_side='left'
            ))
        if side in ('right', 'both'):
            self._makeup_layers.append(CheekLayer(
                part_name='cheek_right',
                params=params,
                cheek_side='right'
            ))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added cheek layer (side={side})")
        return self

    def eyelid(
        self,
        params: Optional[EyeShadowParams] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 0.3,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye shadow makeup.

        Args:
            params: Eye shadow parameters (alias).
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (180, 130, 150).
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyelid(color=(180, 130, 150), opacity=0.3)
            >>> face.eyelid(EyeShadowParams(...))  # alias
        """
        self._validate_side(side)
        if params is None:
            if color is None:
                raise ValueError("Either 'params' or 'color' must be provided")
            self._validate_color(color)
            self._validate_opacity(opacity)
            self._validate_blur_strength(blur_strength)
            params = EyeShadowParams(
                color_params=ColorParams(target_rgb=color),
                opacity=opacity,
                blur_strength=blur_strength
            )
        if side in ('left', 'both'):
            self._makeup_layers.append(EyeShadowLayer(
                part_name='eyelid_left',
                params=params
            ))
        if side in ('right', 'both'):
            self._makeup_layers.append(EyeShadowLayer(
                part_name='eyelid_right',
                params=params
            ))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye shadow layer (side={side})")
        return self

    def eyelid_edge(
        self,
        params: Optional[EyeShadowParams] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 0.3,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye edge (extended area) makeup.

        Args:
            params: Eye shadow parameters (alias).
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (180, 130, 150).
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyelid_edge(color=(245, 41, 41), opacity=0.3)
            >>> face.eyelid_edge(EyeShadowParams(...))  # alias
        """
        self._validate_side(side)
        if params is None:
            if color is None:
                raise ValueError("Either 'params' or 'color' must be provided")
            self._validate_color(color)
            self._validate_opacity(opacity)
            self._validate_blur_strength(blur_strength)
            params = EyeShadowParams(
                color_params=ColorParams(target_rgb=color),
                opacity=opacity,
                blur_strength=blur_strength
            )
        if side in ('left', 'both'):
            self._makeup_layers.append(EyeShadowLayer(
                part_name='eyelid_edge_left',
                params=params
            ))
        if side in ('right', 'both'):
            self._makeup_layers.append(EyeShadowLayer(
                part_name='eyelid_edge_right',
                params=params
            ))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye edge layer (side={side})")
        return self

    def eyeliner(
        self,
        params: Optional[EyeLinerParams] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 1.0,
        thickness: int = 2,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye liner makeup.

        Args:
            params: Eye liner parameters (alias).
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (0, 0, 0) for black.
            opacity: Makeup strength 0.0-1.0.
            thickness: Line thickness 1-10.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyeliner(color=(0, 0, 0), opacity=1.0, thickness=2)
            >>> face.eyeliner(EyeLinerParams(...))  # alias
        """
        self._validate_side(side)
        if params is None:
            if color is None:
                raise ValueError("Either 'params' or 'color' must be provided")
            self._validate_color(color)
            self._validate_opacity(opacity)
            self._validate_blur_strength(blur_strength)
            params = EyeLinerParams(
                color_params=ColorParams(target_rgb=color),
                opacity=opacity,
                thickness=thickness,
                blur_strength=blur_strength
            )
        if side in ('left', 'both'):
            self._makeup_layers.append(EyeLinerLayer(
                part_name='eyeliner_left',
                params=params
            ))
        if side in ('right', 'both'):
            self._makeup_layers.append(EyeLinerLayer(
                part_name='eyeliner_right',
                params=params
            ))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye liner layer (side={side})")
        return self

    def eyeiris(self,
        params: Optional[EyeIrisParams] = None,
        side: Literal['left', 'right', 'both'] = 'both',
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 0.5,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye iris (color contact lens) makeup.

        Args:
            params: Eye iris parameters (alias).
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (100, 150, 255) for blue.
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyeiris(color=(100, 150, 255), opacity=0.5)
            >>> face.eyeiris(EyeIrisParams(...))  # alias
        """
        self._validate_side(side)
        if params is None:
            if color is None:
                raise ValueError("Either 'params' or 'color' must be provided")
            self._validate_color(color)
            self._validate_opacity(opacity)
            self._validate_blur_strength(blur_strength)
            params = EyeIrisParams(
                color_params=ColorParams(target_rgb=color),
                opacity=opacity,
                blur_strength=blur_strength
            )
        if side in ('left', 'both'):
            self._makeup_layers.append(EyeIrisLayer(
                part_name='eyeiris_left',
                params=params
            ))
        if side in ('right', 'both'):
            self._makeup_layers.append(EyeIrisLayer(
                part_name='eyeiris_right',
                params=params
            ))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye iris layer (side={side})")
        return self
            
    def clear(self) -> Face:
        """
        Clear all makeup layers.

        Returns:
            Self for chaining.
        """
        self._makeup_layers.clear()
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Cleared all makeup layers")
        return self

    def _notify_change(self) -> None:
        """Notify context that makeup has changed."""
        if self._context is not None:
            self._context.invalidate_cache()

    def _set_context(self, context: ImageContext) -> None:
        """Set parent context (called by ImageContext)."""
        self._context = context
