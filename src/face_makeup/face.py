"""
Face class for makeup application.

Represents a single detected face with its bounding box, landmarks,
and makeup layers. Provides a fluent API for applying makeup.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Literal, Optional, Tuple

from face_makeup.core.passes.cheek import CheekPass
from face_makeup.core.passes.eyeliner import EyelinerPass
from face_makeup.core.passes.eyeiris import EyeIrisPass
from face_makeup.core.passes.eyeshadow import EyeShadowPass
from face_makeup.core.passes.foundation import FoundationPass
from face_makeup.core.passes.lips import LipsPass
from face_makeup.core.pipeline import MakeupPass
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

logger = logging.getLogger(__name__)


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
        change_callback: Optional[Callable[[], None]] = None
    ):
        """
        Initialize face.

        Args:
            face_id: Unique identifier within image.
            bbox: Face bounding box.
            landmarks: Facial landmarks.
            change_callback: Optional callback to invoke when makeup changes (for cache invalidation).
        """
        self._face_id = face_id
        self._bbox = bbox
        self._landmarks = landmarks
        self._change_callback = change_callback
        self._makeup_passes: List[MakeupPass] = []

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
    def makeup_passes(self) -> List[MakeupPass]:
        """Get list of makeup passes (copy)."""
        return self._makeup_passes.copy()

    def lip(
        self,
        *,
        color: Tuple[int, int, int],
        opacity: float = 0.5,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add lip makeup.

        Args:
            color: RGB color tuple, e.g. (255, 0, 0) for red.
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.lip(color=(255, 100, 100), opacity=0.5)
        """
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_blur_strength(blur_strength)
        params = LipsParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            blur_strength=blur_strength
        )
        self._makeup_passes.append(LipsPass(params))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added lip layer")
        return self

    def cheek(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int],
        opacity: float = 0.3,
        center: Tuple[float, float] = (0.0, 0.0),
        radius: Tuple[float, float] = (0.5, 0.5),
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add cheek makeup.

        Args:
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
        """
        self._validate_side(side)
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
            self._makeup_passes.append(CheekPass(params, 'left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(CheekPass(params, 'right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added cheek layer (side={side})")
        return self

    def eyelid(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int],
        opacity: float = 0.3,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye shadow makeup.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (180, 130, 150).
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyelid(color=(180, 130, 150), opacity=0.3)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_blur_strength(blur_strength)
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            blur_strength=blur_strength
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(EyeShadowPass(params, 'eyelid_left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(EyeShadowPass(params, 'eyelid_right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye shadow layer (side={side})")
        return self

    def eyelid_edge(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int],
        opacity: float = 0.3,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye edge (extended area) makeup.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (180, 130, 150).
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyelid_edge(color=(245, 41, 41), opacity=0.3)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_blur_strength(blur_strength)
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            blur_strength=blur_strength
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(EyeShadowPass(params, 'eyelid_edge_left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(EyeShadowPass(params, 'eyelid_edge_right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye edge layer (side={side})")
        return self

    def eyeliner(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int],
        opacity: float = 1.0,
        thickness: int = 2,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add eye liner makeup.

        Args:
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
        """
        self._validate_side(side)
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
            self._makeup_passes.append(EyelinerPass(params, 'left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(EyelinerPass(params, 'right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye liner layer (side={side})")
        return self

    def eyeiris(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int],
        opacity: float = 0.5,
        blur_strength: float = 0.5,
        hue_shift: bool = False
    ) -> Face:
        """
        Add eye iris (color contact lens) makeup.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (100, 150, 255) for blue.
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0). Boundary retains
                this fraction of opacity; center stays 1.0.
            hue_shift: If True, use hue-shift blend: shifts only the H
                channel to target hue while preserving original S/V.
                If False (default), use legacy HSV replace blend.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyeiris(color=(100, 150, 255), opacity=0.5)
            >>> face.eyeiris(color=(100, 150, 255), opacity=0.5, hue_shift=True)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_blur_strength(blur_strength)
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            blur_strength=blur_strength,
            hue_shift=hue_shift
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(EyeIrisPass(params, 'left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(EyeIrisPass(params, 'right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye iris layer (side={side})")
        return self

    def foundation(
        self,
        *,
        color: Tuple[int, int, int],
        opacity: float = 0.3,
        blur_strength: float = 0.5
    ) -> Face:
        """
        Add foundation (full-face skin tone) makeup.

        Applies a face-oval mask with alpha blending for natural-looking
        skin tone adjustment. The mask excludes eye and lip regions.

        Args:
            color: RGB color tuple matching the desired skin tone.
            opacity: Makeup strength 0.0-1.0.
            blur_strength: Edge attenuation factor (0.0-1.0).

        Returns:
            Self for chaining.

        Examples:
            >>> face.foundation(color=(255, 220, 200), opacity=0.3)
        """
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_blur_strength(blur_strength)
        params = PartParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            blur_strength=blur_strength
        )
        self._makeup_passes.append(FoundationPass(params))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added foundation")
        return self

    def clear(self) -> Face:
        """
        Clear all makeup layers.

        Returns:
            Self for chaining.
        """
        self._makeup_passes.clear()
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Cleared all makeup layers")
        return self

    def set_change_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """
        Set callback invoked when makeup layers change.

        Args:
            callback: Callable with no arguments, or None to clear.
        """
        self._change_callback = callback

    def _notify_change(self) -> None:
        """Notify callback that makeup has changed."""
        if self._change_callback is not None:
            self._change_callback()
