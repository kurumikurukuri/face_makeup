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
from face_makeup.core.passes.tearbag import TearBagHighlightPass, TearBagShadowPass
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
    EyeIrisParams,
    TearBagHighlightParams,
    TearBagShadowParams,
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

    _VALID_ZONES = ('full', 'outer', 'center', 'inner', 'inner_center', 'outer_center')

    @staticmethod
    def _validate_zone(zone: str) -> None:
        """
        Validate eyelid zone parameter.

        Args:
            zone: Must be 'full', 'outer', 'center', 'inner',
                  'inner_center', or 'outer_center'.

        Raises:
            ValueError: If zone is not a valid value.
        """
        if zone not in Face._VALID_ZONES:
            raise ValueError(
                f"zone must be one of {Face._VALID_ZONES}, got '{zone}'"
            )

    @staticmethod
    def _eyelid_part(zone: str, side: str) -> str:
        """
        Build an eyeshadow part name from zone and side.

        Args:
            zone: 'full', 'outer', 'center', 'inner',
                  'inner_center', or 'outer_center'.
            side: 'left' or 'right'.

        Returns:
            Part name key, e.g. 'eyelid_left' (zone='full') or
            'eyelid_outer_right'.
        """
        return f"eyelid_{side}" if zone == 'full' else f"eyelid_{zone}_{side}"

    @staticmethod
    def _validate_softness(softness: float) -> None:
        """
        Validate softness value.

        Args:
            softness: Feather width as a face-size ratio 0.0-1.0.
                Larger values produce a wider/softer edge.

        Raises:
            ValueError: If softness is not in valid range.
        """
        if not isinstance(softness, (int, float)) or not 0.0 <= softness <= 1.0:
            raise ValueError(f"softness must be 0.0-1.0, got {softness}")

    @staticmethod
    def _validate_glitter(glitter: float) -> None:
        """
        Validate glitter value.

        Args:
            glitter: Glitter intensity 0.0-1.0.

        Raises:
            ValueError: If glitter is not in valid range.
        """
        if not isinstance(glitter, (int, float)) or not 0.0 <= glitter <= 1.0:
            raise ValueError(f"glitter must be 0.0-1.0, got {glitter}")

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
        softness: float = 0.5
    ) -> Face:
        """
        Add lip makeup.

        Args:
            color: RGB color tuple, e.g. (255, 0, 0) for red.
            opacity: Makeup strength 0.0-1.0.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.

        Returns:
            Self for chaining.

        Examples:
            >>> face.lip(color=(255, 100, 100), opacity=0.5)
        """
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        params = LipsParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            softness=softness
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
        softness: float = 0.6
    ) -> Face:
        """
        Add cheek makeup.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (255, 180, 180) for pink.
            opacity: Makeup strength 0.0-1.0.
            center: Ellipse center position (x, y) in normalized coords.
            radius: Ellipse radius (x, y) in normalized coords.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.

        Returns:
            Self for chaining.

        Examples:
            >>> face.cheek(color=(255, 180, 180), opacity=0.3)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        params = CheekParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            softness=softness,
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
        zone: Literal['full', 'outer', 'center', 'inner',
                       'inner_center', 'outer_center'] = 'full',
        color: Tuple[int, int, int],
        opacity: float = 0.3,
        softness: float = 0.4,
        size: float = 1.0,
        glitter: float = 0.0
    ) -> Face:
        """
        Add eye shadow makeup (band between the lash line and crease line).

        Args:
            side: 'left', 'right', or 'both'.
            zone: Horizontal region of the eyelid to paint.
                'full' (default) = whole width (backward compatible),
                'outer' = 目尻側, 'center' = 中心部, 'inner' = 目頭側.
            color: RGB color tuple, e.g. (180, 130, 150).
            opacity: Makeup strength 0.0-1.0.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.
            size: Band height 0.0-1.0. The upper boundary is lerped from the
                lash line (0.0) toward the crease line (1.0), so a small size
                paints a thin band along the lash line and 1.0 (default)
                covers the whole lid. Call eyelid() multiple times with
                different sizes to layer colors (e.g. a dark rim under a
                lighter full lid).
            glitter: Glitter (lamé) effect intensity 0.0-1.0.
                0.0 = matte only (default). Higher values add sparkle.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyelid(color=(180, 130, 150), opacity=0.3)
            >>> face.eyelid(color=(180, 130, 150), zone='outer')
            >>> face.eyelid(color=(180, 130, 150), glitter=0.5)
            >>> face.eyelid(color=(224, 88, 140)).eyelid(color=(245, 41, 41), size=0.3)
        """
        self._validate_side(side)
        self._validate_zone(zone)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        self._validate_glitter(glitter)
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            softness=softness,
            size=size,
            glitter=glitter
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(EyeShadowPass(params, self._eyelid_part(zone, 'left')))
        if side in ('right', 'both'):
            self._makeup_passes.append(EyeShadowPass(params, self._eyelid_part(zone, 'right')))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye shadow layer (side={side}, zone={zone}, size={size})")
        return self

    def eyeliner(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int],
        opacity: float = 1.0,
        thickness: int = 2,
        softness: float = 0.5
    ) -> Face:
        """
        Add eye liner makeup.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (0, 0, 0) for black.
            opacity: Makeup strength 0.0-1.0.
            thickness: Line thickness 1-10.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.

        Returns:
            Self for chaining.

        Examples:
            >>> face.eyeliner(color=(0, 0, 0), opacity=1.0, thickness=2)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            thickness=thickness,
            softness=softness
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
        softness: float = 0.1,
        hue_shift: bool = False
    ) -> Face:
        """
        Add eye iris (color contact lens) makeup.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (100, 150, 255) for blue.
            opacity: Makeup strength 0.0-1.0.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge. Defaults lower
                than other parts — a contact lens has a physically defined
                edge, so it only needs anti-aliasing, not a cosmetic fade.
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
        self._validate_softness(softness)
        params = EyeIrisParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            softness=softness,
            hue_shift=hue_shift
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(EyeIrisPass(params, 'left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(EyeIrisPass(params, 'right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added eye iris layer (side={side})")
        return self

    def tear_bag_highlight(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int] = (255, 255, 255),
        opacity: float = 0.3,
        softness: float = 0.5,
        size: float = 1.0,
        glitter: float = 0.0,
    ) -> Face:
        """
        Add tear bag (涙袋) highlight — filled polygon on the lower-eyelid bulge.

        Brightens the tear bag area via a screen blend, so a white or pearl
        highlight reads correctly (lightens rather than darkens).

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (255, 255, 255) for white.
            opacity: Highlight strength 0.0-1.0.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.
            size: Tear bag size 0.0-1.0. 0.0 collapses the outer edge onto the
                lash line (no visible tear bag); 1.0 is the full natural ridge.
            glitter: Glitter (lamé) effect intensity 0.0-1.0.
                0.0 = matte highlight only (default). Higher values add
                silver-white sparkle over the highlight.

        Returns:
            Self for chaining.

        Examples:
            >>> face.tear_bag_highlight(color=(255, 255, 255), opacity=0.3, size=0.8)
            >>> face.tear_bag_highlight(color=(255, 255, 255), glitter=0.5)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        self._validate_glitter(glitter)
        params = TearBagHighlightParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            softness=softness,
            size=size,
            glitter=glitter,
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(TearBagHighlightPass(params, 'tear_bag_highlight_left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(TearBagHighlightPass(params, 'tear_bag_highlight_right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added tear bag highlight (side={side}, size={size})")
        return self

    def tear_bag_shadow(
        self,
        *,
        side: Literal['left', 'right', 'both'] = 'both',
        color: Tuple[int, int, int] = (150, 110, 90),
        opacity: float = 0.4,
        thickness: int = 2,
        softness: float = 0.5,
        size: float = 1.0,
    ) -> Face:
        """
        Add tear bag (涙袋) shadow line — open polyline tracing the lerped outer
        edge of the tear bag.

        Args:
            side: 'left', 'right', or 'both'.
            color: RGB color tuple, e.g. (150, 110, 90) for a soft brown.
            opacity: Shadow strength 0.0-1.0.
            thickness: Line thickness (pixels, >= 1).
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.
            size: Shadow position 0.0-1.0. Should match tear_bag_highlight size
                so highlight and shadow track together.

        Returns:
            Self for chaining.

        Examples:
            >>> face.tear_bag_shadow(color=(150, 110, 90), opacity=0.4, size=0.8)
        """
        self._validate_side(side)
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        params = TearBagShadowParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            thickness=thickness,
            softness=softness,
            size=size,
        )
        if side in ('left', 'both'):
            self._makeup_passes.append(TearBagShadowPass(params, 'tear_bag_shadow_left'))
        if side in ('right', 'both'):
            self._makeup_passes.append(TearBagShadowPass(params, 'tear_bag_shadow_right'))
        self._notify_change()
        logger.debug(f"Face {self._face_id}: Added tear bag shadow (side={side}, size={size})")
        return self


    def foundation(
        self,
        *,
        color: Tuple[int, int, int],
        opacity: float = 0.3,
        softness: float = 0.5
    ) -> Face:
        """
        Add foundation (full-face skin tone) makeup.

        Applies a face-oval mask with alpha blending for natural-looking
        skin tone adjustment. The mask excludes eye and lip regions.

        Args:
            color: RGB color tuple matching the desired skin tone.
            opacity: Makeup strength 0.0-1.0.
            softness: Feather width as a face-size ratio (0.0-1.0).
                Larger values produce a wider/softer edge.

        Returns:
            Self for chaining.

        Examples:
            >>> face.foundation(color=(255, 220, 200), opacity=0.3)
        """
        self._validate_color(color)
        self._validate_opacity(opacity)
        self._validate_softness(softness)
        params = PartParams(
            color_params=ColorParams(target_rgb=color),
            opacity=opacity,
            softness=softness
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
