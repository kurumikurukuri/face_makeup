"""
Internal rendering engine for makeup composition.

Handles composition of makeup layers onto images.
Not exposed in public API - used by ImageContext.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np

from src.face_makeup.core.segmenter import FaceSegmenter
from src.face_makeup.core.makeup_blender import MakeupBlender
from src.face_makeup.core.types import (
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams
)

if TYPE_CHECKING:
    from src.face_makeup.face import Face, MakeupLayer, CheekLayer
    from src.face_makeup.core.debug_visualizer import DebugVisualizer

logger = logging.getLogger(__name__)


class Renderer:
    """
    Internal rendering engine.

    Handles composition of makeup layers onto images.
    """

    def __init__(
        self,
        segmenter: FaceSegmenter,
        blender: MakeupBlender,
        visualizer: Optional[DebugVisualizer] = None
    ):
        """
        Initialize renderer with processing components.

        Args:
            segmenter: Face segmenter for mask generation.
            blender: Makeup blender for color application.
            visualizer: Optional debug visualizer.
        """
        self._segmenter = segmenter
        self._blender = blender
        self._visualizer = visualizer

    def render_face(
        self,
        image: np.ndarray,
        face: Face,
        debug: bool = False,
        session_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Render all makeup layers for a single face.

        Args:
            image: Current image state.
            face: Face with makeup layers.
            debug: Generate debug output.
            session_id: Session ID for debug files.

        Returns:
            Image with face makeup applied.
        """
        if not face.makeup_layers:
            logger.debug(f"Face {face.face_id}: No makeup layers to render")
            return image

        session_id = session_id or self._generate_session_id()
        face_session = f"{session_id}_face{face.face_id}"

        # Get unique part names from layers
        part_names = list(set(layer.part_name for layer in face.makeup_layers))

        # Generate hard masks
        hard_masks = self._generate_masks(image.shape[:2], face, part_names)

        if debug and self._visualizer:
            self._visualizer.save_masks(image, hard_masks, face_session, "hard")

        # Refine masks to soft masks
        soft_masks = self._refine_masks(image, hard_masks, face)

        if debug and self._visualizer:
            self._visualizer.save_masks(image, soft_masks, face_session, "soft")

        # Apply each layer
        result = image.copy()
        for layer in face.makeup_layers:
            if layer.part_name not in soft_masks:
                logger.warning(f"No mask for part: {layer.part_name}")
                continue

            mask = soft_masks[layer.part_name]
            result = self._apply_layer(result, mask, layer, face)

            if debug and self._visualizer:
                logger.debug(f"Applied layer: {layer.part_name}")

        return result

    def compose(
        self,
        image: np.ndarray,
        faces: List[Face],
        debug: bool = False,
        session_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Compose all faces' makeup onto image.

        Args:
            image: Original image.
            faces: List of faces with makeup.
            debug: Generate debug output.
            session_id: Session ID for debug files.

        Returns:
            Final composed image.
        """
        session_id = session_id or self._generate_session_id()

        if debug and self._visualizer:
            self._visualizer.save_original(image, session_id)

        result = image.copy()

        for face in faces:
            if not face.makeup_layers:
                continue

            if debug and self._visualizer:
                self._visualizer.save_bbox(image, face.bbox, f"{session_id}_face{face.face_id}")
                self._visualizer.save_landmarks(image, face.landmarks, f"{session_id}_face{face.face_id}")

            result = self.render_face(result, face, debug, session_id)

        if debug and self._visualizer:
            self._visualizer.save_result(image, result, session_id)

        logger.debug(f"Composed {len(faces)} faces")
        return result

    def _generate_masks(
        self,
        image_shape: tuple,
        face: Face,
        parts: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Generate hard masks for face parts.

        Args:
            image_shape: (H, W) tuple.
            face: Face with landmarks.
            parts: List of part names.

        Returns:
            Dictionary of part_name -> hard mask (uint8 0/255).
        """
        return self._segmenter.segment(
            image_shape,
            face.landmarks,
            parts
        )

    def _refine_masks(
        self,
        image: np.ndarray,
        hard_masks: Dict[str, np.ndarray],
        face: Face
    ) -> Dict[str, np.ndarray]:
        """
        Refine hard masks to soft masks.

        Args:
            image: Original image.
            hard_masks: Dictionary of hard masks.
            face: Face.

        Returns:
            Dictionary of part_name -> soft mask (float32 0-1).
        """
        img_short_side = min(image.shape[:2])
        soft_masks = {}

        for part_name, hard_mask in hard_masks.items():
            # Apply Gaussian blur refinement (inlined from MaskRefiner)
            blur_strength = 0.02
            kernel_size = int(img_short_side * blur_strength)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            blurred = cv2.GaussianBlur(hard_mask, (kernel_size, kernel_size), 0)
            soft_mask = blurred.astype(np.float32) / 255.0

            soft_masks[part_name] = soft_mask

        return soft_masks

    def _apply_layer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        layer: MakeupLayer,
        face: Face
    ) -> np.ndarray:
        """
        Apply a single makeup layer.

        Args:
            image: Current result image.
            mask: Soft mask for this part.
            layer: Makeup layer to apply.
            face: Face data (for landmarks).

        Returns:
            Image with this layer's makeup applied.
        """
        params = layer.params

        if isinstance(params, LipsParams):
            return self._blender.blend_lips(image, mask, params)

        if isinstance(params, CheekParams):
            # Import here to avoid circular dependency
            from src.face_makeup.face import CheekLayer
            cheek_side = 'left'
            if isinstance(layer, CheekLayer):
                cheek_side = layer.cheek_side
            return self._blender.blend_cheek(
                image, mask, params,
                landmarks=face.landmarks,
                cheek_side=cheek_side
            )

        if isinstance(params, EyeShadowParams):
            return self._blender.blend_eyeshadow(image, mask, params)

        if isinstance(params, EyeLinerParams):
            return self._blender.blend_eyeliner(image, mask, params)

        # Default blending
        return self._blender.blend(
            image, mask, params.color_params, params.opacity
        )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a timestamp-based session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
