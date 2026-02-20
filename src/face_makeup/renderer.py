"""
Internal rendering engine for makeup composition.

Handles composition of makeup layers onto images.
Not exposed in public API - used by ImageContext.
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from face_makeup.core.segmenter import FaceSegmenter
from face_makeup.core.makeup_blender import MakeupBlender
from face_makeup.core.types import (
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams,
    EyeIrisParams,
    PARAMS_PRIORITY
)

if TYPE_CHECKING:
    from face_makeup.face import Face, MakeupLayer
    from face_makeup.core.debug_visualizer import DebugVisualizer

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

        # Work on shallow copies to avoid mutating Face internal state
        layers = [copy.copy(layer) for layer in face.makeup_layers]

        # Get unique part names from layers
        part_names = list(set(layer.part_name for layer in layers))

        # Preprocess for eyeliner once so segment() and the debug path share the
        # same gray array — avoids a second bilateralFilter call in debug mode.
        eyeliner_gray: Optional[np.ndarray] = None
        if any(isinstance(layer.params, EyeLinerParams) for layer in layers):
            eyeliner_gray = self._segmenter.preprocess_for_eyeliner(image)

        # Generate hard masks (pass pre-computed gray for eyeliner edge correction)
        hard_masks = self._generate_masks(image.shape[:2], face, part_names, image=image, eyeliner_gray=eyeliner_gray)

        if debug and self._visualizer:
            self._visualizer.save_masks(image, hard_masks, face_session, "hard")

        # Assign masks to layers (float32 conversion only, blur applied in blend)
        for layer in layers:
            if layer.part_name in hard_masks:
                hard_mask = hard_masks[layer.part_name]
                layer.mask = hard_mask.astype(np.float32) / 255.0

        # Resolve overlaps by type priority
        self._resolve_overlaps(layers)

        if debug and self._visualizer:
            # Create soft_masks dict for debug visualization
            soft_masks = {layer.part_name: layer.mask for layer in layers if layer.mask is not None}
            self._visualizer.save_masks(image, soft_masks, face_session, "soft")

        # Apply each layer
        result = image.copy()
        for layer in layers:
            if layer.mask is None:
                logger.warning(f"No mask for part: {layer.part_name}")
                continue

            result = self._apply_layer(result, layer.mask, layer, face)

            if debug and self._visualizer:
                logger.debug(f"Applied layer: {layer.part_name}")

                # Save cheek 3D debug visualization
                if isinstance(layer.params, CheekParams):
                    # Use getattr to avoid circular import of CheekLayer
                    cheek_side = getattr(layer, 'cheek_side', 'left')
                    self._visualizer.save_cheek_3d_debug(
                        image=image,
                        soft_mask=layer.mask,
                        params=layer.params,
                        landmarks=face.landmarks,
                        segmenter=self._segmenter,
                        part_name=layer.part_name,
                        cheek_side=cheek_side,
                        session_id=face_session
                    )

                # Save eyeliner correction debug visualization
                if isinstance(layer.params, EyeLinerParams):
                    eye_side = 'right' if layer.part_name.endswith('_right') else 'left'
                    debug_entries = self._segmenter.get_eyeliner_correction_debug(
                        image=image,
                        landmarks=face.landmarks,
                        image_shape=image.shape[:2],
                        part_name=layer.part_name,
                        side=eye_side,
                        gray=eyeliner_gray,
                    )
                    for i, entry in enumerate(debug_entries):
                        if 'samples' not in entry:
                            continue
                        self._visualizer.save_eyeliner_correction_debug(
                            image=image,
                            side=entry['side'],
                            gray=entry['gray'],
                            px=entry['px'], py=entry['py'],
                            qx=entry['qx'], qy=entry['qy'],
                            detected_x=entry['detected_x'],
                            detected_y=entry['detected_y'],
                            samples=entry['samples'],
                            first_derivative=entry['first_derivative'],
                            second_derivative=entry['second_derivative'],
                            t_values=entry['t_values'],
                            zero_crossings=entry['zero_crossings'],
                            best_idx=entry['best_idx'],
                            point_index=i,
                            session_id=face_session
                        )

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
        parts: List[str],
        image: Optional[np.ndarray] = None,
        eyeliner_gray: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate hard masks for face parts.

        Args:
            image_shape: (H, W) tuple.
            face: Face with landmarks.
            parts: List of part names.
            image: Original image for eyeliner edge correction.
            eyeliner_gray: Pre-computed grayscale from preprocess_for_eyeliner.
                           Passed through to avoid recomputing bilateralFilter.

        Returns:
            Dictionary of part_name -> hard mask (uint8 0/255).
        """
        return self._segmenter.segment(
            image_shape,
            face.landmarks,
            parts,
            image=image,
            eyeliner_gray=eyeliner_gray
        )

    def _resolve_overlaps(self, layers: List[MakeupLayer]) -> None:
        """
        Resolve mask overlaps based on PartParams type priority.

        Same PartParams type layers are allowed to overlap.
        Different type layers are resolved by priority (higher priority wins).

        Args:
            layers: List of MakeupLayer with masks already assigned.
                    Modified in-place.
        """
        # Filter layers with masks
        layers_with_masks = [layer for layer in layers if layer.mask is not None]
        if not layers_with_masks:
            return

        h, w = layers_with_masks[0].mask.shape[:2]

        # Sort by priority (process HIGH priority first)
        sorted_layers = sorted(
            layers_with_masks,
            key=lambda layer: PARAMS_PRIORITY.get(type(layer.params), 0),
            reverse=True
        )

        # Track used masks per type (category)
        # Higher priority types are processed first, so their masks are recorded
        type_used_masks: Dict[type, np.ndarray] = {}

        for layer in sorted_layers:
            params_type = type(layer.params)

            # Subtract masks from higher priority types (already processed)
            # Same type overlap is OK (don't subtract from same type)
            for other_type, used_mask in type_used_masks.items():
                if other_type != params_type:
                    layer.mask = np.clip(layer.mask - used_mask, 0, 1)

            # Update used mask for this type
            if params_type not in type_used_masks:
                type_used_masks[params_type] = np.zeros((h, w), dtype=np.float32)
            type_used_masks[params_type] = np.maximum(
                type_used_masks[params_type], layer.mask
            )

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
            # Use getattr to avoid circular import of CheekLayer
            cheek_side = getattr(layer, 'cheek_side', 'left')
            cheek_mask = self._segmenter.create_cheek_mask(
                image.shape[:2], mask, params,
                landmarks=face.landmarks,
                cheek_side=cheek_side
            )
            return self._blender.blend_cheek(
                image, mask, cheek_mask, params
            )

        if isinstance(params, EyeShadowParams):
            return self._blender.blend_eyeshadow(image, mask, params)

        if isinstance(params, EyeLinerParams):
            return self._blender.blend_eyeliner(image, mask, params)

        if isinstance(params, EyeIrisParams):
            # eyeiris mask (already float32 from render_face)
            eyeiris_mask = mask

            # Generate eye mask for constraining iris to eye region
            eye_part = layer.part_name.replace('eyeiris_', 'eye_')
            eye_masks = self._segmenter.segment(
                image.shape[:2],
                face.landmarks,
                [eye_part]
            )
            eye_mask = eye_masks.get(eye_part)
            if eye_mask is None:
                raise ValueError(f"Failed to generate eye mask for '{eye_part}'")
            eye_mask_soft = eye_mask.astype(np.float32) / 255.0 if eye_mask.dtype == np.uint8 else eye_mask

            return self._blender.blend_eyeiris(image, eyeiris_mask, eye_mask_soft, params)

        # Default blending
        return self._blender.blend(
            image, mask, params.color_params, params.opacity
        )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a timestamp-based session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
