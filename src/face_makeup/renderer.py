"""
Internal rendering engine for makeup composition.

Handles composition of makeup layers onto images via two-phase execution:
Phase 1 (topological sort): generate masks, subtract exclusions.
Phase 2 (priority sort): apply blends.
Not exposed in public API - used by ImageContext.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np

from face_makeup.core.pipeline import topological_sort
from face_makeup.core.passes.cheek import CheekBlendStrategy
from face_makeup.core.types import CheekParams, EyeLinerParams

if TYPE_CHECKING:
    from face_makeup.face import Face
    from face_makeup.core.debug_visualizer import DebugVisualizer

logger = logging.getLogger(__name__)


class Renderer:
    """
    Internal rendering engine.

    Handles composition of makeup layers onto images.
    """

    def __init__(
        self,
        visualizer: Optional[DebugVisualizer] = None
    ):
        """
        Initialize renderer.

        Args:
            visualizer: Optional debug visualizer.
        """
        self._visualizer = visualizer

    def render_face(
        self,
        image: np.ndarray,
        face: Face,
        debug: bool = False,
        session_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Render all makeup layers for a single face via two-phase execution.

        Phase 1: Topological sort by exclusion dependency, generate masks,
                 and subtract exclusion masks.
        Phase 2: Priority sort and apply blends.
        """
        passes = face.makeup_passes
        if not passes:
            logger.debug(f"Face {face.face_id}: No makeup passes to render")
            return image

        session_id = session_id or self._generate_session_id()
        face_session = f"{session_id}_face{face.face_id}"

        h, w = image.shape[:2]
        coords = face.landmarks.to_image_coords(w, h)

        # Phase 1 — Mask generation (dependency-ordered)
        mask_registry: Dict[str, np.ndarray] = {}
        hard_masks: Dict[str, np.ndarray] = {}

        for pass_ in topological_sort(list(passes)):
            if pass_.mask_strategy is None:
                logger.warning(f"No mask strategy for pass: {pass_.pass_type}")
                continue

            mask = pass_.mask_strategy.generate(
                coords, image.shape[:2],
                image=image,
                landmarks=face.landmarks,
            )

            # Subtract exclusion masks
            for exclusion_name in pass_.exclusion_passes:
                if exclusion_name in mask_registry:
                    exclusion = mask_registry[exclusion_name]
                    if exclusion.dtype != mask.dtype:
                        exclusion = exclusion.astype(mask.dtype)
                    mask = cv2.subtract(mask, exclusion)

            mask_registry[pass_.pass_type] = mask
            hard_masks[pass_.pass_type] = mask

        if debug and self._visualizer:
            self._visualizer.save_masks(image, hard_masks, face_session, "hard")

        # Convert masks to float32 0-1 in local dict (no mutation of pass objects)
        soft_mask_registry: Dict[str, np.ndarray] = {}
        for pass_ in passes:
            if pass_.pass_type in mask_registry:
                soft_mask_registry[pass_.pass_type] = (
                    mask_registry[pass_.pass_type].astype(np.float32) / 255.0
                )

        if debug and self._visualizer:
            self._visualizer.save_masks(image, soft_mask_registry, face_session, "soft")

        # Phase 2 — Blend application (priority-sorted, lowest first)
        result = image.copy()
        for pass_ in sorted(passes, key=lambda p: p.priority):
            cached = soft_mask_registry.get(pass_.pass_type)
            if cached is None or not cached.any():
                if cached is None:
                    logger.warning(f"No mask for pass: {pass_.pass_type}")
                continue

            if pass_.blend_strategy is None:
                logger.warning(f"No blend strategy for pass: {pass_.pass_type}")
                continue

            result = pass_.blend_strategy.apply(
                result, cached, pass_.params,
                landmarks=face.landmarks,
            )

            if debug and self._visualizer:
                logger.debug(f"Applied pass: {pass_.pass_type}")

            # Eyeliner correction debug visualization
            if (
                debug
                and self._visualizer
                and isinstance(pass_.params, EyeLinerParams)
                and pass_.mask_strategy is not None
            ):
                corrector = getattr(pass_.mask_strategy, '_corrector', None)
                if corrector is not None:
                    eye_side = (
                        'right' if pass_.pass_type.endswith('_right') else 'left'
                    )
                    debug_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    debug_entries = corrector.get_debug_entries(
                        image=image,
                        landmarks=face.landmarks,
                        image_shape=image.shape[:2],
                        part_name=pass_.pass_type,
                        side=eye_side,
                        eyeliner=image,
                    )
                    for i, entry in enumerate(debug_entries):
                        if 'samples' not in entry:
                            continue
                        self._visualizer.save_eyeliner_correction_debug(
                            image=image,
                            side=entry['side'],
                            gray=debug_gray,
                            px=entry['px'], py=entry['py'],
                            qx=entry['qx'], qy=entry['qy'],
                            detected_x=entry['detected_x'],
                            detected_y=entry['detected_y'],
                            samples=entry.get('samples'),
                            first_derivative=entry.get('first_derivative'),
                            second_derivative=entry.get('second_derivative'),
                            t_values=entry.get('t_values'),
                            zero_crossings=entry.get('zero_crossings'),
                            best_idx=entry.get('best_idx'),
                            point_index=i,
                            session_id=face_session,
                        )

            # Cheek 3D debug visualization
            if (
                debug
                and self._visualizer
                and isinstance(pass_.params, CheekParams)
                and isinstance(pass_.blend_strategy, CheekBlendStrategy)
            ):
                cheek_side = (
                    "right" if pass_.pass_type.endswith("_right") else "left"
                )
                cheek_result = pass_.blend_strategy.create_cheek_mask(
                    image_shape=image.shape[:2],
                    base_mask=cached,
                    params=pass_.params,
                    landmarks=face.landmarks,
                    cheek_side=cheek_side,
                    return_debug_info=True,
                )
                if isinstance(cheek_result, tuple):
                    cheek_mask, debug_info = cheek_result
                    self._visualizer.save_cheek_3d_debug(
                        image=image,
                        cheek_mask=cheek_mask,
                        debug_info=debug_info,
                        cheek_side=cheek_side,
                        session_id=face_session,
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
            if not face.makeup_passes:
                continue

            if debug and self._visualizer:
                self._visualizer.save_bbox(image, face.bbox, f"{session_id}_face{face.face_id}")
                self._visualizer.save_landmarks(image, face.landmarks, f"{session_id}_face{face.face_id}")

            result = self.render_face(result, face, debug, session_id)

        if debug and self._visualizer:
            self._visualizer.save_result(image, result, session_id)

        logger.debug(f"Composed {len(faces)} faces")
        return result

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a timestamp-based session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
