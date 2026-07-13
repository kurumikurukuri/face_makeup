"""
Internal rendering engine for makeup composition.

Handles composition of makeup layers onto images via two-phase execution:
Phase 1: generate final soft masks, then gate them by clip_landmarks (AND),
    exclusion_landmarks (AND NOT), and feathered hair exclusion (AND NOT).
Phase 2 (priority sort): apply blends (pure colour).
Not exposed in public API - used by ImageContext.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import uuid4

import cv2
import numpy as np

from face_makeup.core.passes.cheek import CheekBlendStrategy
from face_makeup.core.types import CheekParams, EyeLinerParams

if TYPE_CHECKING:
    from face_makeup.face import Face
    from face_makeup.core.debug_visualizer import DebugVisualizer

logger = logging.getLogger(__name__)


def _to_soft01(mask: np.ndarray) -> np.ndarray:
    """Normalize a mask-strategy output to float32 in [0, 1].

    Mask strategies return their final soft mask as float32 in [0, 1] via
    soften_mask.  This is defensive handling for any MaskStrategy that does
    not follow that contract and returns uint8 (0-255) instead.  We use a
    value-range check rather than dtype sniffing so that a float32 mask
    whose values happen to be in the 0-255 range is still normalized
    correctly.

    Uses a threshold of 1.5 (not 1.0) so that float32 rounding above 1.0
    (e.g. 1.0000004 from the smootherstep feather ramp) is not mistaken for
    an un-normalized 0-255 mask.
    """
    soft = mask.astype(np.float32, copy=False)
    if soft.max() > 1.5:
        soft /= 255.0
    return np.clip(soft, 0.0, 1.0)


def _feather_exclusion(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Feather a hard exclusion (e.g. hair) mask into a soft 0..1 gate.

    Softening the exclusion edge means makeup fades out across the boundary
    instead of ending in a hard cut, matching how pass-to-pass exclusions
    (which reuse a source pass's already-soft footprint) behave — one
    consistent, feathered exclusion path for every source.
    """
    soft = _to_soft01(mask)
    soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(soft, 0.0, 1.0)


def _build_landmark_group_mask(
    coords: np.ndarray,
    groups: List[List[int]],
    image_shape: tuple[int, int],
    feather_sigma: float = 2.0,
) -> np.ndarray | None:
    """Build a feathered soft mask from landmark point groups.

    Each group is a list of landmark indices.  Points from each group are
    filled with ``fillPoly`` (subpixel shift=4, concave OK), then all
    groups are OR-combined and feathered with a Gaussian blur.

    Returns:
        Float32 soft mask in [0, 1], or ``None`` when *groups* is empty
        or every group has fewer than 3 valid landmark points.
    """
    if not groups:
        return None

    h, w = image_shape
    n_coords = len(coords)
    shift = 4
    scale = 1 << shift
    combined = np.zeros((h, w), dtype=np.uint8)

    any_valid = False
    for indices in groups:
        valid = [i for i in indices if i < n_coords]
        if len(valid) < 3:
            continue
        any_valid = True
        pts = coords[valid]
        pts_shifted = (pts * scale).astype(np.int32)
        cv2.fillPoly(combined, [pts_shifted], 255, shift=shift)

    if not any_valid:
        return None

    return _feather_exclusion(combined, sigma=feather_sigma)


class Renderer:
    """
    Internal rendering engine.

    Handles composition of makeup layers onto images.
    """

    def __init__(
        self,
        visualizer: Optional[DebugVisualizer] = None,
        hair_segmenter: Optional[object] = None
    ):
        """
        Initialize renderer.

        Args:
            visualizer: Optional debug visualizer.
            hair_segmenter: Optional HairSegmenter for hair mask subtraction.
        """
        self._visualizer = visualizer
        self._hair_segmenter = hair_segmenter

    def render_face(
        self,
        image: np.ndarray,
        face: Face,
        debug: bool = False,
        session_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Render all makeup layers for a single face via two-phase execution.

        Phase 1: Generate final soft masks; apply clip_landmarks (AND),
                 exclusion_landmarks (AND NOT), and hair exclusion (AND NOT).
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

        # Phase 1 — Mask generation.
        # Each mask strategy returns its FINAL soft mask (geometry + fade baked
        # in). Clip / exclusion / hair are applied afterwards as feathered soft
        # gates so a hole never corrupts the distance-based fade (computed on
        # the solid geometry).
        #
        # Keyed by id(pass_) rather than pass_type: several passes can share the
        # same pass_type (e.g. two tear_bag_highlight layers with different
        # `size` stacked for a gradient look). A pass_type-keyed dict would let
        # the later instance silently overwrite the earlier one, so every pass
        # gets its own slot.
        mask_cache: dict[int, np.ndarray] = {}          # final soft mask
        hard_masks: Dict[str, np.ndarray] = {}          # debug preview

        # Run hair segmentation once → one feathered soft exclusion mask reused
        # by every pass (applied after each pass's own fade).
        hair_soft = None
        if self._hair_segmenter is not None:
            hair_raw = self._hair_segmenter.segment(image)
            if hair_raw is not None:
                hair_soft = _feather_exclusion(hair_raw)

        for pass_ in passes:
            if pass_.mask_strategy is None:
                logger.warning(f"No mask strategy for pass: {pass_.pass_type}")
                continue

            soft = _to_soft01(
                pass_.mask_strategy.generate(
                    coords, image.shape[:2],
                    image=image,
                    landmarks=face.landmarks,
                )
            )

            # Step A: clip — restrict mask to landmark-defined regions (AND).
            clip_mask = _build_landmark_group_mask(
                coords, pass_.clip_landmarks, image.shape[:2],
            )
            if clip_mask is not None:
                soft = soft * clip_mask

            # Step B: landmark-based exclusion — subtract landmark-defined
            # regions from the mask (AND NOT).  Always applied regardless of
            # which other passes are present.
            lm_exclusion = _build_landmark_group_mask(
                coords, pass_.exclusion_landmarks, image.shape[:2],
            )
            if lm_exclusion is not None:
                soft = soft * (1.0 - lm_exclusion)

            # Step C: hair exclusion — feathered soft gate (AND NOT).
            if hair_soft is not None:
                soft = soft * (1.0 - hair_soft)

            mask_cache[id(pass_)] = soft
            # Debug dump keyed by pass_type; if multiple passes share a
            # pass_type only the last one's preview is saved (rendering itself
            # is unaffected — see mask_cache above).
            hard_masks[pass_.pass_type] = (soft * 255.0).astype(np.uint8)

        if debug and self._visualizer:
            self._visualizer.save_masks(image, hard_masks, face_session, "hard")
            soft_masks = {
                pass_.pass_type: mask_cache[id(pass_)]
                for pass_ in passes
                if id(pass_) in mask_cache
            }
            self._visualizer.save_masks(image, soft_masks, face_session, "soft")

        # Phase 2 — Blend application (priority-sorted, lowest first)
        result = image.copy()
        for pass_ in sorted(passes, key=lambda p: p.priority):
            cached = mask_cache.get(id(pass_))
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
                and hasattr(pass_.mask_strategy, "create_cheek_mask")
            ):
                cheek_side = (
                    "right" if pass_.pass_type.endswith("_right") else "left"
                )
                # Geometry now lives on the mask strategy; base_mask content is
                # unused by the 3D path (landmark-driven), so the cached soft
                # mask is a fine non-empty placeholder.
                cheek_result = pass_.mask_strategy.create_cheek_mask(
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
        """Generate a unique session ID with timestamp and random component."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{uuid4().hex[:8]}"
