"""
Face-wide lash-line landmark correction.

The eye-opening contour MediaPipe reports sits slightly inside the visible
lash line. :class:`LashLineCorrector` moves every edge landmark (1401-1407,
1501-1507, 1601-1607, 1701-1707) onto the detected lash line once per face,
and the renderer hands the corrected coordinates to every pass, so eyeliner,
eyeshadow, tear bag and the foundation's eye exclusion all share one lash
line. The mode comes from ``ProcessorConfig.landmark_correction``:

* ``'ml'`` (default) — the EyeContourRefiner ONNX model. A model that cannot
  be loaded (missing, or not matching the contract) raises; there is no fallback.
* ``'rule'`` — the 2nd-derivative zero-crossing detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..constants import EDGE_LANDMARK_DIRECTIONS, EYELINER_CORRECTION_MAP
from .eyeliner_corrector import EyelinerCorrector
from .eyeliner_ml_corrector import EyelinerMLCorrector
from .types import FaceLandmarks

LandmarkCorrection = Literal["ml", "rule"]
LANDMARK_CORRECTION_MODES: tuple[str, ...] = ("ml", "rule")

EdgeCorrector = EyelinerCorrector | EyelinerMLCorrector


@dataclass(frozen=True)
class LashLineCorrection:
    """Result of one face's lash-line correction.

    Attributes:
        coords: (N, 2) float32 copy of the input coordinates whose edge
            landmark rows hold the corrected lash-line points.
        corrector: The edge corrector that produced them; its
            ``get_debug_entries`` returns this render's cached entries.
    """

    coords: np.ndarray
    corrector: EdgeCorrector


class LashLineCorrector:
    """Correct the lash-line edge landmarks of a face (see the module docstring)."""

    def __init__(self, mode: str) -> None:
        """
        Args:
            mode: 'ml' or 'rule'.

        Raises:
            ValueError: for any other *mode*.
        """
        if mode not in LANDMARK_CORRECTION_MODES:
            raise ValueError(f"mode must be 'ml' or 'rule', got {mode!r}")
        self._mode = mode
        # Built on first use; the ML corrector loads the installed model on its
        # first correction, so building a processor never does.
        self._corrector: EdgeCorrector | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def resolve(self) -> EdgeCorrector:
        """Return the edge corrector for this mode, building it on first use."""
        if self._corrector is None:
            self._corrector = (
                EyelinerMLCorrector()
                if self._mode == "ml"
                else EyelinerCorrector()
            )
        return self._corrector

    def apply(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        coords: np.ndarray,
    ) -> LashLineCorrection:
        """Return *coords* with both eyes' edge landmarks moved onto the lash line.

        Args:
            image: Source RGB image (H, W, 3) uint8 the lash line is detected in.
            landmarks: The face's (extended) landmarks; the ML model reads
                their unrounded float coordinates.
            coords: Renderer coordinates (``landmarks.to_image_coords``), on
                which the corrected points are reconstructed. Not modified.

        Raises:
            RuntimeError / ModelContractError: in 'ml' mode when the model
                cannot be loaded; any inference error.
        """
        corrector = self.resolve()
        is_ml = isinstance(corrector, EyelinerMLCorrector)
        if is_ml:
            h, w = image.shape[:2]
            model_coords = landmarks.to_image_coords_float(w, h)
        corrected = np.asarray(coords, dtype=np.float32).copy()
        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            if is_ml:
                points = corrector.correct_points(
                    coords, part_name, eyeliner=image, model_coords=model_coords
                )
            else:
                points = corrector.correct_points(coords, part_name, eyeliner=image)
            slots = [idx for polyline in correction_map.values() for idx in polyline]
            for slot, idx in enumerate(slots):
                if idx in EDGE_LANDMARK_DIRECTIONS:
                    corrected[idx] = points[slot]
        return LashLineCorrection(coords=corrected, corrector=corrector)
