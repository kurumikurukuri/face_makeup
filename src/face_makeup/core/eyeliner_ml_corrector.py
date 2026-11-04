"""
Eyeliner edge correction using the EyeContourRefiner dual-branch ONNX model.

The model predicts, for each edge landmark, the parameter ``t`` along the
P->Q segment such that the corrected lash-line point is ``R = t*P + (1-t)*Q``
(P = eye-opening contour landmark, Q = outward landmark). Two patches are
extracted per point:

* branch A: raw P/Q coordinates (t in [-2.0, 3.0], perp_scale 2.0)
* branch B: eye_scale-rescaled P'/Q' coordinates (t in [-0.5, 1.5],
  perp_scale 2.0, factor 0.2)

The branch fusion is baked into the exported ONNX graph, so a single
``session.run`` over all edge points returns their raw ``t`` values directly.
Model fetching, loading and contract validation live in :mod:`.eyeliner_model`.
"""
from __future__ import annotations

import logging
import os
import weakref
from typing import TYPE_CHECKING

import numpy as np

from ..constants import (
    EDGE_LANDMARK_DIRECTIONS,
    EYELINER_CORRECTION_MAP,
)
from .eyeliner_model import PATCH_SIZE, load_session
from .eyeliner_patch import (
    compute_eye_scale,
    extract_patch,
    reconstruct_point,
    rescale_pq_by_eye_scale,
)
from .lash_line_edges import unique_edges
from .types import FaceLandmarks

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)

# Branch configs frozen to match the shipped ``eye_contour_refiner.onnx``
# (upstream eyeliner/training/dual_branch.py: branch_a_x0 / branch_b_eye_scale).
_BRANCH_A = {
    "t_min": -2.0,
    "t_max": 3.0,
    "perp_scale": 2.0,
    # Raw-t range the training labels were clamped to; predictions are
    # clipped to it so an off-distribution output cannot fling R off the eye.
    "label_t_min": -1.0,
    "label_t_max": 2.0,
}
# t used when the model returns NaN/inf: R = P, the uncorrected eye-opening
# contour point (the same fallback the rule corrector uses).
_NON_FINITE_T_FALLBACK = 1.0
_BRANCH_B = {
    "t_min": -0.5,
    "t_max": 1.5,
    "perp_scale": 2.0,
    "eye_scale_factor": 0.2,
}


def _pq(coords: np.ndarray, edge_id: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the (P, Q) pixel coordinates of an edge landmark."""
    p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[edge_id]
    return (
        (float(coords[p_idx][0]), float(coords[p_idx][1])),
        (float(coords[q_idx][0]), float(coords[q_idx][1])),
    )


def _sanitize_t(raw_t: np.ndarray) -> np.ndarray:
    """Return *raw_t* with NaN/inf replaced by the fallback and clipped to the label range."""
    finite = np.isfinite(raw_t)
    if not finite.all():
        logger.warning(
            "Lash-line correction model returned %d non-finite t value(s); using the "
            "uncorrected landmark for those points.",
            int((~finite).sum()),
        )
    t = np.where(finite, raw_t, _NON_FINITE_T_FALLBACK)
    return np.clip(t, _BRANCH_A["label_t_min"], _BRANCH_A["label_t_max"])


def _edge_entry(coords: np.ndarray, edge_id: int, t: float) -> dict:
    """Return the P/Q/R geometry of one refined edge (also its debug entry)."""
    (px, py), (qx, qy) = _pq(coords, edge_id)
    rx, ry = reconstruct_point(px, py, qx, qy, t)
    return {
        "landmark_id": edge_id,
        "px": px, "py": py, "qx": qx, "qy": qy,
        "detected_x": float(rx), "detected_y": float(ry),
        "t": t,
    }


class EyelinerMLCorrector:
    """Refine eyeliner lash-line landmarks with the EyeContourRefiner ONNX model."""

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
    ) -> None:
        """
        Args:
            model_path: Explicit model file; ``None`` uses the model installed
                with face_makeup (fetched at install / pull time).
        """
        self._model_path = model_path
        self._session: "ort.InferenceSession | None" = None
        # Latest correct_points result per part, reused by get_debug_entries so
        # debug rendering does not run the model a second time. A weakref to
        # the source image ties the entries to that render without pinning it.
        self._debug_cache: dict[str, tuple[weakref.ref, list[dict]]] = {}

    def _load_session(self) -> "ort.InferenceSession":
        if self._session is None:
            self._session = load_session(self._model_path)
        return self._session

    def correct_points(
        self,
        coords: np.ndarray,
        part_name: str,
        eyeliner: np.ndarray,
        model_coords: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply ML-based correction to eyeliner landmark points.

        For each distinct edge landmark in the correction map, extracts branch
        A/B patches, runs the model in a single batched call, and reconstructs
        ``R = t*P + (1-t)*Q``. Plain int landmarks pass through unchanged.
        ``eye_scale`` is computed from the P/Q points of those distinct edges
        (the 14 edges the model was trained on). The per-edge P/Q/R/t are
        cached for :meth:`get_debug_entries`.

        The model sees *model_coords* (unrounded float, like its training
        data) while R is reconstructed on the P/Q of *coords* (the integer
        pixel coordinates every pass draws with). R itself keeps its sub-pixel
        float value.

        Args:
            coords: All landmark pixel coordinates (N, 2), as passed to every
                mask strategy (``landmarks.to_image_coords``, int32). Used for
                drawing: R and the uncorrected points.
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            eyeliner: Source RGB image (H, W, 3) uint8, used for patch extraction.
            model_coords: Float landmark pixel coordinates (N, 2)
                (``landmarks.to_image_coords_float``) for the model input
                (patches, P/Q, eye_scale). ``None`` uses *coords*.

        Returns:
            np.ndarray: Corrected point coordinates (M, 2) float32,
                        concatenating upper and lower corrected points in order.

        Raises:
            ValueError: if ``part_name`` is not a known eyeliner part.
        """
        if part_name not in EYELINER_CORRECTION_MAP:
            raise ValueError(
                f"Unknown part_name: '{part_name}'. "
                f"Expected one of {sorted(EYELINER_CORRECTION_MAP.keys())}."
            )

        correction_map = EYELINER_CORRECTION_MAP[part_name]
        coords = np.asarray(coords, dtype=np.float32)
        model_coords = (
            coords if model_coords is None
            else np.asarray(model_coords, dtype=np.float32)
        )

        edges = unique_edges(part_name)
        t_by_edge: dict[int, float] = {}
        if edges:
            eye_scale = compute_eye_scale(
                point for idx, _ in edges for point in _pq(model_coords, idx)
            )
            t_by_edge = self._infer_t(model_coords, edges, eye_scale, eyeliner)

        out: list[tuple[float, float]] = []
        debug_entries: list[dict] = []
        for indices in correction_map.values():
            for idx in indices:
                if idx in t_by_edge:
                    entry = _edge_entry(coords, idx, t_by_edge[idx])
                    debug_entries.append(entry)
                    out.append((entry["detected_x"], entry["detected_y"]))
                else:
                    out.append((float(coords[idx][0]), float(coords[idx][1])))

        self._debug_cache[part_name] = (weakref.ref(eyeliner), debug_entries)
        return np.array(out, dtype=np.float32)

    def _infer_t(
        self,
        coords: np.ndarray,
        edges: list[tuple[int, bool]],
        eye_scale: float,
        eyeliner: np.ndarray,
    ) -> dict[int, float]:
        """Run the model on *edges* in one batch and return ``{edge_id: t}``.

        Non-finite outputs fall back to ``t = 1`` (R = P) and every ``t`` is
        clipped to the training label range, so the model can never place a
        point at NaN or far outside the eye.
        """
        session = self._load_session()

        n = len(edges)
        patch_h, patch_w = PATCH_SIZE
        patch_a = np.empty((n, 3, patch_h, patch_w), dtype=np.float32)
        patch_b = np.empty((n, 3, patch_h, patch_w), dtype=np.float32)
        is_upper_f = np.empty((n, 1), dtype=np.float32)
        px_f = np.empty((n, 1), dtype=np.float32)
        py_f = np.empty((n, 1), dtype=np.float32)
        qx_f = np.empty((n, 1), dtype=np.float32)
        qy_f = np.empty((n, 1), dtype=np.float32)

        for i, (idx, is_upper) in enumerate(edges):
            (px, py), (qx, qy) = _pq(coords, idx)

            # branch A: raw P/Q coordinates.
            pa = extract_patch(
                eyeliner, px, py, qx, qy,
                patch_size=PATCH_SIZE,
                t_min=_BRANCH_A["t_min"],
                t_max=_BRANCH_A["t_max"],
                perp_scale=_BRANCH_A["perp_scale"],
            )

            # branch B: eye_scale-rescaled P'/Q' coordinates.
            pb_px, pb_py, pb_qx, pb_qy = rescale_pq_by_eye_scale(
                px, py, qx, qy, eye_scale, _BRANCH_B["eye_scale_factor"],
            )
            pb = extract_patch(
                eyeliner, pb_px, pb_py, pb_qx, pb_qy,
                patch_size=PATCH_SIZE,
                t_min=_BRANCH_B["t_min"],
                t_max=_BRANCH_B["t_max"],
                perp_scale=_BRANCH_B["perp_scale"],
            )

            patch_a[i] = pa.transpose(2, 0, 1)
            patch_b[i] = pb.transpose(2, 0, 1)
            is_upper_f[i, 0] = is_upper
            px_f[i, 0] = px
            py_f[i, 0] = py
            qx_f[i, 0] = qx
            qy_f[i, 0] = qy

        feed = {
            "patch_a": patch_a,
            "patch_b": patch_b,
            "is_upper": is_upper_f,
            "px": px_f,
            "py": py_f,
            "qx": qx_f,
            "qy": qy_f,
            "eye_scale": np.full((n, 1), eye_scale, dtype=np.float32),
        }
        t = _sanitize_t(session.run(None, feed)[0])  # (n, 1)
        return {idx: float(t[i, 0]) for i, (idx, _) in enumerate(edges)}

    def get_debug_entries(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        image_shape: tuple[int, int],
        part_name: str,
        side: str,
        eyeliner: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Return per-point P/Q/R debug info for ML-corrected edge landmarks.

        Reuses the entries cached by the latest :meth:`correct_points` call
        for *part_name* on the same source image (the normal render path), so
        no second inference runs. Otherwise the correction is run once here.
        Entries carry no rule-based ``samples``; the renderer draws them with
        :meth:`DebugVisualizer.save_eyeliner_ml_debug` instead.

        Args:
            image: Original RGB image (H, W, 3) uint8.
            landmarks: Face landmarks.
            image_shape: (H, W).
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            side: Eye side label ('left' or 'right') stored in each entry.
            eyeliner: Image to use for correction (defaults to image).

        Returns:
            One dict per edge slot of the correction map (a shared edge
            appears once per polyline) with landmark_id, px, py, qx, qy,
            detected_x, detected_y (= R), t, side and eyeliner.
        """
        if part_name not in EYELINER_CORRECTION_MAP:
            raise ValueError(
                f"Unknown part_name: '{part_name}'. "
                f"Expected one of {sorted(EYELINER_CORRECTION_MAP.keys())}."
            )

        if eyeliner is None:
            eyeliner = image

        cached = self._debug_cache.get(part_name)
        if cached is None or cached[0]() is not eyeliner:
            h, w = image_shape
            self.correct_points(
                landmarks.to_image_coords(w, h), part_name, eyeliner,
                model_coords=landmarks.to_image_coords_float(w, h),
            )
            cached = self._debug_cache[part_name]

        return [dict(entry, side=side, eyeliner=eyeliner) for entry in cached[1]]
