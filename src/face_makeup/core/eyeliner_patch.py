"""
Patch geometry for the EyeContourRefiner lash-line correction model.

Ported from EyeContourRefiner's ``eye_contour/data/patch_utils.py``
(https://github.com/kurumikurukuri/EyeContourRefiner). The model was trained on
patches produced by exactly this arithmetic, so any change here shifts patch
pixels and silently degrades predictions: keep it numerically identical to the
upstream implementation (including the ``cv2.warpAffine`` call, whose rounding
changed in OpenCV 5 — hence the ``<5`` pin on both ``opencv-python-headless``
and mediapipe's ``opencv-contrib-python``, which ship the same ``cv2`` module).

Geometry: P is the eye-opening contour landmark and Q the outward landmark of
an edge; a point on the P->Q line is ``R = t*P + (1-t)*Q``.
"""
from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

# Below this |PQ| (pixels) the P->Q direction is undefined.
_DEGENERATE_PQ_DIST = 1e-6


def extract_patch(
    image: np.ndarray,
    px: float,
    py: float,
    qx: float,
    qy: float,
    patch_size: tuple[int, int],
    t_min: float,
    t_max: float,
    perp_scale: float,
) -> np.ndarray:
    """Extract a rotated patch aligned along P->Q covering t in [t_min, t_max].

    The affine transform maps the ``t = t_max`` point to the left edge and the
    ``t = t_min`` point to the right edge of the patch, both on the vertical
    centre row. The vertical axis covers ``perp_scale * |PQ|`` image pixels.

    Args:
        image: RGB image (H, W, 3) uint8.
        px, py: Landmark P in image pixel coordinates.
        qx, qy: Landmark Q in image pixel coordinates.
        patch_size: (height, width) of the output patch.
        t_min: Lower bound of the t range.
        t_max: Upper bound of the t range.
        perp_scale: Perpendicular coverage as a multiple of |PQ|.

    Returns:
        float32 patch (patch_h, patch_w, 3) with values in [0, 1].
        A uniform 0.5 patch when P == Q.
    """
    patch_h, patch_w = patch_size
    dx, dy = qx - px, qy - py
    norm = float(np.hypot(dx, dy))

    if norm == 0.0:
        return np.full((patch_h, patch_w, 3), 0.5, dtype=np.float32)

    ux, uy = dx / norm, dy / norm
    scale = (patch_w - 1) / ((t_max - t_min) * norm)
    sx = t_max * px + (1.0 - t_max) * qx
    sy = t_max * py + (1.0 - t_max) * qy
    actual_perp = (patch_h - 1) / (perp_scale * norm)

    m = np.float32([
        [scale * ux, scale * uy, -scale * (ux * sx + uy * sy)],
        [-uy * actual_perp, ux * actual_perp, actual_perp * (uy * sx - ux * sy) + patch_h // 2],
    ])
    patch = cv2.warpAffine(
        image, m, (patch_w, patch_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return patch.astype(np.float32) / 255.0


def rescale_pq_by_eye_scale(
    px: float,
    py: float,
    qx: float,
    qy: float,
    eye_scale: float,
    factor: float,
) -> tuple[float, float, float, float]:
    """Return P'/Q' with |P'Q'| = max(eye_scale * factor, 1), same midpoint and direction.

    Upstream ``compute_pq_rescale`` in ``"eye_scale"`` mode, without the
    training-only label transform.

    Args:
        px, py, qx, qy: Original P/Q pixel coordinates.
        eye_scale: Eye-level scale from :func:`compute_eye_scale`.
        factor: Target |P'Q'| as a fraction of ``eye_scale``.

    Returns:
        (px', py', qx', qy'). The input unchanged when |PQ| is degenerate.
    """
    orig_pq_dist = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
    if orig_pq_dist <= _DEGENERATE_PQ_DIST:
        return px, py, qx, qy
    target = max(float(eye_scale) * factor, 1.0)
    ox, oy = (px + qx) / 2.0, (py + qy) / 2.0
    ux, uy = (qx - px) / orig_pq_dist, (qy - py) / orig_pq_dist
    return (
        ox - (target / 2.0) * ux,
        oy - (target / 2.0) * uy,
        ox + (target / 2.0) * ux,
        oy + (target / 2.0) * uy,
    )


def compute_eye_scale(points: Iterable[tuple[float, float]]) -> float:
    """Bounding-box diagonal of all P/Q points of one eye (minimum 1.0).

    Args:
        points: Every P and Q point (upper + lower edges) of a single eye.

    Returns:
        The eye scale in pixels.

    Raises:
        ValueError: if *points* is empty.
    """
    pts = list(points)
    if not pts:
        raise ValueError("compute_eye_scale requires at least one point")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    scale = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)
    return max(scale, 1.0)


def reconstruct_point(
    px: float, py: float, qx: float, qy: float, t: float
) -> tuple[float, float]:
    """Return ``R = t*P + (1-t)*Q``."""
    return t * px + (1.0 - t) * qx, t * py + (1.0 - t) * qy
