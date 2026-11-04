"""
Debug overlay for ML eyeliner correction.

Draws, for every refined edge of one eye, the P->Q segment and its three points
on an enlarged crop of the source image:

* P (blue)  — eye-opening contour landmark
* Q (green) — outward landmark
* R (red)   — corrected lash-line point ``R = t*P + (1-t)*Q``, labelled with the
  edge landmark id and ``t``

The crop is upscaled with nearest-neighbour sampling so pixel-level lash-line
placement stays visible. Points are mapped to the enlarged pixel grid with
OpenCV fixed-point (shift) drawing, keeping their sub-pixel positions.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# RGB (the overlay is drawn on the RGB source image).
P_COLOR = (0, 0, 255)
Q_COLOR = (0, 255, 0)
R_COLOR = (255, 0, 0)
_PQ_LINE_COLOR = (255, 255, 0)
_TEXT_COLOR = (255, 255, 255)
_TEXT_OUTLINE_COLOR = (0, 0, 0)

_CROP_PAD_RATIO = 0.25
_MIN_CROP_PAD_PX = 8
_TARGET_LONG_SIDE_PX = 800
_MAX_UPSCALE = 12.0
_POINT_RADIUS_PX = 4
_SUBPIXEL_BITS = 4
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.4
_LEGEND = "P: blue  Q: green  R = tP+(1-t)Q: red"


def _crop_bounds(
    xs: Sequence[float], ys: Sequence[float], image_shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Padded bounding box ``(x0, y0, x1, y1)`` of the points, clipped to the image."""
    h, w = image_shape
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = max(_MIN_CROP_PAD_PX, _CROP_PAD_RATIO * span)
    x0 = max(0, int(np.floor(min(xs) - pad)))
    y0 = max(0, int(np.floor(min(ys) - pad)))
    x1 = min(w, int(np.ceil(max(xs) + pad)) + 1)
    y1 = min(h, int(np.ceil(max(ys) + pad)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _draw_text(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    """Draw outlined text so it stays legible on any background."""
    cv2.putText(canvas, text, org, _FONT, scale, _TEXT_OUTLINE_COLOR, 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, _FONT, scale, _TEXT_COLOR, 1, cv2.LINE_AA)


def draw_eyeliner_ml_overlay(
    image: np.ndarray, entries: Sequence[dict]
) -> np.ndarray | None:
    """Return an enlarged RGB eye crop with P / Q / R drawn, or ``None``.

    Args:
        image: Source RGB image (H, W, 3) uint8. Not modified.
        entries: Debug entries of one eye (``EyelinerMLCorrector.get_debug_entries``)
            with px, py, qx, qy, detected_x, detected_y, and optionally t and
            landmark_id.

    Returns:
        The overlay (h', w', 3) uint8, or ``None`` when there are no entries or
        the points lie entirely outside the image.
    """
    if not entries:
        return None
    xs = [e[k] for e in entries for k in ("px", "qx", "detected_x")]
    ys = [e[k] for e in entries for k in ("py", "qy", "detected_y")]
    bounds = _crop_bounds(xs, ys, image.shape[:2])
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds

    scale = max(1.0, min(_MAX_UPSCALE, _TARGET_LONG_SIDE_PX / max(x1 - x0, y1 - y0)))
    canvas = cv2.resize(
        image[y0:y1, x0:x1],
        (round((x1 - x0) * scale), round((y1 - y0) * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    fixed = 1 << _SUBPIXEL_BITS

    def to_canvas(x: float, y: float) -> tuple[int, int]:
        # Pixel centres sit at (i + 0.5) * scale on the enlarged grid.
        return (
            int(round((x - x0 + 0.5) * scale * fixed)),
            int(round((y - y0 + 0.5) * scale * fixed)),
        )

    radius = _POINT_RADIUS_PX * fixed
    drawn_ids: set = set()
    for e in entries:
        key = e.get("landmark_id", id(e))
        if key in drawn_ids:  # a shared edge appears once per polyline
            continue
        drawn_ids.add(key)
        p, q = to_canvas(e["px"], e["py"]), to_canvas(e["qx"], e["qy"])
        r = to_canvas(e["detected_x"], e["detected_y"])
        cv2.line(canvas, p, q, _PQ_LINE_COLOR, 1, cv2.LINE_AA, _SUBPIXEL_BITS)
        for center, color in ((q, Q_COLOR), (p, P_COLOR), (r, R_COLOR)):
            cv2.circle(canvas, center, radius, color, -1, cv2.LINE_8, _SUBPIXEL_BITS)
        if "t" in e:
            label = f"{e.get('landmark_id', '')} t={e['t']:.2f}".strip()
            org = (r[0] // fixed + _POINT_RADIUS_PX + 2, r[1] // fixed - _POINT_RADIUS_PX)
            _draw_text(canvas, label, org, _LABEL_SCALE)

    _draw_text(canvas, _LEGEND, (8, 18), 0.5)
    return canvas


def save_eyeliner_ml_overlay(path: str, image: np.ndarray, entries: Sequence[dict]) -> bool:
    """Draw the overlay and write it to *path* (PNG). Returns whether a file was written."""
    overlay = draw_eyeliner_ml_overlay(image, entries)
    if overlay is None:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        logger.warning("[EyelinerMLDebug] failed to write %s", path)
        return False
    logger.debug("[EyelinerMLDebug] saved: %s (%d entries)", path, len(entries))
    return True
