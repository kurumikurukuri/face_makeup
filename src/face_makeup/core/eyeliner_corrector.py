"""
Eyeliner edge correction using 2nd-derivative zero-crossing detection.

Detects the true eyeliner boundary by sampling pixel intensities along
a landmark->delta direction, smoothing, differentiating, and selecting
the zero-crossing with the strongest dark-to-light transition.
"""
import weakref
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ..constants import EYELINER_CORRECTION_MAP
from .types import FaceLandmarks

# Tuning constants for edge detection
_T_STEP = 0.01          # Sampling step along P->Q normalised direction
_T_MAX = 0.5            # Max fraction of P->Q distance to search (t=0 to 0.5)
_KERNEL_HALF = 3        # Gaussian half-width (ksize = 2*half+1 = 7)
_GRAD_THRESHOLD = 0.1   # Minimum 1st-derivative to accept a crossing as an edge


class EyelinerCorrector:
    """Detects true eyeliner edge positions using image-based analysis."""

    def __init__(self) -> None:
        # Cache the debug entries produced by correct_points so that
        # get_debug_entries can reuse them without re-running find_edge.
        # Structure: {part_name: {'gray_ref': weakref, 'entries': List[dict]}}
        # A weakref to gray is stored instead of id(): if the original gray array
        # is GC'd and CPython reuses its address for a new array, weakref() returns
        # None, so the identity check `weakref() is gray` correctly returns False.
        # Only 'eyeliner_left' / 'eyeliner_right' keys ever exist, so this dict
        # is naturally bounded — no eviction is needed.
        self._correction_cache: dict = {}

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for eyeliner edge detection.

        Stateless: the caller is responsible for reusing the result across
        left/right eyeliner parts within a single frame (see
        FaceSegmenter.segment's eyeliner_gray parameter). Caching this
        internally by image identity is unsafe: video pipelines commonly
        reuse the same array object for successive frames (e.g.
        cap.read(frame)), and an identity-keyed cache would silently return
        a stale frame's result.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            Preprocessed grayscale image (H, W) uint8, normalized to 0-255.
        """
        filtered = cv2.bilateralFilter(image, 9, 50, 50)
        gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def _sample_intensity_along_ray(
        self,
        gray: np.ndarray,
        px: float, py: float,
        dx: float, dy: float,
        norm: float,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], np.ndarray]:
        """
        Sample and smooth pixel intensities from P along direction (dx, dy).

        Args:
            gray: Preprocessed grayscale image (H, W).
            px, py: Start point.
            dx, dy: Unit direction vector.
            norm: Length of original P->Q vector (scales t to pixel distance).

        Returns:
            Tuple of (smoothed_samples, pixel_coords, t_values).
        """
        h, w = gray.shape
        t_values = np.arange(0, _T_MAX, _T_STEP)

        sx = px + dx * (t_values * norm)
        sy = py + dy * (t_values * norm)

        ix = np.round(sx).astype(np.int32)
        iy = np.round(sy).astype(np.int32)
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        ix_safe = np.clip(ix, 0, w - 1)
        iy_safe = np.clip(iy, 0, h - 1)

        raw = gray[iy_safe, ix_safe].astype(np.float64)
        # Forward-fill out-of-bounds samples with the previous valid value
        raw[~valid] = np.nan
        if not valid[0]:
            raw[0] = 0.0
        mask = np.isnan(raw)
        idx = np.where(~mask, np.arange(len(raw)), 0)
        np.maximum.accumulate(idx, out=idx)
        raw[mask] = raw[idx[mask]]

        coords = list(zip(sx.tolist(), sy.tolist()))

        ax = np.arange(-_KERNEL_HALF, _KERNEL_HALF + 1)
        kernel = np.exp(-(ax ** 2) / (2 * 1.0 ** 2))
        kernel /= kernel.sum()
        smoothed = np.convolve(raw, kernel, mode='same')
        return smoothed, coords, t_values

    def _find_best_crossing(
        self,
        samples: np.ndarray,
        coords: List[Tuple[float, float]],
        t_values: np.ndarray,
        fallback: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], dict]:
        """
        Find the 2nd-derivative zero-crossing with the largest positive 1st derivative.

        Selects the dark-to-light transition (eyeliner outer edge) by choosing
        the crossing whose 1st derivative exceeds _GRAD_THRESHOLD and is maximal.

        Args:
            samples: Smoothed intensity samples.
            coords: Pixel coordinates for each sample.
            t_values: t parameter values for each sample.
            fallback: Point returned when no valid crossing is found.

        Returns:
            Tuple of (result_point, debug_dict). debug_dict always contains
            derivative and crossing data regardless of whether a crossing was found.
        """
        first_derivative = np.gradient(samples)
        second_derivative = np.gradient(first_derivative)
        zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]

        debug: dict = {
            'samples': samples,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative,
            't_values': t_values,
            'zero_crossings': zero_crossings if len(zero_crossings) > 0 else np.array([], dtype=int),
            'best_idx': None,
        }

        if len(zero_crossings) == 0:
            return fallback, debug

        best_idx = None
        best_grad = -np.inf
        for idx_z in zero_crossings:
            grad_val = first_derivative[idx_z]
            if grad_val > _GRAD_THRESHOLD and grad_val > best_grad:
                best_grad = grad_val
                best_idx = idx_z

        if best_idx is None:
            return fallback, debug

        detected_x, detected_y = coords[best_idx]
        return (int(detected_x), int(detected_y)), {**debug, 'best_idx': best_idx}

    def find_edge(
        self,
        gray: np.ndarray,
        px: float, py: float,
        qx: float, qy: float,
        return_debug_info: bool = False,
    ) -> Union[Tuple[int, int], Tuple[Tuple[int, int], Optional[dict]]]:
        """
        Find the eyeliner edge along the P->Q direction using
        2nd-derivative zero-crossing detection.

        Samples intensity from P along P->Q (t=0 to _T_MAX), smooths with a
        Gaussian kernel, computes 1st/2nd derivatives, and selects the
        zero-crossing with the largest positive 1st derivative (dark->light
        transition = eyeliner edge).

        Args:
            gray: Preprocessed grayscale image (H, W) uint8.
            px, py: Start point (landmark position).
            qx, qy: End point (delta landmark position).
            return_debug_info: If True, return (point, debug_dict) tuple.

        Returns:
            (x, y) corrected edge point as integers.
            Falls back to (px, py) if no valid edge is found.
            If return_debug_info is True, returns ((x, y), Optional[dict]);
            the dict is None only when px==qx and py==qy or samples are too few.
        """
        fallback = (int(px), int(py))
        dx = qx - px
        dy = qy - py
        norm = np.hypot(dx, dy)

        if norm == 0:
            return (fallback, None) if return_debug_info else fallback

        dx /= norm
        dy /= norm

        samples, coords, t_values = self._sample_intensity_along_ray(
            gray, px, py, dx, dy, norm
        )

        if len(samples) < 5:
            return (fallback, None) if return_debug_info else fallback

        result, debug = self._find_best_crossing(samples, coords, t_values, fallback)
        return (result, debug) if return_debug_info else result

    def correct_points(
        self,
        coords: np.ndarray,
        part_name: str,
        gray: np.ndarray,
    ) -> np.ndarray:
        """
        Apply edge-based correction to eyeliner landmark points.

        For each landmark in the correction map, finds the true eyeliner
        edge using 2nd-derivative zero-crossing detection along the
        landmark->delta direction.

        Args:
            coords: All landmark pixel coordinates (N, 2).
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            gray: Preprocessed grayscale image from preprocess().

        Returns:
            np.ndarray: Corrected point coordinates (M, 2) float32,
                        concatenating upper and lower corrected points in order.

        Raises:
            ValueError: If part_name is not in EYELINER_CORRECTION_MAP.
        """
        if part_name not in EYELINER_CORRECTION_MAP:
            raise ValueError(f"Unknown eyeliner part_name: {part_name!r}")
        correction_map = EYELINER_CORRECTION_MAP[part_name]

        all_corrected = []
        cache_entries: List[dict] = []
        for _key, indices in correction_map.items():
            for idx in indices:
                if isinstance(idx, tuple) and idx[0] == 'edge':
                    _, p_idx, q_idx = idx
                    px, py = coords[p_idx]
                    qx, qy = coords[q_idx]
                    result, dbg = self.find_edge(gray, px, py, qx, qy, return_debug_info=True)
                    # 'gray' is intentionally NOT stored in the cache entry: storing the
                    # full preprocessed frame would pin it in memory for the lifetime of the
                    # corrector.  Callers that need gray receive it as a direct parameter.
                    entry: dict = {
                        'px': float(px), 'py': float(py),
                        'qx': float(qx), 'qy': float(qy),
                        'detected_x': result[0], 'detected_y': result[1],
                    }
                    if dbg is not None:
                        entry.update(dbg)
                    cache_entries.append(entry)
                else:
                    # Plain int landmark — pass through without edge correction.
                    result = (int(coords[idx][0]), int(coords[idx][1]))
                all_corrected.append(result)

        self._correction_cache[part_name] = {'gray_ref': weakref.ref(gray), 'entries': cache_entries}
        return np.array(all_corrected, dtype=np.float32)

    def get_debug_entries(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        image_shape: Tuple[int, int],
        part_name: str,
        side: str,
        gray: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Run eyeliner correction and return per-point debug information.

        Used by DebugVisualizer to generate correction visualizations.

        Args:
            image: Original RGB image (H, W, 3) uint8.
            landmarks: Face landmarks.
            image_shape: (H, W).
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            side: Eye side label ('left' or 'right') stored in each debug entry.
            gray: Pre-computed grayscale from preprocess(). If None,
                  preprocessing is run internally from ``image``.

        Returns:
            List of debug info dicts, one per corrected point. Each dict contains:
                px, py, qx, qy, detected_x, detected_y, side, and a shared
                reference to the preprocessed gray image. When edge detection
                data is available, also includes samples, first_derivative,
                second_derivative, t_values, zero_crossings, and best_idx.
        """
        if part_name not in EYELINER_CORRECTION_MAP:
            return []

        h, w = image_shape
        if gray is None:
            gray = self.preprocess(image)

        # Reuse debug entries cached by the most recent correct_points call
        # for this part (same gray array = same render pass).
        # weakref identity (`weakref() is gray`) avoids the CPython id()-reuse hazard:
        # if the original gray was GC'd, weakref() returns None, so the check fails.
        cached = self._correction_cache.get(part_name)
        if cached is not None and cached['gray_ref']() is gray:
            return [dict(entry, side=side, gray=gray) for entry in cached['entries']]

        # Fallback: re-run edge detection with debug info (e.g. when called
        # outside the normal segment() → correct_points path).
        coords = landmarks.to_image_coords(w, h)
        correction_map = EYELINER_CORRECTION_MAP[part_name]
        debug_entries: List[dict] = []
        for _key, indices in correction_map.items():
            for idx in indices:
                if isinstance(idx, tuple) and idx[0] == 'edge':
                    _, p_idx, q_idx = idx
                    px, py = coords[p_idx]
                    qx, qy = coords[q_idx]
                    result, dbg = self.find_edge(
                        gray, px, py, qx, qy, return_debug_info=True
                    )
                    entry: dict = {
                        'px': float(px), 'py': float(py),
                        'qx': float(qx), 'qy': float(qy),
                        'detected_x': result[0], 'detected_y': result[1],
                        'gray': gray,
                        'side': side,
                    }
                    if dbg is not None:
                        entry.update(dbg)
                    debug_entries.append(entry)
                # Plain int landmarks pass through without edge detection —
                # they produce no debug entry.

        return debug_entries
