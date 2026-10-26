"""
Eyeliner edge correction using 2nd-derivative zero-crossing detection.

Detects the true eyeliner boundary by sampling pixel intensities along a
landmark->delta direction, smoothing, differentiating, and selecting the
zero-crossing with the strongest dark-to-light transition.

Primary path:
    preprocess() -> _sample_intensity_along_ray() -> _find_best_crossing()

If no valid edge is found along P->Q, the original landmark position P is
returned unchanged.
"""
from __future__ import annotations

import weakref

import cv2
import numpy as np

from ..constants import EDGE_LANDMARK_DIRECTIONS, EYELINER_CORRECTION_MAP
from .lash_line_edges import unique_edges
from .types import FaceLandmarks

# Tuning constants for edge detection
_T_STEP = 0.01          # Sampling step along P->Q normalised direction
_T_MAX = 0.5            # Max fraction of P->Q distance to search (t=0 to 0.5)
_KERNEL_HALF = 3        # Gaussian half-width (ksize = 2*half+1 = 7)
_GRAD_THRESHOLD = 0.1   # Minimum 1st-derivative to accept a crossing as an edge


class EyelinerCorrector:
    """Detects true eyeliner edge positions using image-based analysis."""

    # Per-instance correction cache cap to prevent unbounded growth
    # during video / batch processing.
    _MAX_CACHE_SIZE = 16

    # Maximum number of preprocessed images to cache per instance.
    _PREPROCESS_CACHE_MAX = 4

    def __init__(self) -> None:
        # Per-instance preprocess cache so that bilateralFilter is only
        # applied once per source image even when left and right eyeliner
        # parts each have their own corrector instance within the same
        # MakeupProcessor. Instance-level (not class-level) avoids unrelated
        # EyelinerCorrector instances evicting each other's cache entries.
        # Note: EyelinerCorrector itself is not thread-safe (see
        # MakeupProcessor's class docstring).
        self._preprocess_cache: dict = {}

        # Cache the debug entries produced by correct_points so that
        # get_debug_entries can reuse them without re-running find_edge.
        # Structure: {part_name: {'img_ref': weakref, 'entries': list[dict]}}
        # A weakref to the source image is stored instead of id(): if the array
        # is GC'd and CPython reuses its address for a new array, weakref()
        # returns None, so the identity check `weakref() is image` is correct.
        self._correction_cache: dict = {}
        self._cache_order: list = []

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for eyeliner edge detection.

        Results are cached at the class level (keyed by image identity)
        so that bilateralFilter runs at most once per source image even
        when multiple EyelinerCorrector instances process the same frame.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            Preprocessed grayscale image (H, W) uint8, normalized to 0-255.
        """
        cache_key = (id(image), image.ctypes.data)
        cached = self._preprocess_cache.get(cache_key)
        if cached is not None and cached['ref']() is image:
            return cached['result']

        filtered = cv2.bilateralFilter(image, 9, 50, 50)
        gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Evict oldest entry when at capacity (FIFO, dead-weakref first).
        if len(self._preprocess_cache) >= self._PREPROCESS_CACHE_MAX:
            stale = [k for k, v in self._preprocess_cache.items()
                      if v['ref']() is None]
            if stale:
                del self._preprocess_cache[stale[0]]
            else:
                first_key = next(iter(self._preprocess_cache))
                del self._preprocess_cache[first_key]

        self._preprocess_cache[cache_key] = {
            'ref': weakref.ref(image),
            'result': gray,
        }
        return gray

    # ------------------------------------------------------------------
    # Ray sampling + crossing detection
    # ------------------------------------------------------------------

    def _sample_intensity_along_ray(
        self,
        gray: np.ndarray,
        px: float, py: float,
        dx: float, dy: float,
        norm: float,
    ) -> tuple[np.ndarray, list[tuple[float, float]], np.ndarray]:
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
        coords: list[tuple[float, float]],
        t_values: np.ndarray,
        fallback: tuple[int, int],
    ) -> tuple[tuple[int, int], dict]:
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
    ) -> tuple[int, int] | tuple[tuple[int, int], dict | None]:
        """
        Find the eyeliner edge along the P->Q direction using
        2nd-derivative zero-crossing detection.

        Samples intensity from P along P->Q (t=0 to _T_MAX), smooths with a
        Gaussian kernel, computes 1st/2nd derivatives, and selects the
        zero-crossing with the largest positive 1st derivative (dark->light
        transition = eyeliner edge).

        Args:
            gray: Preprocessed grayscale image (H, W) uint8.
            px, py: Start point (landmark position P).
            qx, qy: End point (delta landmark position Q).
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

        if norm < 1e-6:  # guard against sub-px float residue from landmark rounding
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

    # ------------------------------------------------------------------
    # Cache eviction
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest correction-cache entry when at capacity.

        Prioritises entries whose source-image weakref is dead (image
        already garbage-collected).  Falls back to plain FIFO when all
        tracked images are still alive.
        """
        if len(self._correction_cache) < self._MAX_CACHE_SIZE:
            return

        # Priority 1: dead weakref entries
        for key in list(self._cache_order):
            entry = self._correction_cache.get(key)
            if entry is not None and entry['img_ref']() is None:
                del self._correction_cache[key]
                self._cache_order.remove(key)
                return

        # Priority 2: oldest entry (FIFO)
        if self._cache_order:
            oldest = self._cache_order.pop(0)
            self._correction_cache.pop(oldest, None)

    # ------------------------------------------------------------------
    # Batch correction
    # ------------------------------------------------------------------

    def _detect_edges(
        self, gray: np.ndarray, coords: np.ndarray, part_name: str
    ) -> dict[int, dict]:
        """Run edge detection once per distinct edge of *part_name*.

        Returns:
            ``{edge_id: entry}`` in slot order; each entry holds px, py, qx,
            qy, detected_x, detected_y and, when detection ran, the
            derivative debug info.
        """
        entries: dict[int, dict] = {}
        for idx, _is_upper in unique_edges(part_name):
            p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[idx]
            px, py = coords[p_idx]
            qx, qy = coords[q_idx]
            result, dbg = self.find_edge(gray, px, py, qx, qy, return_debug_info=True)
            entry: dict = {
                'px': float(px), 'py': float(py),
                'qx': float(qx), 'qy': float(qy),
                'detected_x': result[0], 'detected_y': result[1],
            }
            if dbg is not None:
                entry.update(dbg)
            entries[idx] = entry
        return entries

    def correct_points(
        self,
        coords: np.ndarray,
        part_name: str,
        eyeliner: np.ndarray,
    ) -> np.ndarray:
        """
        Apply edge-based correction to eyeliner landmark points.

        For each distinct edge landmark in the correction map, finds the true
        eyeliner edge using 2nd-derivative zero-crossing detection along the
        landmark->delta direction (from EDGE_LANDMARK_DIRECTIONS). The edge
        shared by both polylines (1507/1707) is detected once and both
        polylines use that point. Plain int landmarks pass through without
        correction.

        Args:
            coords: All landmark pixel coordinates (N, 2).
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            eyeliner: Source RGB image (H, W, 3) uint8; preprocessed internally.

        Returns:
            np.ndarray: Corrected point coordinates (M, 2) float32,
                        concatenating upper and lower corrected points in order.
        """
        if part_name not in EYELINER_CORRECTION_MAP:
            raise ValueError(
                f"Unknown part_name: '{part_name}'. "
                f"Expected one of {sorted(EYELINER_CORRECTION_MAP.keys())}."
            )
        correction_map = EYELINER_CORRECTION_MAP[part_name]
        gray = self.preprocess(eyeliner)

        edges = self._detect_edges(gray, coords, part_name)
        cache_entries = list(edges.values())
        cache_key = (part_name, id(eyeliner), eyeliner.ctypes.data)
        all_corrected = []
        for indices in correction_map.values():
            for idx in indices:
                if idx in edges:
                    result = (edges[idx]['detected_x'], edges[idx]['detected_y'])
                else:
                    # Plain int landmark — pass through without edge correction.
                    result = (int(coords[idx][0]), int(coords[idx][1]))
                all_corrected.append(result)

        # 'gray' is intentionally NOT stored in the cache entry: storing the full
        # preprocessed frame would pin it in memory for the corrector's lifetime.
        if cache_key in self._correction_cache:
            self._cache_order.remove(cache_key)
        else:
            self._evict_if_needed()
        self._correction_cache[cache_key] = {
            'img_ref': weakref.ref(eyeliner), 'entries': cache_entries
        }
        self._cache_order.append(cache_key)
        return np.array(all_corrected, dtype=np.float32)

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
        Run eyeliner correction and return per-point debug information.

        Args:
            image: Original RGB image (H, W, 3) uint8.
            landmarks: Face landmarks.
            image_shape: (H, W).
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            side: Eye side label ('left' or 'right') stored in each entry.
            eyeliner: Image to use for edge detection (defaults to image).

        Returns:
            List of debug info dicts, one per distinct edge point (the edge
            shared by both polylines appears once). Each dict
            contains px, py, qx, qy, detected_x, detected_y, side, and the
            source image. When edge detection ran, also includes samples,
            first_derivative, second_derivative, t_values, zero_crossings,
            and best_idx.
        """
        if part_name not in EYELINER_CORRECTION_MAP:
            raise ValueError(
                f"Unknown part_name: '{part_name}'. "
                f"Expected one of {sorted(EYELINER_CORRECTION_MAP.keys())}."
            )

        h, w = image_shape
        if eyeliner is None:
            eyeliner = image

        # Reuse debug entries cached by the most recent correct_points call
        # for this part (same source image = same render pass).
        cache_key = (part_name, id(eyeliner), eyeliner.ctypes.data)
        cached = self._correction_cache.get(cache_key)
        if cached is not None and cached['img_ref']() is eyeliner:
            return [dict(entry, side=side, eyeliner=eyeliner) for entry in cached['entries']]

        # Fallback: re-run edge detection with debug info (e.g. when called
        # outside the normal correct_points path).
        coords = landmarks.to_image_coords(w, h)
        edges = self._detect_edges(self.preprocess(eyeliner), coords, part_name)
        return [dict(entry, side=side, eyeliner=eyeliner) for entry in edges.values()]
