"""
Hair segmentation using OpenCV DNN + MediaPipe hair segmentation model.

The model accepts a dynamic input shape; we feed it at 512x512 with 4 channels
— RGB plus the previous frame's hair mask (a temporal prior for video; all
zeros for still images) — and it outputs (1, 2, H, W) logits. Channel 0 =
background, channel 1 = hair.

For high-resolution portraits the whole-frame path downsamples wispy strands
below 1px and they vanish. When a ``face_bbox`` is supplied, ``segment`` instead
splits the face region (face bbox extended upward by headroom) into 4
overlapping quadrants, infers each at the model's native resolution, and
linear-feather-blends them into a seamless hair-probability map. This captures
wispy forehead hair that whole-frame inference misses.
"""
from __future__ import annotations

import logging
import weakref
from typing import Optional

import cv2
import numpy as np

from face_makeup.core.types import FaceBBox

logger = logging.getLogger(__name__)


class HairSegmenter:
    """Hair segmentation using OpenCV DNN + hair model.

    The model (hair_segmenter.tflite) is loaded via cv2.dnn.readNet. It
    accepts a dynamic input shape; inference runs at 512x512. Input is RGB +
    a zero prior-mask channel -> 512x512 blob. Output is sigmoid(chan1) ->
    anti-alias blur -> threshold -> uint8 0/255 mask.

    With ``face_bbox`` set, ``segment`` runs 4-quadrant tiled inference around
    the face to resolve wispy hair at near-native resolution. Without it,
    whole-frame inference is used (legacy behavior).

    Thread safety: the per-(image, bbox) mask cache is not locked; concurrent
    ``segment`` calls on one instance must be serialised by the caller.

    Usage:
        >>> seg = HairSegmenter("models/hair_segmenter.tflite")
        >>> mask = seg.segment(image)                         # whole-frame
        >>> mask = seg.segment(image, face_bbox=face.bbox)    # 4-quadrant tiled
        >>> seg.close()
        >>>
        >>> # Or as context manager:
        >>> with HairSegmenter("models/hair_segmenter.tflite") as seg:
        ...     mask = seg.segment(image, face_bbox=bbox)
    """

    _INPUT_SIZE = (512, 512)
    # Anti-alias blur on the probability map at model-native resolution before
    # resizing back (suppresses the transposed-conv checkerboard). Empirically
    # tuned for 512x512 inference; if _INPUT_SIZE changes, revisit this too.
    _BLUR_SIGMA = 2.0
    # Tiling parameters (face_bbox path).
    _TILE_OVERLAP = 0.2          # overlap fraction between quadrants
    _TOP_HEADROOM = 0.5          # region extends above face bbox by this * bbox_h
    _MIN_REGION_FOR_TILING = 4   # below this per axis, fall back to whole-frame
    # Per-(image, face_bbox) mask cache. The hair mask depends only on the
    # source image and the face bbox, never on makeup params, so it is reused
    # across renders (eyeliner_corrector's id+weakref+FIFO idiom). Without it
    # every makeup-param tweak would re-run the 4 tiled forwards.
    _MASK_CACHE_MAX = 8

    def __init__(self, model_path: str):
        """Create hair segmenter.

        Args:
            model_path: Path to hair_segmenter.tflite model file.
        """
        self._net = cv2.dnn.readNet(model_path)
        self._mask_cache: dict = {}
        logger.debug("HairSegmenter initialized (model=%s)", model_path)

    def segment(
        self,
        image: np.ndarray,
        face_bbox: Optional[FaceBBox] = None,
    ) -> Optional[np.ndarray]:
        """Segment hair from an RGB image.

        Results are memoised per (image identity, face_bbox): the hair mask
        depends only on the source pixels and the face bbox, never on makeup
        params, so repeated renders (e.g. tweaking a Streamlit slider) reuse
        the cached mask instead of re-running inference.

        Args:
            image: RGB uint8 image (H, W, 3). Must not be mutated in place
                after this call — results are cached by array identity, so
                overwriting the buffer would make a later ``segment`` return a
                stale mask. Pass a fresh copy if you need to reuse a buffer.
            face_bbox: Face bounding box. When provided, hair is inferred by
                4-quadrant tiling around the face region (resolves wispy hair
                at near-native resolution). When None, whole-frame inference
                is used.

        Returns:
            (H, W) uint8 mask where 255 = hair, 0 = other, or None on error.
        """
        # Drop entries whose source image was GC'd so they don't linger until
        # the cache fills up (matters in long sessions with many images).
        self._drop_dead_refs()
        key = (id(image), image.ctypes.data, self._bbox_key(face_bbox))
        cached = self._mask_cache.get(key)
        if cached is not None and cached["ref"]() is image:
            return cached["mask"].copy()

        try:
            h, w = image.shape[:2]
            if face_bbox is None:
                prob = self._infer_tile(image)
            else:
                region = self._region_from_bbox((h, w), face_bbox)
                rw, rh = region[2] - region[0], region[3] - region[1]
                if rw < self._MIN_REGION_FOR_TILING or rh < self._MIN_REGION_FOR_TILING:
                    # Degenerate region (tiny/edge face): fall back to whole-frame
                    # rather than slice zero-size tiles.
                    prob = self._infer_tile(image)
                else:
                    prob = self._infer_region_quadrants(image, region)
            mask = (prob > 0.5).astype(np.uint8) * 255
        except (cv2.error, MemoryError):
            logger.exception("Hair segmentation failed")
            return None

        # Only successful masks are cached; failures stay uncached so the next
        # call retries instead of returning None forever.
        self._evict_if_full()
        self._mask_cache[key] = {"ref": weakref.ref(image), "mask": mask}
        return mask.copy()

    def _infer_tile(self, tile: np.ndarray) -> np.ndarray:
        """Run the model on one sub-image; return hair probability at tile size.

        The 4th input channel is the previous frame's hair mask — a temporal
        prior for video, not alpha. A saturated prior (255) tells the model
        "everything was hair last frame", dragging the whole person toward hair.
        Still images have no previous frame, so the prior is all zeros.
        """
        hh, ww = tile.shape[:2]
        model_input = np.dstack([tile, np.zeros((hh, ww, 1), dtype=np.uint8)])
        blob = cv2.dnn.blobFromImage(
            model_input,
            scalefactor=1.0 / 255.0,
            size=self._INPUT_SIZE,
            mean=(0, 0, 0, 0),
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        out = self._net.forward()  # (1, 2, H_out, W_out)
        prob = 1.0 / (1.0 + np.exp(-out[0, 1]))  # channel 1 = hair logits -> sigmoid
        # Anti-alias at native resolution before resize (checkerboard fix).
        prob = cv2.GaussianBlur(
            prob, (0, 0), sigmaX=self._BLUR_SIGMA, sigmaY=self._BLUR_SIGMA,
        )
        if prob.shape != (hh, ww):
            prob = cv2.resize(prob, (ww, hh), interpolation=cv2.INTER_LINEAR)
        return prob.astype(np.float32)

    def _region_from_bbox(
        self,
        image_hw: tuple[int, int],
        face_bbox: FaceBBox,
    ) -> tuple[int, int, int, int]:
        """Build the inference region: face bbox extended upward by headroom.

        Returns (x0, y0, x1, y1) clamped to the image, where y0 is pushed above
        the bbox top so the hairline / hair mass above the forehead is included.
        """
        bx, by, bw, bh = face_bbox.x, face_bbox.y, face_bbox.w, face_bbox.h
        h, w = image_hw
        # Floor the headroom at 1px so a tiny bbox still extends above the
        # hairline instead of collapsing to a zero-width band.
        headroom = max(1, int(bh * self._TOP_HEADROOM))
        x0 = max(0, bx)
        y0 = max(0, by - headroom)
        x1 = min(w, bx + bw)
        y1 = min(h, by + bh)
        return x0, y0, x1, y1

    def _infer_region_quadrants(
        self,
        image: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Split ``region`` into 4 overlapping quadrants, infer each, blend seams.

        Each quadrant extends ``overlap`` pixels toward the centre. A separable
        bilinear fade window (outer product of two 1D ramps) fades each tile
        from 1→0 across its inner edges. Because multiple tiles overlap at the
        seams the raw weight sum is *not* constant — it varies 1→2→1 across
        edge overlaps and reaches 4 at the centre cross. The final probability
        map is ``acc / wsum``, i.e. a per-pixel weighted local average, which
        normalises into a seamless, monotonic (but non-linear) blend at full
        image resolution.

        Returns a float32 (H, W) probability map (caller thresholds).
        """
        h, w = image.shape[:2]
        x0, y0, x1, y1 = region
        rw, rh = x1 - x0, y1 - y0
        half_w, half_h = rw // 2, rh // 2
        # Floor the overlap at 1px so small regions still feather their seams
        # instead of abutting with a hard step (int() truncates to 0 below 5).
        ext_w = max(1, int(half_w * self._TILE_OVERLAP))
        ext_h = max(1, int(half_h * self._TILE_OVERLAP))

        # (ty0, ty1, tx0, tx1) per quadrant — TL, TR, BL, BR.
        tiles = [
            (y0,                  y0 + half_h + ext_h, x0,                  x0 + half_w + ext_w),
            (y0,                  y0 + half_h + ext_h, x0 + half_w - ext_w, x1),
            (y0 + half_h - ext_h, y1,                  x0,                  x0 + half_w + ext_w),
            (y0 + half_h - ext_h, y1,                  x0 + half_w - ext_w, x1),
        ]

        acc = np.zeros((h, w), dtype=np.float32)
        wsum = np.zeros((h, w), dtype=np.float32)
        for ty0, ty1, tx0, tx1 in tiles:
            prob = self._infer_tile(image[ty0:ty1, tx0:tx1])
            th, tw = prob.shape
            ry = np.ones(th, dtype=np.float32)
            rx = np.ones(tw, dtype=np.float32)
            if ext_h > 0:
                if ty0 == y0:
                    ry[-ext_h:] = np.linspace(1, 0, ext_h)   # top edge -> grows down
                else:
                    ry[:ext_h] = np.linspace(0, 1, ext_h)    # bottom edge -> grows up
            if ext_w > 0:
                if tx0 == x0:
                    rx[-ext_w:] = np.linspace(1, 0, ext_w)
                else:
                    rx[:ext_w] = np.linspace(0, 1, ext_w)
            window = np.outer(ry, rx)
            acc[ty0:ty1, tx0:tx1] += prob * window
            wsum[ty0:ty1, tx0:tx1] += window

        return acc / np.maximum(wsum, 1e-6)

    @staticmethod
    def _bbox_key(face_bbox: Optional[FaceBBox]) -> Optional[tuple[int, int, int, int]]:
        """Hashable cache key for a face bbox (None stays None)."""
        if face_bbox is None:
            return None
        return (face_bbox.x, face_bbox.y, face_bbox.w, face_bbox.h)

    def _drop_dead_refs(self) -> None:
        """Remove cache entries whose source image was garbage-collected.

        Called at the top of every ``segment`` so dead weakrefs are reclaimed
        promptly instead of lingering until the cache reaches capacity.
        """
        stale = [k for k, v in self._mask_cache.items() if v["ref"]() is None]
        for k in stale:
            del self._mask_cache[k]

    def _evict_if_full(self) -> None:
        """Drop the oldest cache entry (FIFO) when at capacity.

        Dead weakrefs are already reclaimed at the top of ``segment`` by
        ``_drop_dead_refs``, so every remaining entry here is live — eviction
        is plain FIFO. Storing a weakref instead of relying on id() alone
        avoids false hits when CPython reuses a freed array's address.
        """
        if len(self._mask_cache) < self._MASK_CACHE_MAX:
            return
        del self._mask_cache[next(iter(self._mask_cache))]

    def close(self) -> None:
        """Release model resources and clear the mask cache."""
        self._net = None
        self._mask_cache.clear()
        logger.debug("HairSegmenter closed")

    def __enter__(self) -> HairSegmenter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # __del__ must never raise (interpreter state is tearing down), but
            # leave a trace so a release failure is still diagnosable.
            logger.debug("HairSegmenter __del__ close failed", exc_info=True)
