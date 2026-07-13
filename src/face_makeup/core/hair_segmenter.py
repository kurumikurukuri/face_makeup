"""
Hair segmentation using OpenCV DNN + MediaPipe hair_segmentation.tflite.

The model expects 256x256 4-channel input — RGB plus the previous frame's
hair mask (a temporal prior for video; all zeros for still images) — and
outputs (1, 2, H, W) logits. Channel 0 = background, channel 1 = hair.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HairSegmenter:
    """Hair segmentation using OpenCV DNN + hair model.

    The model (hair_segmentation.tflite) is loaded via cv2.dnn.readNet.
    Input is RGB + zero prior-mask channel → 256x256 blob.
    Output is sigmoid(chan1) → threshold → uint8 0/255 mask.

    Usage:
        >>> seg = HairSegmenter("models/hair_segmenter.tflite")
        >>> mask = seg.segment(image)  # (H, W) uint8, 255=hair, 0=other
        >>> seg.close()
        >>>
        >>> # Or as context manager:
        >>> with HairSegmenter("models/hair_segmenter.tflite") as seg:
        ...     mask = seg.segment(image)
    """

    _INPUT_SIZE = (256, 256)

    def __init__(self, model_path: str):
        """Create hair segmenter.

        Args:
            model_path: Path to hair_segmenter.tflite model file.
        """
        self._net = cv2.dnn.readNet(model_path)
        logger.debug(f"HairSegmenter initialized (model={model_path})")

    def segment(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Segment hair from an RGB image.

        Args:
            image: RGB uint8 image (H, W, 3).

        Returns:
            (H, W) uint8 mask where 255 = hair, 0 = other, or None on error.
        """
        try:
            h, w = image.shape[:2]

            # The 4th input channel is the PREVIOUS FRAME's hair mask — a
            # temporal prior for video, not alpha. A saturated prior (255)
            # tells the model "everything was hair last frame", dragging the
            # whole person toward hair with striped garbage logits. Still
            # images have no previous frame, so the prior is all zeros.
            model_input = np.dstack([
                image,
                np.zeros((h, w, 1), dtype=np.uint8),
            ])

            # Prepare blob: normalize to [0, 1], resize to 256x256,
            # channel order preserved (swapRB=False)
            blob = cv2.dnn.blobFromImage(
                model_input,
                scalefactor=1.0 / 255.0,
                size=self._INPUT_SIZE,
                mean=(0, 0, 0, 0),
                swapRB=False,
                crop=False,
            )

            self._net.setInput(blob)
            output: np.ndarray = self._net.forward()  # (1, 2, H_out, W_out)

            # Channel 1 = hair logits → sigmoid → threshold
            hair_conf = output[0, 1]  # (H_out, W_out)
            hair_prob = 1.0 / (1.0 + np.exp(-hair_conf))

            # Resize back to original dimensions
            if hair_prob.shape != (h, w):
                hair_prob = cv2.resize(
                    hair_prob, (w, h), interpolation=cv2.INTER_LINEAR
                )

            hair_mask = (hair_prob > 0.5).astype(np.uint8) * 255
            return hair_mask

        except Exception:
            logger.exception("Hair segmentation failed")
            return None

    def close(self) -> None:
        """Release model resources (no-op for OpenCV DNN)."""
        self._net = None
        logger.debug("HairSegmenter closed")

    def __enter__(self) -> HairSegmenter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
