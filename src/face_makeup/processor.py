"""
makeup processing class.

Models are loaded once and can process multiple images.
Note: This class is NOT thread-safe. Create separate instances for concurrent use.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import request

import numpy as np

from face_makeup.core.face_detector import FaceDetector
from face_makeup.core.landmark_correction import (
    LANDMARK_CORRECTION_MODES,
    LandmarkCorrection,
    LashLineCorrector,
)
from face_makeup.core.landmark_extractor import LandmarkExtractor
from face_makeup.core.model_files import resolve_model_dir as _resolve_model_dir
from face_makeup.rendering_pipeline import RenderingPipeline
from face_makeup.io_handler import IOHandler
from face_makeup.context import ImageContext
from face_makeup.face import Face

logger = logging.getLogger(__name__)

_HAIR_MODEL_FILENAME = "hair_segmenter.tflite"
_HAIR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite"
)
# Known-good SHA256 of the model file. Verified at download time and on every
# load to defend against CDN compromise, proxy injection, or local tampering
# that could substitute a malicious .tflite (cv2.dnn reads serialized weights,
# so a tampered model is a parser-exploit surface).
_HAIR_MODEL_SHA256 = (
    "0bec40bc9ba97c4143f3d4225a935014abffea37c1f3766ae32aba3f2748e711"
)


def _verify_model_checksum(model_path: Path) -> bool:
    """Verify the SHA256 checksum of the hair segmentation model.

    Defends against CDN compromise, proxy injection, or local tampering that
    could substitute a malicious ``.tflite``. The digest is the fixed
    known-good value recorded when the model was first vetted.

    Args:
        model_path: Path to the model file to verify.

    Returns:
        True if the file matches ``_HAIR_MODEL_SHA256``, False otherwise
        (including when the file does not exist).
    """
    if not model_path.exists():
        return False
    digest = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != _HAIR_MODEL_SHA256:
        logger.warning(
            "Hair model checksum mismatch: expected %s, got %s",
            _HAIR_MODEL_SHA256,
            actual,
        )
        return False
    return True


def _download_hair_model(model_path: Path) -> bool:
    """Download hair segmentation model.

    Uses urllib (stdlib) — no extra dependencies needed.
    Downloads to a temp file and renames atomically to avoid leaving
    a partial file on interrupt.

    Args:
        model_path: Destination path for the model file.

    Returns:
        True if download succeeded, False otherwise.
    """
    model_dir = model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".download")

    logger.info(
        "Downloading %s from %s ...",
        _HAIR_MODEL_FILENAME,
        _HAIR_MODEL_URL,
    )

    try:
        req = request.Request(_HAIR_MODEL_URL)
        with request.urlopen(req, timeout=60) as resp:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total else None
            downloaded = 0

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded * 100 // total_bytes
                        log_interval = max(total_bytes // 5, 1)
                        if pct % 20 < 5 and downloaded % log_interval < 8192:
                            logger.info(
                                "  %d%% (%d / %d bytes)", pct, downloaded, total_bytes
                            )

            if not _verify_model_checksum(tmp_path):
                logger.warning(
                    "Downloaded %s checksum mismatch — possible tampering. "
                    "File discarded; hair exclusion will be skipped.",
                    _HAIR_MODEL_FILENAME,
                )
                tmp_path.unlink(missing_ok=True)
                return False

            os.replace(tmp_path, model_path)
            logger.info(
                "Downloaded %s (%d bytes)",
                _HAIR_MODEL_FILENAME,
                downloaded,
            )
            return True

    except Exception:
        logger.warning(
            "Failed to download %s. "
            "Hair exclusion will be skipped. "
            "Download it manually with: "
            "curl -L -o models/%s %s",
            _HAIR_MODEL_FILENAME,
            _HAIR_MODEL_FILENAME,
            _HAIR_MODEL_URL,
            exc_info=True,
        )
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


# Maximum decoded pixel count accepted by open()/open_path(). Guards against
# decompression-bomb DoS — a modest compressed upload can expand to hundreds
# of MB of RGB. ~24M px (e.g. 6000x4000) is ample for face photos.
_MAX_IMAGE_PIXELS = 24_000_000


@dataclass
class ProcessorConfig:
    """Configuration class for MakeupProcessor.

    Args:
        min_detection_confidence: Minimum confidence for face detection (0.0-1.0).
        min_tracking_confidence: Minimum confidence for landmark tracking (0.0-1.0).
        refine_landmarks: Whether to apply landmark refinement.
        max_num_faces: Maximum number of faces to detect.
        model_dir: Custom directory for the hair segmentation model file.
            When ``None`` (default), the directory is resolved via the
            :func:`_resolve_model_dir` priority chain (env var, source
            checkout, platform cache).
        landmark_correction: Lash-line correction of the eye edge landmarks,
            applied once per face and shared by eyeliner, eyeshadow, tear bag
            and the foundation's eye exclusion.
            ``'ml'`` (default) uses the EyeContourRefiner ONNX model
            (fetched from EyeContourRefiner main when face_makeup is
            installed, never while rendering) and raises when the installed
            model is missing or cannot be loaded; there is no fallback. ``'rule'`` uses the
            2nd-derivative zero-crossing detector.
    """
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    max_num_faces: int = 10
    model_dir: Optional[str] = None
    landmark_correction: LandmarkCorrection = "ml"

    def __post_init__(self):
        """Validate parameter bounds."""
        if not isinstance(self.min_detection_confidence, (int, float)) or not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError(
                f"min_detection_confidence must be 0.0-1.0, got {self.min_detection_confidence}"
            )
        if not isinstance(self.min_tracking_confidence, (int, float)) or not 0.0 <= self.min_tracking_confidence <= 1.0:
            raise ValueError(
                f"min_tracking_confidence must be 0.0-1.0, got {self.min_tracking_confidence}"
            )
        if not isinstance(self.max_num_faces, int) or self.max_num_faces < 1:
            raise ValueError(f"max_num_faces must be a positive integer, got {self.max_num_faces}")
        if self.model_dir is not None and not isinstance(self.model_dir, str):
            raise ValueError(
                f"model_dir must be a string or None, got {type(self.model_dir).__name__}"
            )
        if self.landmark_correction not in LANDMARK_CORRECTION_MODES:
            raise ValueError(
                "landmark_correction must be 'ml' or 'rule', "
                f"got {self.landmark_correction!r}"
            )


class MakeupProcessor:
    """
    makeup processor.
    Models are initialized once and can process multiple images.

    Example:
        >>> processor = MakeupProcessor()
        >>> ctx = processor.open_path("sample.jpg")
        >>> ctx.faces[0].lip(lip_params).cheek(cheek_params)
        >>> result = ctx.render()
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize processor and load models.

        Args:
            config: Processor configuration. Uses default if None.
        """
        self._config = config or ProcessorConfig()

        # Initialize face detection model
        self._detector = FaceDetector(
            min_detection_confidence=self._config.min_detection_confidence
        )

        # Initialize landmark extraction model
        self._landmark_extractor = LandmarkExtractor(
            max_num_faces=self._config.max_num_faces,
            refine_landmarks=self._config.refine_landmarks,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence
        )

        # Create hair segmenter (auto-download if model is missing)
        hair_segmenter = None
        try:
            model_dir = _resolve_model_dir(self._config.model_dir)
            model_path = model_dir / _HAIR_MODEL_FILENAME

            if not model_path.exists():
                _download_hair_model(model_path)  # warning logged inside on failure
            elif not _verify_model_checksum(model_path):
                logger.warning(
                    "Existing %s checksum mismatch — possible tampering. "
                    "Re-downloading.",
                    _HAIR_MODEL_FILENAME,
                )
                model_path.unlink(missing_ok=True)
                _download_hair_model(model_path)  # warning logged inside on failure

            if model_path.exists():
                from face_makeup.core.hair_segmenter import HairSegmenter
                hair_segmenter = HairSegmenter(str(model_path))
                logger.info("HairSegmenter loaded for hair exclusion")
        except ImportError:
            logger.warning(
                "HairSegmenter dependencies not available, skipping hair exclusion"
            )
        except Exception:
            logger.warning(
                "Failed to initialize HairSegmenter, skipping hair exclusion",
                exc_info=True,
            )

        # Create rendering pipeline (shared across all contexts)
        lash_corrector = LashLineCorrector(self._config.landmark_correction)
        self._rendering_pipeline = RenderingPipeline(
            hair_segmenter=hair_segmenter,
            lash_corrector=lash_corrector,
        )

        # Create I/O handler
        self._io_handler = IOHandler()

        self._closed = False

        logger.info("MakeupProcessor initialized")

    @property
    def config(self) -> ProcessorConfig:
        """Get current processor configuration."""
        return self._config


    def open(self, image: np.ndarray) -> ImageContext:
        """
        Open ndarray image and generate ImageContext for makeup processing.

        Performs face detection and landmark extraction.

        Args:
            image: RGB format uint8 image (H, W, 3).

        Returns:
            ImageContext: Context for subsequent processing.

        Raises:
            RuntimeError: If the processor has already been closed.
        """
        if self._closed:
            raise RuntimeError("MakeupProcessor is closed")
        self._validate_image(image)

        logger.debug(f"Opening image: shape={image.shape}")

        # Execute face detection
        bboxes = self._detector.detect(image)
        logger.debug(f"Number of faces detected: {len(bboxes)}")

        # Extract landmarks for all faces
        landmarks_list = self._landmark_extractor.extract_all(image)
        logger.debug(f"Number of landmarks extracted: {len(landmarks_list)}")

        # Extend with custom (virtual) landmarks
        from face_makeup.core.landmark_resolver import extend_landmarks
        landmarks_list = [extend_landmarks(lm) for lm in landmarks_list]

        # Use minimum count if bbox and landmarks counts differ
        num_faces = min(len(bboxes), len(landmarks_list))

        if len(bboxes) != len(landmarks_list):
            logger.warning(
                f"Face count mismatch: bbox={len(bboxes)}, "
                f"landmarks={len(landmarks_list)}. "
                f"Using only {num_faces}."
            )

        # Generate Face objects
        faces = []
        for i in range(num_faces):
            face = Face(
                face_id=i,
                bbox=bboxes[i],
                landmarks=landmarks_list[i],
                change_callback=None  # Set by ImageContext
            )
            faces.append(face)

        # Generate ImageContext with shared pipeline and handler
        ctx = ImageContext(image, faces, self._rendering_pipeline, self._io_handler)

        logger.info(f"Generated ImageContext with {len(faces)} faces")
        return ctx

    def open_path(self, image_source: str) -> ImageContext:
        """
        Load image from file path and generate ImageContext.

        Args:
            image_source: Image file path.

        Returns:
            ImageContext.
        """
        path = Path(image_source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_source}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_source}")

        # Load image in RGB format using PIL (lazy import to avoid
        # requiring Pillow when only the ndarray-based open() is used)
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required for open_path(). "
                "Install it with: pip install Pillow"
            ) from e
        # Reject decompression bombs before Pillow decodes pixel data, rather
        # than relying solely on the post-decode check in _validate_image.
        # Image.MAX_IMAGE_PIXELS is a process-wide PIL setting, not scoped to
        # this call, so it's restored afterward to avoid capping unrelated
        # PIL usage elsewhere in the same process.
        previous_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
        try:
            with Image.open(path) as pil_image:
                image_np = np.array(pil_image.convert("RGB"))
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_pixels
        return self.open(image_np)
    
    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate input image."""
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        if image.dtype != np.uint8:
            raise ValueError(f"Image dtype must be uint8: {image.dtype}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image shape must be (H, W, 3): {image.shape}")

        height, width = int(image.shape[0]), int(image.shape[1])
        num_pixels = height * width
        if num_pixels > _MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image too large: {width}x{height} ({num_pixels} px) "
                f"exceeds {_MAX_IMAGE_PIXELS} pixel limit"
            )

    def close(self) -> None:
        """Release all resources held by this processor.

        Closes the face detector, landmark extractor, and rendering
        pipeline (including the hair segmenter if loaded).  Idempotent —
        safe to call multiple times. After close(), open()/open_path()
        raise RuntimeError instead of hitting opaque MediaPipe errors on
        already-closed internal objects.
        """
        try:
            self._detector.close()
        except Exception:
            logger.warning("Error closing FaceDetector", exc_info=True)
        try:
            self._landmark_extractor.close()
        except Exception:
            logger.warning("Error closing LandmarkExtractor", exc_info=True)
        try:
            self._rendering_pipeline.close()
        except Exception:
            logger.warning("Error closing RenderingPipeline", exc_info=True)
        self._closed = True

    def __enter__(self) -> "MakeupProcessor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()