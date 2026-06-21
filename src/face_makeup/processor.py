"""
makeup processing class.

Models are loaded once and can process multiple images.
Note: This class is NOT thread-safe. Create separate instances for concurrent use.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import request

import numpy as np

from face_makeup.core.face_detector import FaceDetector
from face_makeup.core.landmark_extractor import LandmarkExtractor
from face_makeup.rendering_pipeline import RenderingPipeline
from face_makeup.io_handler import IOHandler
from face_makeup.context import ImageContext
from face_makeup.face import Face

logger = logging.getLogger(__name__)

_HAIR_MODEL_FILENAME = "hair_segmenter.tflite"
_HAIR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite"
)


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
                        if pct % 20 < 5 and downloaded % (total_bytes // 5) < 8192:
                            logger.info(
                                "  %d%% (%d / %d bytes)", pct, downloaded, total_bytes
                            )

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


@dataclass
class ProcessorConfig:
    """Configuration class for MakeupProcessor."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    max_num_faces: int = 10


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
            model_dir = Path(__file__).parent.parent.parent / "models"
            model_path = model_dir / _HAIR_MODEL_FILENAME

            if not model_path.exists():
                if not _download_hair_model(model_path):
                    pass  # Download failed — warning already logged

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
        self._rendering_pipeline = RenderingPipeline(hair_segmenter=hair_segmenter)

        # Create I/O handler
        self._io_handler = IOHandler()

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
        """
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
        pil_image = Image.open(path).convert("RGB")
        image_np = np.array(pil_image)
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