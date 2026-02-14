"""
makeup processing class.

Models are loaded once and can process multiple images.
Note: This class is NOT thread-safe. Create separate instances for concurrent use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np

from face_makeup.core.face_detector import FaceDetector
from face_makeup.core.landmark_extractor import LandmarkExtractor
from face_makeup.core.segmenter import FaceSegmenter
from face_makeup.core.makeup_blender import MakeupBlender
from face_makeup.rendering_pipeline import RenderingPipeline
from face_makeup.io_handler import IOHandler
from face_makeup.context import ImageContext
from face_makeup.face import Face

logger = logging.getLogger(__name__)


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

        # Create rendering pipeline (shared across all contexts)
        segmenter = FaceSegmenter()
        blender = MakeupBlender()
        self._rendering_pipeline = RenderingPipeline(segmenter, blender)

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