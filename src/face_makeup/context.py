"""
ImageContext for per-image makeup processing state.

Holds the original image and detected faces.
Manages rendering and output operations.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from face_makeup.face import Face

if TYPE_CHECKING:
    from face_makeup.rendering_pipeline import RenderingPipeline
    from face_makeup.io_handler import IOHandler

logger = logging.getLogger(__name__)


class ImageContext:
    """
    Per-image processing context.

    Holds the original image and detected faces.
    Manages rendering and output operations.

    Example:
        >>> ctx = processor.open(image)
        >>> ctx.faces[0].lip(lip_params).cheek(cheek_params)
        >>> result = ctx.render()
        >>> ctx.save("output.png")
    """

    def __init__(
        self,
        image: np.ndarray,
        faces: list[Face],
        rendering_pipeline: RenderingPipeline,
        io_handler: IOHandler | None = None
    ):
        """
        Initialize image context.

        Args:
            image: Original RGB image (H, W, 3).
            faces: List of detected Face objects.
            rendering_pipeline: RenderingPipeline for rendering makeup.
            io_handler: Optional IOHandler for save/show operations.
        """
        self._original_image = image.copy()
        self._faces = faces
        self._rendering_pipeline = rendering_pipeline
        self._io_handler = io_handler

        # Set change callbacks for all faces
        for face in self._faces:
            face.set_change_callback(self._invalidate_cache)

        logger.debug("ImageContext created with %d faces", len(faces))

    @property
    def original_image(self) -> np.ndarray:
        """Get copy of original image."""
        return self._original_image.copy()

    @property
    def faces(self) -> list[Face]:
        """Get list of detected faces (shallow copy)."""
        return list(self._faces)

    @property
    def face_count(self) -> int:
        """Get number of detected faces."""
        return len(self._faces)

    def face(self, index: int = 0) -> Face:
        """
        Get face by index.

        Args:
            index: Face index (0-based).

        Returns:
            Face object.

        Raises:
            IndexError: If index out of range.
        """
        if index < 0 or index >= len(self._faces):
            if not self._faces:
                raise IndexError(f"Face index {index} out of range (no faces detected)")
            raise IndexError(f"Face index {index} out of range (0-{len(self._faces) - 1})")
        return self._faces[index]

    @property
    def _io(self) -> IOHandler:
        """Lazily initialize and return IOHandler."""
        if self._io_handler is None:
            from face_makeup.io_handler import IOHandler
            self._io_handler = IOHandler()
        return self._io_handler

    def render(self, debug: bool = False) -> np.ndarray:
        """
        Render all makeup layers to produce final image.

        Delegates to RenderingPipeline which handles caching.

        Args:
            debug: Include debug overlays (landmarks, masks).

        Returns:
            Rendered RGB image with all makeup applied.
        """
        return self._rendering_pipeline.render(
            self,
            self._original_image,
            self._faces,
            debug=debug
        )

    def save(
        self,
        path: str,
        debug: bool = False
    ) -> None:
        """
        Save rendered image to file.

        Args:
            path: Output file path.
            debug: If True, save debug images to outputs/debug/.
        """
        result = self.render(debug=debug)
        self._io.save(result, path)

    def show(
        self,
        window_name: str = "preview",
        debug: bool = False,
        wait_key: int = 0
    ) -> None:
        """
        Display rendered image in window.

        Args:
            window_name: OpenCV window name.
            debug: Show debug overlays.
            wait_key: cv2.waitKey duration (0 = wait indefinitely).
        """
        result = self.render(debug=debug)
        self._io.show(result, window_name, wait_key)

    def _invalidate_cache(self) -> None:
        """Clear rendered cache (called when makeup changes)."""
        self._rendering_pipeline.invalidate_cache(self)
        logger.debug("Render cache invalidated")
