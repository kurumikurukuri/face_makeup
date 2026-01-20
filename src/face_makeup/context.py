"""
ImageContext for per-image makeup processing state.

Holds the original image and detected faces.
Manages rendering and output operations.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np

from src.face_makeup.face import Face
from src.face_makeup.renderer import Renderer

if TYPE_CHECKING:
    from src.face_makeup.processor import MakeupProcessor

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
        faces: List[Face],
        processor: MakeupProcessor
    ):
        """
        Initialize image context.

        Args:
            image: Original RGB image (H, W, 3).
            faces: List of detected Face objects.
            processor: Reference to parent processor (for rendering components).
        """
        self._original_image = image.copy()
        self._faces = faces
        self._processor = processor
        self._rendered_cache: Optional[np.ndarray] = None

        # Set context reference in all faces
        for face in self._faces:
            face._set_context(self)

        logger.debug(f"ImageContext created with {len(faces)} faces")

    @property
    def original_image(self) -> np.ndarray:
        """Get copy of original image."""
        return self._original_image.copy()

    @property
    def faces(self) -> List[Face]:
        """Get list of detected faces."""
        return self._faces

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
            raise IndexError(f"Face index {index} out of range (0-{len(self._faces) - 1})")
        return self._faces[index]

    def render(self, debug: bool = False) -> np.ndarray:
        """
        Render all makeup layers to produce final image.

        This is where lazy evaluation happens - all Face makeup_layers
        are applied to the original image.

        Args:
            debug: Include debug overlays (landmarks, masks).

        Returns:
            Rendered RGB image with all makeup applied.
        """
        # Return cached result if available and not debug mode
        if self._rendered_cache is not None and not debug:
            logger.debug("Returning cached render result")
            return self._rendered_cache.copy()

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create renderer with debug visualizer if needed
        visualizer = None
        if debug:
            from src.face_makeup.core.debug_visualizer import DebugVisualizer
            visualizer = DebugVisualizer()

        renderer = Renderer(
            segmenter=self._processor._segmenter,
            blender=self._processor._blender,
            visualizer=visualizer
        )

        result = renderer.compose(
            self._original_image,
            self._faces,
            debug=debug,
            session_id=session_id
        )

        # Cache result (only if not debug mode)
        if not debug:
            self._rendered_cache = result.copy()

        logger.debug("Render complete")
        return result

    def save(
        self,
        path: str,
        debug: bool = False,
        debug_dir: Optional[str] = None
    ) -> None:
        """
        Save rendered image to file.

        Args:
            path: Output file path.
            debug: If True, save debug images to debug_dir.
            debug_dir: Directory for debug images (default: outputs/debug).
        """
        result = self.render(debug=debug)

        # Convert RGB to BGR for OpenCV
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, result_bgr)

        logger.info(f"Saved result to {path}")

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

        # Convert RGB to BGR for OpenCV
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, result_bgr)
        cv2.waitKey(wait_key)

        logger.debug(f"Showing preview in window: {window_name}")

    def invalidate_cache(self) -> None:
        """Clear rendered cache (called when makeup changes)."""
        self._rendered_cache = None
        logger.debug("Render cache invalidated")
