"""
I/O handler for saving and displaying images.

Provides utility functions for file operations and display.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp',
})


class IOHandler:
    """
    Handles I/O operations for makeup processing results.

    Provides methods for saving and displaying images,
    handling RGB to BGR conversion for OpenCV compatibility.

    Example:
        >>> handler = IOHandler()
        >>> result_image = ctx.render()  # RGB image
        >>> handler.save(result_image, "output.jpg")
        >>> handler.show(result_image)
    """

    def _validate_save_path(self, path: str) -> Path:
        """
        Validate output file path.

        Args:
            path: Output file path.

        Returns:
            Resolved Path object.

        Raises:
            ValueError: If path is empty or has unsupported extension.
            FileNotFoundError: If parent directory does not exist.
        """
        if not path or not path.strip():
            raise ValueError("Output path must not be empty")

        resolved = Path(path).resolve()

        ext = resolved.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{ext}'. "
                f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
            )

        if not resolved.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {resolved.parent}"
            )

        return resolved

    def save(self, image: np.ndarray, path: str) -> None:
        """
        Save rendered image to file.

        Converts RGB image to BGR format for OpenCV compatibility.

        Args:
            image: RGB image to save (H, W, 3).
            path: Output file path.

        Raises:
            ValueError: If path is empty or has unsupported extension.
            FileNotFoundError: If parent directory does not exist.
            OSError: If cv2.imwrite fails.
        """
        resolved = self._validate_save_path(path)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(resolved), image_bgr)
        if not success:
            raise OSError(f"Failed to write image to {resolved}")

        logger.info("Saved result to %s", path)

    def show(self, image: np.ndarray, window_name: str = "preview", wait_key: int = 0) -> None:
        """
        Display rendered image in window.

        Converts RGB image to BGR format for OpenCV compatibility.

        Args:
            image: RGB image to display (H, W, 3).
            window_name: OpenCV window name.
            wait_key: cv2.waitKey duration (0 = wait indefinitely).
        """
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow(window_name, image_bgr)
        cv2.waitKey(wait_key)

        logger.debug("Showing preview in window: %s", window_name)
