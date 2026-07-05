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

    Relative save paths are validated to stay within *output_base* to
    prevent path traversal. Absolute paths are honoured as explicit user
    intent — callers embedding this library in a service should only accept
    relative paths from untrusted input.

    Example:
        >>> handler = IOHandler()
        >>> result_image = ctx.render()  # RGB image
        >>> handler.save(result_image, "output.jpg")
        >>> handler.show(result_image)
    """

    def __init__(self, output_base: str | Path = ".") -> None:
        """
        Initialize I/O handler.

        Args:
            output_base: Base directory for output files. All save paths are
                resolved relative to this directory and validated to stay
                within it. Defaults to the current working directory.
        """
        self._output_base = Path(output_base).resolve()
        logger.debug("IOHandler initialized (output_base=%s)", self._output_base)

    @property
    def output_base(self) -> Path:
        """Return the resolved output base directory."""
        return self._output_base

    def _validate_save_path(self, path: str) -> Path:
        """
        Validate output file path and resolve it within *output_base*.

        Args:
            path: Output file path (relative or absolute).

        Returns:
            Resolved Path object. For relative inputs, guaranteed to be
            within *output_base*; absolute inputs are returned as-is.

        Raises:
            ValueError: If path is empty, has an unsupported extension, or
                escapes the output base directory.
            FileNotFoundError: If parent directory does not exist.
        """
        if not path or not path.strip():
            raise ValueError("Output path must not be empty")

        # Resolve the path relative to output_base, then resolve any symlinks
        # and ".." segments to get the canonical absolute path.
        # When *path* is absolute, / operator returns *path* unchanged, so
        # absolute paths bypass the output_base containment check (they
        # represent explicit user intent).
        resolved = (self._output_base / path).resolve()

        # Prevent path traversal for relative paths: the resolved path must
        # stay within output_base. Absolute paths are exempt — the caller
        # has explicitly chosen the destination.
        if not Path(path).is_absolute():
            if not resolved.is_relative_to(self._output_base):
                raise ValueError(
                    f"Path escapes output directory: '{path}' "
                    f"resolves to '{resolved}' which is outside "
                    f"'{self._output_base}'"
                )

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
            path: Output file path relative to output_base.

        Raises:
            ValueError: If path is empty, has an unsupported extension, or
                escapes the output base directory.
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
