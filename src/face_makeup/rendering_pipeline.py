"""
RenderingPipeline for managing makeup rendering and caching.

Encapsulates rendering logic with per-context caching using WeakKeyDictionary.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, List
from weakref import WeakKeyDictionary

import numpy as np

from face_makeup.renderer import Renderer

if TYPE_CHECKING:
    from face_makeup.context import ImageContext
    from face_makeup.face import Face

logger = logging.getLogger(__name__)


class RenderingPipeline:
    """
    Rendering pipeline that manages makeup rendering with per-context caching.

    Uses WeakKeyDictionary for automatic cleanup of cached results when
    ImageContext instances are garbage collected.

    Example:
        >>> pipeline = RenderingPipeline()
        >>> ctx = processor.open(image)  # ImageContext uses pipeline
        >>> result = ctx.render()  # Uses cached result if available
    """

    def __init__(self):
        """
        Initialize rendering pipeline.
        """
        # Cache keyed by ImageContext object (auto-cleanup via weak references)
        self._cache: WeakKeyDictionary[ImageContext, np.ndarray] = WeakKeyDictionary()

        logger.debug("RenderingPipeline initialized")


    def render(
        self,
        context: ImageContext,
        original_image: np.ndarray,
        faces: List[Face],
        debug: bool = False
    ) -> np.ndarray:
        """
        Render all makeup layers to produce final image.

        This method handles caching - cached results are returned if available
        and debug mode is off.

        Args:
            context: ImageContext for caching purposes.
            original_image: Original RGB image (H, W, 3).
            faces: List of Face objects with makeup layers.
            debug: Include debug overlays (landmarks, masks).

        Returns:
            Rendered RGB image with all makeup applied.
        """
        # Return cached result if available and not debug mode
        if context in self._cache and not debug:
            logger.debug("Returning cached render result")
            return self._cache[context].copy()

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create renderer with debug visualizer if needed
        visualizer = None
        if debug:
            from face_makeup.core.debug_visualizer import DebugVisualizer
            visualizer = DebugVisualizer()

        renderer = Renderer(
            visualizer=visualizer
        )

        result = renderer.compose(
            original_image,
            faces,
            debug=debug,
            session_id=session_id
        )

        # Cache result (only if not debug mode)
        if not debug:
            self._cache[context] = result.copy()

        logger.debug("Render complete")
        return result

    @property
    def cache_size(self) -> int:
        """Return number of cached render results."""
        return len(self._cache)

    def has_cached_result(self, context: ImageContext) -> bool:
        """Check whether a cached render exists for *context*."""
        return context in self._cache

    def invalidate_cache(self, context: ImageContext) -> None:
        """
        Clear cached render for a specific context.

        Called when makeup changes to ensure fresh render on next render() call.

        Args:
            context: ImageContext whose cache should be invalidated.
        """
        if context in self._cache:
            del self._cache[context]
            logger.debug("Render cache invalidated")
