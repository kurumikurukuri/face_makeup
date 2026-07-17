"""
Tests for RenderingPipeline class.
"""
import pytest
import numpy as np
import gc

from face_makeup.rendering_pipeline import RenderingPipeline
from face_makeup.face import Face
from face_makeup.core.types import FaceBBox, FaceLandmarks
from face_makeup.core.landmark_resolver import extend_landmarks


@pytest.fixture
def pipeline():
    """RenderingPipeline instance."""
    return RenderingPipeline()


@pytest.fixture
def test_bbox():
    """Test bounding box."""
    return FaceBBox(x=100, y=100, w=200, h=200, score=0.95)


@pytest.fixture
def test_landmarks():
    """Test landmarks (478 base points, extended with custom)."""
    return extend_landmarks(
        FaceLandmarks(points=np.random.rand(478, 2).astype(np.float32))
    )


@pytest.fixture
def test_face(test_bbox, test_landmarks):
    """Test Face object."""
    return Face(
        face_id=0,
        bbox=test_bbox,
        landmarks=test_landmarks,
        change_callback=None
    )


class SimpleContext:
    """Simple context object for testing (can be used as WeakKeyDictionary key)."""
    pass


class TestRenderingPipelineInitialization:
    """Tests for RenderingPipeline initialization."""

    def test_pipeline_initializes(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.cache_size == 0

    def test_empty_cache_on_init(self, pipeline):
        """Test that pipeline starts with empty cache."""
        assert pipeline.cache_size == 0


class TestRenderingPipelineCacheManagement:
    """Tests for cache management in RenderingPipeline."""

    def test_cache_returns_none_for_missing_context(self, pipeline, test_face):
        """Test that cache has no entry for a fresh context."""
        ctx = SimpleContext()

        assert not pipeline.has_cached_result(ctx)

    def test_cache_stores_result_after_render(self, pipeline, test_face, test_image):
        """Test that cache stores result after render."""
        ctx = SimpleContext()

        pipeline.render(ctx, test_image, [test_face])

        assert pipeline.has_cached_result(ctx)

    def test_cache_returns_copy_on_hit(self, pipeline, test_face, test_image):
        """Test that cache returns a copy, not the original."""
        ctx = SimpleContext()

        result1 = pipeline.render(ctx, test_image, [test_face])
        result2 = pipeline.render(ctx, test_image, [test_face])

        assert np.array_equal(result1, result2)
        assert result1 is not result2

    def test_cache_invalidation(self, pipeline, test_face, test_image):
        """Test cache invalidation."""
        ctx = SimpleContext()

        pipeline.render(ctx, test_image, [test_face])
        assert pipeline.has_cached_result(ctx)

        pipeline.invalidate_cache(ctx)

        assert not pipeline.has_cached_result(ctx)

    def test_automatic_cleanup_on_context_deletion(self, pipeline, test_face, test_image):
        """Test that WeakKeyDictionary automatically cleans up when context is deleted."""
        ctx = SimpleContext()

        pipeline.render(ctx, test_image, [test_face])
        assert pipeline.cache_size == 1

        del ctx
        gc.collect()

        assert pipeline.cache_size == 0

    def test_cache_per_context(self, pipeline, test_face, test_image):
        """Test that cache is per-context (multiple contexts have independent caches)."""
        face2 = Face(
            face_id=1,
            bbox=test_face.bbox,
            landmarks=test_face.landmarks,
            change_callback=None
        )

        ctx1 = SimpleContext()
        ctx2 = SimpleContext()

        pipeline.render(ctx1, test_image, [test_face])
        pipeline.render(ctx2, test_image, [face2])

        assert pipeline.has_cached_result(ctx1)
        assert pipeline.has_cached_result(ctx2)
        assert pipeline.cache_size == 2


class TestRenderingPipelineRender:
    """Tests for RenderingPipeline.render() method."""

    def test_render_returns_ndarray(self, pipeline, test_face, test_image):
        """Test that render returns numpy array."""
        ctx = SimpleContext()

        result = pipeline.render(ctx, test_image, [test_face])

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == test_image.shape

    def test_render_without_makeup(self, pipeline, test_face, test_image):
        """Test render without makeup returns original image."""
        ctx = SimpleContext()

        # No makeup applied
        result = pipeline.render(ctx, test_image, [test_face])

        # Should match original
        assert np.array_equal(result, test_image)

    def test_render_with_makeup(self, pipeline, test_face, test_image):
        """Test render with makeup applied."""
        ctx = SimpleContext()

        # Apply makeup
        test_face.lip(color=(255, 100, 100), opacity=0.5)

        result = pipeline.render(ctx, test_image, [test_face])

        # Result should be different from original
        assert not np.array_equal(result, test_image)
        assert result.shape == test_image.shape

    def test_render_multiple_faces(self, pipeline, test_image):
        """Test render with multiple faces."""
        from face_makeup.core.types import FaceLandmarks
        face1 = Face(
            face_id=0,
            bbox=FaceBBox(x=100, y=100, w=200, h=200, score=0.95),
            landmarks=extend_landmarks(FaceLandmarks(points=np.random.rand(478, 2).astype(np.float32))),
            change_callback=None
        )
        face2 = Face(
            face_id=1,
            bbox=FaceBBox(x=350, y=100, w=200, h=200, score=0.95),
            landmarks=extend_landmarks(FaceLandmarks(points=np.random.rand(478, 2).astype(np.float32))),
            change_callback=None
        )

        face1.lip(color=(255, 0, 0), opacity=0.5)
        face2.lip(color=(0, 255, 0), opacity=0.5)

        ctx = SimpleContext()
        result = pipeline.render(ctx, test_image, [face1, face2])

        assert result.shape == test_image.shape
        assert not np.array_equal(result, test_image)

    def test_render_debug_mode(self, pipeline, test_face, test_image):
        """Test render with debug=True."""
        ctx = SimpleContext()
        test_face.lip(color=(255, 100, 100), opacity=0.5)

        # Debug mode should not use cache
        result_debug = pipeline.render(ctx, test_image, [test_face], debug=True)

        assert isinstance(result_debug, np.ndarray)
        assert result_debug.shape == test_image.shape


class TestRenderingPipelineSharedPipeline:
    """Tests for RenderingPipeline sharing across contexts."""

    def test_pipeline_shared_across_contexts(self, test_image):
        """Test that multiple contexts can share the same pipeline."""
        from face_makeup.core.types import FaceLandmarks
        pipeline = RenderingPipeline()

        face1 = Face(
            face_id=0,
            bbox=FaceBBox(x=100, y=100, w=200, h=200, score=0.95),
            landmarks=extend_landmarks(FaceLandmarks(points=np.random.rand(478, 2).astype(np.float32))),
            change_callback=None
        )
        face2 = Face(
            face_id=0,
            bbox=FaceBBox(x=100, y=100, w=200, h=200, score=0.95),
            landmarks=extend_landmarks(FaceLandmarks(points=np.random.rand(478, 2).astype(np.float32))),
            change_callback=None
        )

        ctx1 = SimpleContext()
        ctx2 = SimpleContext()

        # Render both contexts with same pipeline
        pipeline.render(ctx1, test_image, [face1])
        pipeline.render(ctx2, test_image, [face2])

        # Both should be in cache with independent entries
        assert pipeline.has_cached_result(ctx1)
        assert pipeline.has_cached_result(ctx2)
        assert pipeline.cache_size == 2
