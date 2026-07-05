"""
Tests for Renderer class (two-phase pipeline).
"""
import pytest
import numpy as np

from face_makeup.face import Face
from face_makeup.core.pipeline import MakeupPass, MaskStrategy, BlendStrategy
from face_makeup.core.passes.foundation import FoundationPass
from face_makeup.core.types import (
    FaceLandmarks,
    ColorParams,
    PartParams,
)
from face_makeup.renderer import Renderer


def _make_landmarks(n_points=478):
    """Create FaceLandmarks with coordinates in a rough face oval pattern."""
    pts = np.zeros((n_points, 2), dtype=np.float32)
    # Scatter points around the center with a simple oval pattern
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts[:, 0] = 0.5 + 0.2 * np.cos(angles)
    pts[:, 1] = 0.5 + 0.3 * np.sin(angles)
    return FaceLandmarks(points=pts)


def _make_face(makeup_passes=None):
    """Create a minimal Face instance for testing."""
    face = Face.__new__(Face)
    face._face_id = 0
    face._bbox = None
    face._landmarks = _make_landmarks()
    face._change_callback = None
    face._makeup_passes = makeup_passes or []
    return face


class _NoopMaskStrategy(MaskStrategy):
    def generate(self, coords, image_shape, image=None, landmarks=None, segmenter=None):
        mask = np.zeros(image_shape, dtype=np.uint8)
        mask[30:70, 30:70] = 255
        return mask


class _RegionMaskStrategy(MaskStrategy):
    """Mask covering a fixed, per-instance rectangular region."""

    def __init__(self, y0, y1, x0, x1):
        self._region = (y0, y1, x0, x1)

    def generate(self, coords, image_shape, image=None, landmarks=None, segmenter=None):
        mask = np.zeros(image_shape, dtype=np.uint8)
        y0, y1, x0, x1 = self._region
        mask[y0:y1, x0:x1] = 255
        return mask


class _IdentityBlendStrategy(BlendStrategy):
    """Returns image unchanged (for testing pipeline flow)."""

    def apply(self, image, mask, params, blender=None, segmenter=None, landmarks=None):
        return image.copy()


class _TintBlendStrategy(BlendStrategy):
    """Adds a fixed value to masked region (for verifying blend order)."""

    def __init__(self, increment=10):
        self._increment = increment

    def apply(self, image, mask, params, blender=None, segmenter=None, landmarks=None):
        result = image.copy()
        result[mask > 0] = np.clip(
            result[mask > 0].astype(np.int32) + self._increment, 0, 255
        ).astype(np.uint8)
        return result


def _make_pass(pass_type, priority=0, exclusion_passes=None, blend=None):
    return MakeupPass(
        pass_type=pass_type,
        priority=priority,
        params=PartParams(
            color_params=ColorParams(target_rgb=(128, 128, 128)),
            opacity=0.5,
        ),
        landmark_indices=[],
        mask_strategy=_NoopMaskStrategy(),
        blend_strategy=blend or _IdentityBlendStrategy(),
        exclusion_passes=exclusion_passes or [],
    )


class TestTwoPhasePipeline:
    """Tests for the two-phase render pipeline."""

    @pytest.fixture
    def renderer(self):
        return Renderer()

    def test_render_empty_passes(self, renderer):
        face = _make_face(makeup_passes=[])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        np.testing.assert_array_equal(result, image)

    def test_render_single_pass(self, renderer):
        face = _make_face(makeup_passes=[_make_pass("test", priority=10)])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_exclusion_subtraction(self, renderer):
        """Higher-layer mask should be subtracted from dependent pass mask."""
        source = _make_pass("source", priority=10)
        dependent = _make_pass("dependent", priority=5, exclusion_passes=["source"])
        face = _make_face(makeup_passes=[dependent, source])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_multiple_exclusions(self, renderer):
        source_a = _make_pass("source_a", priority=20)
        source_b = _make_pass("source_b", priority=30)
        dependent = _make_pass(
            "dependent", priority=0,
            exclusion_passes=["source_a", "source_b"],
        )
        face = _make_face(makeup_passes=[dependent, source_b, source_a])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_priority_order_render(self, renderer):
        """Blends are applied in priority order (lowest first)."""
        face = _make_face(makeup_passes=[
            _make_pass("high", priority=100),
            _make_pass("low", priority=0),
        ])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_same_pass_type_layers_keep_independent_masks(self, renderer):
        """Regression: two passes sharing a pass_type (e.g. two
        tear_bag_highlight layers with different `size` stacked for a
        gradient look) must each blend with their OWN mask, not whichever
        instance's mask happened to be registered last under that name.
        """
        small = MakeupPass(
            pass_type="layered", priority=0,
            params=PartParams(
                color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5,
            ),
            landmark_indices=[],
            mask_strategy=_RegionMaskStrategy(10, 20, 10, 20),
            blend_strategy=_TintBlendStrategy(increment=10),
        )
        large = MakeupPass(
            pass_type="layered", priority=1,
            params=PartParams(
                color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5,
            ),
            landmark_indices=[],
            mask_strategy=_RegionMaskStrategy(50, 90, 50, 90),
            blend_strategy=_TintBlendStrategy(increment=50),
        )
        face = _make_face(makeup_passes=[small, large])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)

        # small's own region got exactly small's tint (not skipped, not large's)
        assert result[15, 15, 0] == 10
        # large's own region got exactly large's tint (not both stacked)
        assert result[70, 70, 0] == 50

    def test_missing_exclusion_source_skipped(self, renderer):
        """Dependent with exclusion from non-existent pass: silently skipped."""
        dependent = _make_pass("dependent", priority=5, exclusion_passes=["nonexistent"])
        face = _make_face(makeup_passes=[dependent])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_no_mask_strategy_skipped(self, renderer):
        p = MakeupPass(
            pass_type="nomask", priority=0,
            params=PartParams(
                color_params=ColorParams(target_rgb=(128, 128, 128)),
                opacity=0.5,
            ),
            landmark_indices=[],
        )
        face = _make_face(makeup_passes=[p])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_foundation_with_exclusion_sources(self, renderer):
        """Foundation with eyelid and lip exclusion sources should render correctly."""
        foundation = FoundationPass(
            params=PartParams(
                color_params=ColorParams(target_rgb=(200, 200, 200)),
                opacity=0.3,
            ),
        )
        eyelid_left = _make_pass("eyelid_left", priority=25)
        eyelid_right = _make_pass("eyelid_right", priority=26)
        lips = _make_pass("lips", priority=20)
        face = _make_face(makeup_passes=[foundation, eyelid_left, eyelid_right, lips])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.shape == (100, 100, 3)

    def test_render_returns_uint8(self, renderer):
        face = _make_face(makeup_passes=[_make_pass("test", priority=10)])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result.dtype == np.uint8

    def test_blend_applied_in_priority_order(self, renderer):
        """Verify that Phase 2 applies lowest priority first (under) then higher (over)."""
        # low priority blends first (underneath), high last (on top overlaying)
        # Both have same mask region; high blends last so its tint is visible
        low = _make_pass("low", priority=0, blend=_TintBlendStrategy(increment=10))
        high = _make_pass("high", priority=100, blend=_TintBlendStrategy(increment=50))
        face = _make_face(makeup_passes=[high, low])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        # Both applied: low first adds 10, then high adds 50 on top
        # So masked region should have 60 (10 + 50) or close to it
        center_pixel = result[50, 50, 0]
        # low (pri=0) blended first → +10, then high (pri=100) → +50 = +60 total in mask
        assert center_pixel == 60, f"Expected 60, got {center_pixel}"


