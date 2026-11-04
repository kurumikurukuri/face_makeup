"""
Tests for Renderer class (two-phase pipeline).
"""
import pytest
import numpy as np

from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS
from face_makeup.face import Face
from face_makeup.core.pipeline import (
    BlendStrategy,
    MakeupPass,
    MaskStrategy,
)
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
    def generate(self, coords, image_shape, image=None, landmarks=None):
        mask = np.zeros(image_shape, dtype=np.uint8)
        mask[30:70, 30:70] = 255
        return mask


class _RegionMaskStrategy(MaskStrategy):
    """Mask covering a fixed, per-instance rectangular region."""

    def __init__(self, y0, y1, x0, x1):
        self._region = (y0, y1, x0, x1)

    def generate(self, coords, image_shape, image=None, landmarks=None):
        mask = np.zeros(image_shape, dtype=np.uint8)
        y0, y1, x0, x1 = self._region
        mask[y0:y1, x0:x1] = 255
        return mask


class _IdentityBlendStrategy(BlendStrategy):
    """Returns image unchanged (for testing pipeline flow)."""

    def apply(self, image, mask, params, landmarks=None):
        return image.copy()


class _TintBlendStrategy(BlendStrategy):
    """Adds a fixed value to masked region (for verifying blend order)."""

    def __init__(self, increment=10):
        self._increment = increment

    def apply(self, image, mask, params, landmarks=None):
        result = image.copy()
        result[mask > 0] = np.clip(
            result[mask > 0].astype(np.int32) + self._increment, 0, 255
        ).astype(np.uint8)
        return result


def _make_pass(pass_type, priority=0, blend=None):
    return MakeupPass(
        pass_type=pass_type,
        priority=priority,
        params=PartParams(
            color_params=ColorParams(target_rgb=(128, 128, 128)),
            opacity=0.5,
        ),
        mask_strategy=_NoopMaskStrategy(),
        blend_strategy=blend or _IdentityBlendStrategy(),
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
            mask_strategy=_RegionMaskStrategy(10, 20, 10, 20),
            blend_strategy=_TintBlendStrategy(increment=10),
        )
        large = MakeupPass(
            pass_type="layered", priority=1,
            params=PartParams(
                color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5,
            ),
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

    def test_no_mask_strategy_skipped(self, renderer):
        p = MakeupPass(
            pass_type="nomask", priority=0,
            params=PartParams(
                color_params=ColorParams(target_rgb=(128, 128, 128)),
                opacity=0.5,
            ),
        )
        face = _make_face(makeup_passes=[p])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
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


def _make_face_with_coords(coords_array, makeup_passes=None):
    """Create a Face with specific landmark coordinates (normalized 0-1)."""
    face = Face.__new__(Face)
    face._face_id = 0
    face._bbox = None
    face._landmarks = FaceLandmarks(points=coords_array.astype(np.float32))
    face._change_callback = None
    face._makeup_passes = makeup_passes or []
    return face


class _FullCoverageMaskStrategy(MaskStrategy):
    """Returns a mask covering the entire image."""

    def generate(self, coords, image_shape, image=None, landmarks=None):
        return np.full(image_shape, 255, dtype=np.uint8)


class _ProbeBlend(BlendStrategy):
    """Encodes the final mask into channel 0 for inspection."""

    def apply(self, image, mask, params, landmarks=None):
        result = image.copy()
        result[:, :, 0] = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        return result


class TestLandmarkExclusionAndClip:
    """Tests for exclusion_landmarks and clip_landmarks in Phase 1."""

    @pytest.fixture
    def renderer(self):
        return Renderer()

    def test_exclusion_landmarks_subtraction(self, renderer):
        """Landmark exclusion subtracts the polygon region from the mask."""
        # Normalized coords for a 100×100 image:
        # indices 0-3 form a square around the centre
        coords = np.zeros((10, 2), dtype=np.float32)
        coords[0] = [0.35, 0.35]
        coords[1] = [0.65, 0.35]
        coords[2] = [0.65, 0.65]
        coords[3] = [0.35, 0.65]

        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(
                color_params=ColorParams(target_rgb=(0, 0, 0)), opacity=1.0,
            ),
            mask_strategy=_FullCoverageMaskStrategy(),
            blend_strategy=_ProbeBlend(),
            exclusion_landmarks=[[0, 1, 2, 3]],
        )
        face = _make_face_with_coords(coords, makeup_passes=[p])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)

        # Centre (inside exclusion polygon): masked out
        assert result[50, 50, 0] < 10, f"Expected excluded, got {result[50, 50, 0]}"
        # Corner (outside exclusion polygon): full mask
        assert result[10, 10, 0] > 240, f"Expected full, got {result[10, 10, 0]}"

    def test_clip_landmarks_intersection(self, renderer):
        """Clip restricts the mask to the landmark polygon region."""
        coords = np.zeros((10, 2), dtype=np.float32)
        coords[0] = [0.25, 0.25]
        coords[1] = [0.75, 0.25]
        coords[2] = [0.75, 0.75]
        coords[3] = [0.25, 0.75]

        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(
                color_params=ColorParams(target_rgb=(0, 0, 0)), opacity=1.0,
            ),
            mask_strategy=_FullCoverageMaskStrategy(),
            blend_strategy=_ProbeBlend(),
            clip_landmarks=[[0, 1, 2, 3]],
        )
        face = _make_face_with_coords(coords, makeup_passes=[p])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)

        # Inside clip region: mask > 0
        assert result[50, 50, 0] > 10, f"Expected >0 inside clip, got {result[50, 50, 0]}"
        # Outside clip region: mask == 0
        assert result[10, 10, 0] == 0, f"Expected 0 outside clip, got {result[10, 10, 0]}"

    def test_exclusion_landmarks_no_valid_points(self, renderer):
        """Out-of-range landmark indices are silently skipped (no crash)."""
        coords = np.zeros((5, 2), dtype=np.float32)

        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(
                color_params=ColorParams(target_rgb=(0, 0, 0)), opacity=1.0,
            ),
            mask_strategy=_FullCoverageMaskStrategy(),
            blend_strategy=_ProbeBlend(),
            exclusion_landmarks=[[10, 11, 12, 13, 14]],
        )
        face = _make_face_with_coords(coords, makeup_passes=[p])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)

        # No valid exclusion → full coverage everywhere
        assert result[50, 50, 0] > 240


# ── face-wide lash-line correction ──────────────────────────────────────────


class _FixedTSession:
    """ONNX session stand-in returning t=0.5 for every edge."""

    def __init__(self):
        self.runs = 0

    def run(self, output_names, feed):
        self.runs += 1
        return [np.full((len(feed["px"]), 1), 0.5, dtype=np.float32)]


class _CoordsSpyStrategy(MaskStrategy):
    """Records the coords it is given; optionally declares a lash-line dependency."""

    def __init__(self, uses_lash_line=False):
        self.uses_lash_line = uses_lash_line
        self.received = []

    def generate(self, coords, image_shape, image=None, landmarks=None):
        self.received.append(coords)
        return _NoopMaskStrategy().generate(coords, image_shape)


class _SpyLashCorrector:
    """LashLineCorrector stand-in shifting every edge row by a fixed offset."""

    def __init__(self):
        self.calls = 0
        self.corrector = None

    def apply(self, image, landmarks, coords):
        from face_makeup.core.landmark_correction import LashLineCorrection

        self.calls += 1
        corrected = np.asarray(coords, dtype=np.float32).copy()
        corrected[list(EDGE_LANDMARK_DIRECTIONS)] += (0.0, -3.0)
        return LashLineCorrection(coords=corrected, corrector=self.corrector)


def _extended_face(passes):
    from face_makeup.core.landmark_resolver import extend_landmarks

    rng = np.random.RandomState(0)
    points = (0.3 + 0.4 * rng.rand(478, 2)).astype(np.float32)
    face = _make_face(makeup_passes=passes)
    face._landmarks = extend_landmarks(FaceLandmarks(points=points))
    return face


def _pass_with(strategy, pass_type="test", **kwargs):
    return MakeupPass(
        pass_type=pass_type,
        priority=0,
        params=PartParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)), opacity=0.5
        ),
        mask_strategy=strategy,
        blend_strategy=_IdentityBlendStrategy(),
        **kwargs,
    )


class TestLashLineCorrection:
    """The renderer corrects the lash line once per face and shares the result."""

    def test_lash_line_passes_receive_corrected_coords(self):
        strategy = _CoordsSpyStrategy(uses_lash_line=True)
        corrector = _SpyLashCorrector()
        face = _extended_face([_pass_with(strategy)])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        Renderer(lash_corrector=corrector).render_face(image, face)

        raw = face.landmarks.to_image_coords(100, 100)
        (coords,) = strategy.received
        edge_ids = list(EDGE_LANDMARK_DIRECTIONS)
        np.testing.assert_allclose(coords[edge_ids], raw[edge_ids] + (0.0, -3.0))

    def test_every_pass_shares_the_corrected_coords(self):
        """Eyeshadow, foundation, etc. see the same corrected geometry as eyeliner."""
        lash = _CoordsSpyStrategy(uses_lash_line=True)
        other = _CoordsSpyStrategy()
        corrector = _SpyLashCorrector()
        face = _extended_face([_pass_with(lash), _pass_with(other)])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        Renderer(lash_corrector=corrector).render_face(image, face)

        assert corrector.calls == 1
        assert lash.received[0] is other.received[0]

    def test_exclusion_groups_use_corrected_coords(self, monkeypatch):
        """A pass excluding lash-line regions (foundation) triggers the correction."""
        import face_makeup.renderer as renderer_module

        seen = []
        real = renderer_module._build_landmark_group_mask

        def spy(coords, groups, image_shape, **kwargs):
            if groups:
                seen.append(coords)
            return real(coords, groups, image_shape, **kwargs)

        monkeypatch.setattr(renderer_module, "_build_landmark_group_mask", spy)
        corrector = _SpyLashCorrector()
        strategy = _CoordsSpyStrategy()
        face = _extended_face([
            _pass_with(strategy, exclusion_landmarks=[[1401, 1402, 1403, 1404]]),
        ])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        Renderer(lash_corrector=corrector).render_face(image, face)

        assert corrector.calls == 1
        assert seen and seen[0] is strategy.received[0]

    def test_no_correction_without_lash_line_passes(self):
        """A lips-only face never runs the correction model."""
        corrector = _SpyLashCorrector()
        strategy = _CoordsSpyStrategy()
        face = _extended_face([_pass_with(strategy)])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        Renderer(lash_corrector=corrector).render_face(image, face)

        assert corrector.calls == 0
        np.testing.assert_array_equal(
            strategy.received[0], face.landmarks.to_image_coords(100, 100)
        )

    def test_no_corrector_keeps_uncorrected_coords(self):
        """Without a corrector the raw landmarks are drawn."""
        strategy = _CoordsSpyStrategy(uses_lash_line=True)
        face = _extended_face([_pass_with(strategy)])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        Renderer().render_face(image, face)

        np.testing.assert_array_equal(
            strategy.received[0], face.landmarks.to_image_coords(100, 100)
        )

    def test_real_passes_declare_lash_line_use(self):
        from face_makeup.core.passes.eyeliner import EyelinerMaskStrategy
        from face_makeup.core.passes.eyeshadow import EyeShadowMaskStrategy
        from face_makeup.core.passes.tearbag import (
            TearBagHighlightMaskStrategy,
            TearBagShadowMaskStrategy,
        )

        for cls in (
            EyelinerMaskStrategy,
            EyeShadowMaskStrategy,
            TearBagHighlightMaskStrategy,
            TearBagShadowMaskStrategy,
        ):
            assert cls.uses_lash_line is True, cls.__name__
        assert MaskStrategy.uses_lash_line is False


class TestLashLineDebug:
    """debug=True saves each eye's correction once, from the render's cache."""

    def _render(self, mode, visualizer, session=None):
        from face_makeup.core.landmark_correction import LashLineCorrector
        from face_makeup.core.passes.eyeliner import EyelinerMaskStrategy

        corrector = LashLineCorrector(mode)
        if session is not None:
            corrector.resolve()._session = session
        face = _extended_face([
            _pass_with(EyelinerMaskStrategy("eyeliner_left"), pass_type="eyeliner_left"),
        ])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        Renderer(visualizer=visualizer, lash_corrector=corrector).render_face(
            image, face, debug=True, session_id="s1"
        )

    def test_ml_saves_one_overlay_per_eye_without_reinference(self):
        from unittest.mock import MagicMock

        session = _FixedTSession()
        visualizer = MagicMock()
        self._render("ml", visualizer, session)

        assert session.runs == 2  # one batched inference per eye, none for debug
        visualizer.save_eyeliner_correction_debug.assert_not_called()
        sides = [c.kwargs["side"] for c in visualizer.save_eyeliner_ml_debug.call_args_list]
        assert sides == ["left", "right"]
        for call in visualizer.save_eyeliner_ml_debug.call_args_list:
            assert call.kwargs["entries"]
            assert all("t" in e for e in call.kwargs["entries"])

    def test_rule_keeps_per_point_debug(self):
        from unittest.mock import MagicMock

        visualizer = MagicMock()
        self._render("rule", visualizer)

        visualizer.save_eyeliner_ml_debug.assert_not_called()

    def test_no_debug_without_correction(self):
        from unittest.mock import MagicMock

        visualizer = MagicMock()
        face = _extended_face([_pass_with(_CoordsSpyStrategy(uses_lash_line=True))])
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        Renderer(visualizer=visualizer).render_face(image, face, debug=True)

        visualizer.save_eyeliner_ml_debug.assert_not_called()
        visualizer.save_eyeliner_correction_debug.assert_not_called()
