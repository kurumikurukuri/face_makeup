"""
Tests for HairSegmenter class and hair mask exclusion in Renderer.
"""
import numpy as np
import pytest

from face_makeup.core.hair_segmenter import HairSegmenter
from face_makeup.core.pipeline import MakeupPass, MaskStrategy, BlendStrategy
from face_makeup.core.types import PartParams, ColorParams
from face_makeup.renderer import Renderer


class _FullMaskStrategy(MaskStrategy):
    """Mask that covers the entire image (255 everywhere)."""

    def generate(self, coords, image_shape, image=None, landmarks=None, segmenter=None):
        return np.full(image_shape, 255, dtype=np.uint8)


class _IdentityBlend(BlendStrategy):
    """Adds +10 to R channel in masked region so hair exclusion is observable."""

    def apply(self, image, mask, params, blender=None, segmenter=None, landmarks=None):
        result = image.copy().astype(np.int32)
        result[mask > 0, 2] += 10
        return np.clip(result, 0, 255).astype(np.uint8)


def _make_pass(pass_type, priority=0, landmarks=None):
    return MakeupPass(
        pass_type=pass_type,
        priority=priority,
        params=PartParams(
            color_params=ColorParams(target_rgb=(200, 100, 100)),
            opacity=0.5,
        ),
        landmark_indices=landmarks or [],
        mask_strategy=_FullMaskStrategy(),
        blend_strategy=_IdentityBlend(),
    )


def _make_face(makeup_passes):
    from face_makeup.face import Face
    from face_makeup.core.types import FaceBBox, FaceLandmarks

    face = Face(0, FaceBBox(0, 0, 100, 100, 1.0), FaceLandmarks(np.zeros((478, 2), dtype=np.float32)))
    for p in makeup_passes:
        face._makeup_passes.append(p)
    return face


class TestHairSegmenter:
    """Tests for HairSegmenter class."""

    @pytest.fixture
    def model_path(self):
        from pathlib import Path
        path = Path(__file__).parent.parent / "models" / "hair_segmenter.tflite"
        if not path.exists():
            pytest.skip("hair_segmenter.tflite not found")
        return str(path)

    def test_create_and_close(self, model_path):
        seg = HairSegmenter(model_path)
        assert seg._net is not None
        seg.close()
        assert seg._net is None

    def test_context_manager(self, model_path):
        with HairSegmenter(model_path) as seg:
            assert seg._net is not None
        assert seg._net is None

    def test_segment_returns_uint8_mask(self, model_path):
        seg = HairSegmenter(model_path)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        mask = seg.segment(img)
        seg.close()
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (128, 128)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_segment_non_square_image(self, model_path):
        seg = HairSegmenter(model_path)
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        mask = seg.segment(img)
        seg.close()
        assert mask is not None
        assert mask.shape == (100, 200)

    def test_segment_preserves_input_image(self, model_path):
        seg = HairSegmenter(model_path)
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        original = img.copy()
        seg.segment(img)
        seg.close()
        assert np.array_equal(img, original)


class _FakeHairSegmenter:
    """Fake segmenter that returns a fixed mask — avoids loading the real model."""

    def __init__(self, mask):
        self._mask = mask

    def segment(self, image):
        return self._mask.copy()


class TestHairExclusionInRenderer:
    """Tests that hair mask is subtracted from all pass masks."""

    def test_hair_mask_subtracted_from_all_passes(self):
        """Hair region is excluded from every pass (feathered, applied after fade).

        Exclusion is now a feathered soft gate: the interior of a hair region
        stays fully excluded while its boundary ramps. We assert on a deep
        interior pixel (fully excluded) and pixels well outside the region.
        """
        hair_mask = np.zeros((64, 64), dtype=np.uint8)
        hair_mask[20:44, 20:44] = 255  # large hair patch → solid interior

        fake_seg = _FakeHairSegmenter(hair_mask)
        renderer = Renderer(hair_segmenter=fake_seg)

        pass_a = _make_pass("pass_a", priority=0)
        pass_b = _make_pass("pass_b", priority=1)
        face = _make_face([pass_a, pass_b])

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)

        # Two passes, each adds +10 to R. Deep hair interior excluded → +0.
        assert result[5, 5, 2] == 20      # well outside hair: +10+10 = 20
        assert result[32, 32, 2] == 0     # deep hair interior, fully excluded
        assert result[58, 58, 2] == 20    # well outside hair: +10+10 = 20

    def test_hair_mask_none_does_not_affect(self):
        """Without hair segmenter, behavior is unchanged."""
        renderer = Renderer(hair_segmenter=None)
        pass_a = _make_pass("pass_a", priority=0)
        face = _make_face([pass_a])
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        # Full mask, no exclusions → +10 everywhere
        assert result[30, 30, 2] == 10

    def test_hair_segmenter_returns_none(self):
        """Graceful degradation when segment() returns None."""

        class _BrokenSegmenter:
            def segment(self, image):
                return None

        renderer = Renderer(hair_segmenter=_BrokenSegmenter())
        pass_a = _make_pass("pass_a", priority=0)
        face = _make_face([pass_a])
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result[30, 30, 2] == 10  # behaves as if no hair segmenter
