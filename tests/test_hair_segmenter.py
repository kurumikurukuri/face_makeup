"""
Tests for HairSegmenter class and hair mask exclusion in Renderer.
"""
import numpy as np
import pytest

from face_makeup.core.hair_segmenter import HairSegmenter
from face_makeup.core.pipeline import MakeupPass, MaskStrategy, BlendStrategy
from face_makeup.core.types import PartParams, ColorParams, FaceBBox
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

    def test_prior_mask_channel_is_zero(self):
        """The model's 4th input channel is the previous-frame hair mask.

        Feeding a saturated prior (255 → 1.0 after blob scaling) tells the
        model "everything was hair last frame", which drags the whole person
        toward hair and produces striped garbage masks. For still images the
        prior must be all zeros. No model file needed — a spy net captures
        the blob that segment() builds.
        """
        spy = _CountingSpyNet()

        seg = _make_segmenter(spy)

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        seg.segment(img)  # face_bbox=None -> whole-frame -> 1 forward

        blob = spy.blobs[0]
        assert blob.shape[1] == 4  # RGB + prior-mask channel
        assert np.all(blob[0, 3] == 0.0)

    def test_segment_portrait_hair_coverage_is_plausible(self, model_path):
        """Regression guard: on a real portrait the hair mask must stay a
        hair-sized region, not swallow the whole person (the alpha=255 bug
        marked ~half the image as hair, punching stripes into every pass).
        """
        import cv2
        from pathlib import Path

        img_path = (
            Path(__file__).parent
            / "test_images"
            / "jimmy-fermin-bqe0J0b26RQ-unsplash.jpg"
        )
        if not img_path.exists():
            pytest.skip("portrait test image not found")

        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        seg = HairSegmenter(model_path)
        mask = seg.segment(image_rgb)
        seg.close()

        assert mask is not None
        hair_fraction = float(np.count_nonzero(mask)) / mask.size
        assert 0.01 < hair_fraction < 0.35, (
            f"hair fraction {hair_fraction:.2f} outside plausible range"
        )

    def test_tiled_path_fires_four_forwards(self):
        """face_bbox triggers 4-quadrant tiled inference (4 model forwards)."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        seg.segment(img, face_bbox=FaceBBox(50, 50, 100, 100, 1.0))

        assert spy.forward_count == 4

    def test_tiled_prior_mask_channel_is_zero_on_every_tile(self):
        """Each of the 4 tile blobs carries a zero 4th (prior-mask) channel."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        seg.segment(img, face_bbox=FaceBBox(50, 50, 100, 100, 1.0))

        assert len(spy.blobs) == 4
        for blob in spy.blobs:
            assert blob.shape[1] == 4          # RGB + prior-mask channel
            assert np.all(blob[0, 3] == 0.0)   # prior must be zeros for stills

    def test_segment_without_bbox_still_works(self):
        """face_bbox=None keeps the legacy whole-frame path (1 forward)."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = seg.segment(img)

        assert spy.forward_count == 1
        assert mask is not None
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_segment_with_degenerate_bbox_falls_back(self):
        """A region below the tiling floor falls back to whole-frame (1 forward)."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask = seg.segment(img, face_bbox=FaceBBox(0, 0, 2, 2, 1.0))

        assert spy.forward_count == 1   # whole-frame fallback, no crash on tiny tiles
        assert mask is not None
        assert mask.shape == (64, 64)

    def test_tiled_inference_on_real_portrait(self, model_path):
        """End-to-end: real FaceDetector bbox -> tiled mask covers wispy hair."""
        import cv2
        from pathlib import Path
        from face_makeup.core.face_detector import FaceDetector

        img_path = (
            Path(__file__).parent
            / "test_images"
            / "jimmy-fermin-bqe0J0b26RQ-unsplash.jpg"
        )
        if not img_path.exists():
            pytest.skip("portrait test image not found")

        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        try:
            bboxes = FaceDetector().detect(image_rgb)
        except ValueError:
            # detect() raises on no face (its contract) instead of returning [].
            pytest.skip("no face detected in portrait")

        seg = HairSegmenter(model_path)
        mask = seg.segment(image_rgb, face_bbox=bboxes[0])
        seg.close()

        assert mask is not None
        assert mask.shape[:2] == image_rgb.shape[:2]
        hair_fraction = float(np.count_nonzero(mask)) / mask.size
        # Tiling captures more wispy hair than whole-frame, so the upper bound
        # is looser than the whole-frame regression guard.
        assert 0.01 < hair_fraction < 0.45, (
            f"hair fraction {hair_fraction:.2f} outside plausible range"
        )

    def test_segment_caches_identical_image_and_bbox(self):
        """A repeat call with the same image + bbox is served from cache:
        forward count does not increase on the second call."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        bbox = FaceBBox(50, 50, 100, 100, 1.0)
        seg.segment(img, face_bbox=bbox)
        first = spy.forward_count
        seg.segment(img, face_bbox=bbox)

        assert spy.forward_count == first          # no extra forwards
        assert len(seg._mask_cache) == 1

    def test_segment_cache_miss_on_different_bbox(self):
        """A different face_bbox is a different cache entry (re-infers)."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        seg.segment(img, face_bbox=FaceBBox(50, 50, 100, 100, 1.0))
        seg.segment(img, face_bbox=FaceBBox(10, 10, 80, 80, 1.0))

        assert spy.forward_count == 8              # 4 forwards per distinct bbox
        assert len(seg._mask_cache) == 2

    def test_segment_cache_miss_on_different_image(self):
        """A different source array (different identity) re-infers even with
        the same bbox."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img_a = np.full((200, 200, 3), 128, dtype=np.uint8)
        img_b = np.full((200, 200, 3), 200, dtype=np.uint8)
        bbox = FaceBBox(50, 50, 100, 100, 1.0)
        seg.segment(img_a, face_bbox=bbox)
        seg.segment(img_b, face_bbox=bbox)

        assert spy.forward_count == 8
        assert len(seg._mask_cache) == 2

    def test_close_clears_mask_cache(self):
        """close() drops all cached masks (and the net)."""
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        seg.segment(img, face_bbox=FaceBBox(50, 50, 100, 100, 1.0))
        assert len(seg._mask_cache) == 1

        seg.close()
        assert seg._mask_cache == {}

    def test_mask_cache_evicts_oldest_when_full(self, monkeypatch):
        """At capacity, the oldest entry is dropped (FIFO) to make room."""
        monkeypatch.setattr(HairSegmenter, "_MASK_CACHE_MAX", 2)
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        bbox = FaceBBox(50, 50, 100, 100, 1.0)
        img1 = np.full((200, 200, 3), 1, dtype=np.uint8)
        img2 = np.full((200, 200, 3), 2, dtype=np.uint8)
        seg.segment(img1, face_bbox=bbox)
        seg.segment(img2, face_bbox=bbox)
        assert len(seg._mask_cache) == 2

        img3 = np.full((200, 200, 3), 3, dtype=np.uint8)
        seg.segment(img3, face_bbox=bbox)
        assert len(seg._mask_cache) == 2           # still capped
        # img1 (oldest) was evicted; img2 is still cached. Re-segmenting img2
        # must not infer, but img1 (evicted) must re-infer (4 forwards).
        before = spy.forward_count
        seg.segment(img2, face_bbox=bbox)
        assert spy.forward_count == before         # img2 still cached
        seg.segment(img1, face_bbox=bbox)
        assert spy.forward_count == before + 4     # img1 evicted -> re-inferred

    def test_mask_cache_evicts_dead_weakref_first(self, monkeypatch):
        """A dead weakref (source image GC'd) is evicted before the FIFO entry."""
        import gc
        monkeypatch.setattr(HairSegmenter, "_MASK_CACHE_MAX", 2)
        spy = _CountingSpyNet()
        seg = _make_segmenter(spy)

        bbox = FaceBBox(50, 50, 100, 100, 1.0)
        img_gone = np.full((200, 200, 3), 1, dtype=np.uint8)
        seg.segment(img_gone, face_bbox=bbox)
        del img_gone
        gc.collect()

        img_keep = np.full((200, 200, 3), 2, dtype=np.uint8)
        seg.segment(img_keep, face_bbox=bbox)
        # Cache now holds one dead entry (img_gone) and one live (img_keep).
        img_new = np.full((200, 200, 3), 3, dtype=np.uint8)
        seg.segment(img_new, face_bbox=bbox)       # at capacity -> evict dead first
        assert len(seg._mask_cache) == 2
        before = spy.forward_count
        seg.segment(img_keep, face_bbox=bbox)
        assert spy.forward_count == before         # img_keep survived

    def test_quadrant_blend_normalises_per_tile_values(self):
        """Non-uniform per-tile probabilities blend to a normalised map.

        With TL=0.1, TR=0.2, BL=0.3, BR=0.4 (forward order), each tile's
        exclusive region keeps its own value, the 4-quadrant crossing is the
        mean of all four, and the blend never leaves [min, max] of the inputs
        — i.e. the per-pixel weighted sum is properly normalised (no seam
        spikes from double counting, no holes from under-coverage).
        """
        spy = _GradientSpyNet()
        seg = _make_segmenter(spy)

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        # Face bbox (50,50,100,100) + 50px top headroom -> region (50,0,150,150).
        region = (50, 0, 150, 150)
        prob = seg._infer_region_quadrants(img, region)

        assert prob[5, 55] == pytest.approx(0.1, abs=1e-4)      # TL exclusive
        assert prob[145, 145] == pytest.approx(0.4, abs=1e-4)   # BR exclusive
        assert prob[75, 100] == pytest.approx(0.25, abs=1e-4)   # 4-tile mean

        # Inside the region the blend stays within [min, max] of the inputs.
        roi = prob[0:150, 50:150]
        assert roi.min() >= 0.1 - 1e-4
        assert roi.max() <= 0.4 + 1e-4


class _CountingSpyNet:
    """Spy net: counts forwards and records every input blob (no model file)."""

    def __init__(self) -> None:
        self.forward_count = 0
        self.blobs: list[np.ndarray] = []

    def setInput(self, blob: np.ndarray) -> None:
        self.blobs.append(blob)

    def forward(self) -> np.ndarray:
        self.forward_count += 1
        return np.zeros((1, 2, 512, 512), dtype=np.float32)


class _GradientSpyNet:
    """Spy net returning a distinct constant hair probability per forward call.

    Each call's hair channel is the logit of ``0.1 * call_number`` (TL=0.1,
    TR=0.2, BL=0.3, BR=0.4 in quadrant order), so quadrant-blend normalisation
    can be checked against known per-tile values. No model file required.
    """

    def __init__(self) -> None:
        self.call = 0

    def setInput(self, blob: np.ndarray) -> None:
        pass

    def forward(self) -> np.ndarray:
        self.call += 1
        p = 0.1 * self.call
        logit = float(np.log(p / (1.0 - p)))
        out = np.zeros((1, 2, 512, 512), dtype=np.float32)
        out[0, 1] = logit  # hair channel -> sigmoid(logit) == p
        return out


def _make_segmenter(net: object) -> HairSegmenter:
    """Build a HairSegmenter without loading the model file.

    ``HairSegmenter.__new__`` skips ``__init__``, so the mask-cache dict must
    be set up explicitly — tests that exercise ``segment`` rely on it.
    """
    seg = HairSegmenter.__new__(HairSegmenter)
    seg._net = net
    seg._mask_cache = {}
    return seg


class _FakeHairSegmenter:
    """Fake segmenter that returns a fixed mask — avoids loading the real model."""

    def __init__(self, mask):
        self._mask = mask

    def segment(self, image, face_bbox=None):
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
            def segment(self, image, face_bbox=None):
                return None

        renderer = Renderer(hair_segmenter=_BrokenSegmenter())
        pass_a = _make_pass("pass_a", priority=0)
        face = _make_face([pass_a])
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = renderer.render_face(image, face)
        assert result[30, 30, 2] == 10  # behaves as if no hair segmenter
