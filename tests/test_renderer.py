"""
Tests for Renderer class.
"""
import pytest
import numpy as np

from face_makeup.face import MakeupLayer, EyeShadowLayer, EyeIrisLayer, EyeLinerLayer
from face_makeup.core.types import (
    EyeShadowParams,
    EyeIrisParams,
    EyeLinerParams,
    LipsParams,
    CheekParams,
    ColorParams,
    PARAMS_PRIORITY
)
from face_makeup.renderer import Renderer
from face_makeup.core.segmenter import FaceSegmenter
from face_makeup.core.makeup_blender import MakeupBlender


class TestResolveOverlaps:
    """Tests for _resolve_overlaps method."""

    @pytest.fixture
    def renderer(self):
        """Create renderer instance."""
        segmenter = FaceSegmenter()
        blender = MakeupBlender()
        return Renderer(segmenter, blender)

    @pytest.fixture
    def sample_mask(self):
        """Create a sample mask for testing."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0  # Center square
        return mask

    def test_same_type_overlap_allowed(self, renderer, sample_mask):
        """Same PartParams type layers should overlap."""
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(200, 150, 150)),
            opacity=0.3
        )

        # Two EyeShadowParams layers (eyelid and eyelid_edge)
        layer1 = EyeShadowLayer(part_name='eyelid_left', params=params)
        layer1.mask = sample_mask.copy()

        layer2 = EyeShadowLayer(part_name='eyelid_edge_left', params=params)
        layer2.mask = sample_mask.copy()  # Same mask area

        layers = [layer1, layer2]
        renderer._resolve_overlaps(layers)

        # Both masks should remain unchanged (overlap allowed)
        assert np.allclose(layer1.mask, sample_mask)
        assert np.allclose(layer2.mask, sample_mask)

    def test_different_type_overlap_resolved(self, renderer, sample_mask):
        """Different PartParams type layers should have overlap resolved."""
        eyeshadow_params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(200, 150, 150)),
            opacity=0.3
        )
        eyeiris_params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(100, 200, 100)),
            opacity=0.5
        )

        # EyeShadowParams vs EyeIrisParams
        layer1 = EyeShadowLayer(part_name='eyelid_left', params=eyeshadow_params)
        layer1.mask = sample_mask.copy()

        layer2 = EyeIrisLayer(part_name='eyeiris_left', params=eyeiris_params)
        layer2.mask = sample_mask.copy()  # Same mask area

        layers = [layer1, layer2]
        renderer._resolve_overlaps(layers)

        # EyeIrisParams has higher priority, so eyeshadow should be subtracted
        # EyeIris mask should remain unchanged
        assert np.allclose(layer2.mask, sample_mask)

        # Eyeshadow mask should have the overlap region removed
        assert not np.allclose(layer1.mask, sample_mask)
        # In the overlap region, eyeshadow should be 0
        assert np.all(layer1.mask[30:70, 30:70] == 0)

    def test_priority_order(self, renderer, sample_mask):
        """Higher priority layers should keep their masks."""
        cheek_params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 180, 180)),
            opacity=0.3,
            center=(0.0, 0.0),
            radius=(0.5, 0.5)
        )
        lips_params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        eyeliner_params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0
        )

        # Create layers with same mask area
        layer_cheek = MakeupLayer(part_name='cheek_left', params=cheek_params)
        layer_cheek.mask = sample_mask.copy()

        layer_lips = MakeupLayer(part_name='lips', params=lips_params)
        layer_lips.mask = sample_mask.copy()

        layer_eyeliner = EyeLinerLayer(part_name='eyeliner_left', params=eyeliner_params)
        layer_eyeliner.mask = sample_mask.copy()

        layers = [layer_cheek, layer_lips, layer_eyeliner]
        renderer._resolve_overlaps(layers)

        # Check priority: EyeLiner > Lips > Cheek
        assert PARAMS_PRIORITY[EyeLinerParams] > PARAMS_PRIORITY[LipsParams]
        assert PARAMS_PRIORITY[LipsParams] > PARAMS_PRIORITY[CheekParams]

        # Eyeliner (highest) should be unchanged
        assert np.allclose(layer_eyeliner.mask, sample_mask)

        # Lips and cheek should have overlap removed from eyeliner
        assert np.all(layer_lips.mask[30:70, 30:70] == 0)
        assert np.all(layer_cheek.mask[30:70, 30:70] == 0)

    def test_empty_layers_list(self, renderer):
        """Empty layers list should not raise error."""
        layers = []
        renderer._resolve_overlaps(layers)  # Should not raise

    def test_layers_without_masks(self, renderer):
        """Layers without masks should be skipped."""
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(200, 150, 150)),
            opacity=0.3
        )

        layer1 = EyeShadowLayer(part_name='eyelid_left', params=params)
        layer1.mask = None  # No mask

        layer2 = EyeShadowLayer(part_name='eyelid_right', params=params)
        layer2.mask = None  # No mask

        layers = [layer1, layer2]
        renderer._resolve_overlaps(layers)  # Should not raise

        # Masks should remain None
        assert layer1.mask is None
        assert layer2.mask is None

    def test_partial_overlap(self, renderer):
        """Partial overlap should only affect overlapping region."""
        eyeshadow_params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(200, 150, 150)),
            opacity=0.3
        )
        eyeiris_params = EyeIrisParams(
            color_params=ColorParams(target_rgb=(100, 200, 100)),
            opacity=0.5
        )

        # Eyeshadow mask covers larger area
        eyeshadow_mask = np.zeros((100, 100), dtype=np.float32)
        eyeshadow_mask[20:80, 20:80] = 1.0

        # Eyeiris mask covers smaller center area
        eyeiris_mask = np.zeros((100, 100), dtype=np.float32)
        eyeiris_mask[40:60, 40:60] = 1.0

        layer1 = EyeShadowLayer(part_name='eyelid_left', params=eyeshadow_params)
        layer1.mask = eyeshadow_mask

        layer2 = EyeIrisLayer(part_name='eyeiris_left', params=eyeiris_params)
        layer2.mask = eyeiris_mask

        layers = [layer1, layer2]
        renderer._resolve_overlaps(layers)

        # Eyeiris (higher priority) should be unchanged
        assert np.all(layer2.mask[40:60, 40:60] == 1.0)

        # Eyeshadow should have center removed but outer ring preserved
        assert np.all(layer1.mask[40:60, 40:60] == 0)  # Overlap removed
        assert np.all(layer1.mask[20:40, 20:80] == 1.0)  # Top edge preserved
        assert np.all(layer1.mask[60:80, 20:80] == 1.0)  # Bottom edge preserved


class TestParamsPriority:
    """Tests for PARAMS_PRIORITY constant."""

    def test_priority_values(self):
        """Check that all PartParams types have expected priorities."""
        assert CheekParams in PARAMS_PRIORITY
        assert LipsParams in PARAMS_PRIORITY
        assert EyeShadowParams in PARAMS_PRIORITY
        assert EyeLinerParams in PARAMS_PRIORITY
        assert EyeIrisParams in PARAMS_PRIORITY

    def test_priority_order(self):
        """Check priority ordering is correct."""
        assert PARAMS_PRIORITY[CheekParams] < PARAMS_PRIORITY[LipsParams]
        assert PARAMS_PRIORITY[LipsParams] < PARAMS_PRIORITY[EyeShadowParams]
        assert PARAMS_PRIORITY[EyeShadowParams] < PARAMS_PRIORITY[EyeLinerParams]
        assert PARAMS_PRIORITY[EyeLinerParams] < PARAMS_PRIORITY[EyeIrisParams]
