"""
Tests for FoundationPass, FoundationMaskStrategy, and FoundationBlendStrategy.
"""
import numpy as np
import pytest

from face_makeup.core.passes.foundation import (
    FoundationMaskStrategy,
    FoundationBlendStrategy,
    FoundationPass,
    _EXCLUSION_REGIONS,
)
from face_makeup.core.types import PartParams, ColorParams


class TestFoundationMaskStrategy:
    """Tests for FoundationMaskStrategy."""

    @pytest.fixture
    def strategy(self):
        return FoundationMaskStrategy(softness=1.0)

    @pytest.fixture
    def oval_coords(self):
        """Simulate a face oval with coordinates forming a rough circle (36 points)."""
        _h, _w = 100, 100
        cx, cy = 50, 50
        radius = 30
        n_points = 36
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        xs = cx + radius * np.cos(angles)
        ys = cy + radius * np.sin(angles)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    @pytest.fixture
    def test_indices(self):
        """Indices into oval_coords that FoundationMaskStrategy should use."""
        return list(range(36))

    def test_output_shape(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        shape = (100, 100)
        mask = strategy.generate(oval_coords, shape)
        assert mask.shape == shape

    def test_output_dtype_float32(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask.dtype == np.float32

    def test_mask_has_values_between_0_and_1(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_center_is_filled(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask[50, 50] > 0.0

    def test_corners_are_zero(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask[0, 0] == 0
        assert mask[0, 99] == 0
        assert mask[99, 0] == 0
        assert mask[99, 99] == 0

    def test_soft_edges_produce_gradient(self, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        # High softness → wide feather band visible even on small synthetic masks.
        s = FoundationMaskStrategy(softness=1.0)
        mask = s.generate(oval_coords, (100, 100))
        unique = np.unique(np.round(mask, 2))
        assert len(unique) > 2  # More than just 0.0 and (near) 1.0

    def test_lower_softness_produces_sharper_edge(self, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        sharp = FoundationMaskStrategy(softness=0.0)
        soft = FoundationMaskStrategy(softness=1.0)
        mask_sharp = sharp.generate(oval_coords, (100, 100))
        mask_soft = soft.generate(oval_coords, (100, 100))
        sharp_ones_count = np.sum(mask_sharp > 0.99)
        soft_ones_count = np.sum(mask_soft > 0.99)
        assert soft_ones_count <= sharp_ones_count

    def test_generated_mask_has_full_oval_without_exclusion_holes(self, monkeypatch):
        """Mask strategy generates a full face oval — exclusion holes
        are NOT carved here; the renderer handles them later."""
        h, w = 200, 200
        n_oval = 36
        n_total = 60
        coords = np.zeros((n_total, 2), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, n_oval, endpoint=False)
        coords[:n_oval, 0] = 100 + 80 * np.cos(angles)
        coords[:n_oval, 1] = 100 + 80 * np.sin(angles)

        # Put exclusion-region points in the centre of the oval
        coords[36] = [85, 85]
        coords[37] = [115, 85]
        coords[38] = [115, 115]
        coords[39] = [85, 115]

        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES",
            list(range(n_oval)),
        )

        strategy = FoundationMaskStrategy(softness=0.0)
        mask = strategy.generate(coords, (h, w))

        # The centre (where exclusion points are) should be filled,
        # because the strategy no longer subtracts exclusion regions.
        assert mask[100, 100] > 0.0, (
            "Mask strategy should NOT carve exclusions; "
            f"got {mask[100, 100]} at centre"
        )


class TestFoundationBlendStrategy:
    """Tests for FoundationBlendStrategy."""

    @pytest.fixture
    def strategy(self):
        return FoundationBlendStrategy()

    @pytest.fixture
    def image(self):
        return np.full((50, 50, 3), 128, dtype=np.uint8)

    @pytest.fixture
    def full_mask(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[:, :] = 1.0
        return mask

    @pytest.fixture
    def params(self):
        return PartParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=0.5,
        )

    def test_output_shape(self, strategy, image, full_mask, params):
        result = strategy.apply(image, full_mask, params)
        assert result.shape == image.shape

    def test_output_dtype_uint8(self, strategy, image, full_mask, params):
        result = strategy.apply(image, full_mask, params)
        assert result.dtype == np.uint8

    def test_zero_opacity_no_change(self, strategy, image, full_mask):
        params = PartParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=0.0,
        )
        result = strategy.apply(image, full_mask, params)
        np.testing.assert_array_equal(result, image)

    def test_zero_mask_no_change(self, strategy, image):
        zero_mask = np.zeros((50, 50), dtype=np.float32)
        params = PartParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=1.0,
        )
        result = strategy.apply(image, zero_mask, params)
        np.testing.assert_array_equal(result, image)

    def test_full_opacity_blends_to_target(self, strategy, image, full_mask):
        """At opacity 1.0 and full mask, result matches target_rgb channel values."""
        params = PartParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=1.0,
        )
        result = strategy.apply(image, full_mask, params)
        # target_rgb is applied channel-wise; result channels match target_rgb order
        assert result[0, 0, 0] == 255  # R channel
        assert result[0, 0, 1] == 0    # G channel
        assert result[0, 0, 2] == 0    # B channel

    def test_half_opacity_partial_blend(self, strategy, image, full_mask):
        """At 50% opacity, pixel is halfway between original and target."""
        params = PartParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=0.5,
        )
        result = strategy.apply(image, full_mask, params)
        # Original 128 halfway to target 255 → ~192
        expected = int(128 * 0.5 + 255 * 0.5)
        assert abs(int(result[0, 0, 0]) - expected) <= 1

    def test_preserves_input_image(self, strategy, image, full_mask, params):
        original = image.copy()
        strategy.apply(image, full_mask, params)
        np.testing.assert_array_equal(image, original)

    def test_partial_mask_blends_only_masked(self, strategy, image):
        """Only masked regions should be blended; unmasked should stay original."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[0:25, :] = 1.0  # Top half masked
        params = PartParams(
            color_params=ColorParams(target_rgb=(255, 0, 0)),
            opacity=1.0,
        )
        result = strategy.apply(image, mask, params)
        # Bottom half should be unchanged (original gray)
        np.testing.assert_array_equal(result[25:50, :, :], image[25:50, :, :])
        # Top half should be different
        assert not np.allclose(result[0:25, :, :], image[0:25, :, :])


class TestFoundationPass:
    """Tests for FoundationPass construction."""

    @pytest.fixture
    def params(self):
        return PartParams(
            color_params=ColorParams(target_rgb=(200, 200, 200)),
            opacity=0.3,
        )

    def test_pass_type(self, params):
        p = FoundationPass(params=params)
        assert p.pass_type == "foundation"

    def test_priority_is_zero(self, params):
        p = FoundationPass(params=params)
        assert p.priority == 0

    def test_exclusion_landmarks_set_on_pass(self, params):
        """Exclusion regions are passed as exclusion_landmarks so the
        renderer applies them via the standard AND NOT gate."""
        p = FoundationPass(params=params)
        assert len(p.exclusion_landmarks) == 6  # eyes + eyelids + lips

    def test_has_mask_strategy(self, params):
        p = FoundationPass(params=params)
        assert isinstance(p.mask_strategy, FoundationMaskStrategy)

    def test_has_blend_strategy(self, params):
        p = FoundationPass(params=params)
        assert isinstance(p.blend_strategy, FoundationBlendStrategy)

    def test_mask_strategy_stores_softness(self, params):
        """FoundationPass wires params.softness → strategy._softness."""
        p = FoundationPass(params=params)
        assert p.mask_strategy._softness == 0.5

    def test_exclusion_regions_defined(self):
        """_EXCLUSION_REGIONS constant is defined and passed as
        exclusion_landmarks to the pipeline via FoundationPass.__init__."""
        assert len(_EXCLUSION_REGIONS) == 6  # 2 eyes + 2 eyelids + 2 lips
        for group in _EXCLUSION_REGIONS:
            assert isinstance(group, list)
            assert len(group) > 0
            assert all(isinstance(i, int) for i in group)

    def test_lips_in_exclusion_regions(self):
        """Lips MUST be in _EXCLUSION_REGIONS — always excluded via the
        renderer's exclusion_landmarks gate, not conditional on a lips
        pass being present."""
        has_upper = False
        has_lower = False
        for group in _EXCLUSION_REGIONS:
            if any(0 <= idx <= 25 for idx in group):
                has_upper = True
            if any(260 <= idx <= 330 for idx in group):
                has_lower = True
        assert has_upper, "No upper-lip-range indices found in _EXCLUSION_REGIONS"
        assert has_lower, "No lower-lip-range indices found in _EXCLUSION_REGIONS"

    def test_params_softness_flows_to_strategy(self):
        """Higher params.softness → higher strategy._softness."""
        low = PartParams(
            color_params=ColorParams(target_rgb=(200, 200, 200)),
            opacity=0.3, softness=0.0,
        )
        high = PartParams(
            color_params=ColorParams(target_rgb=(200, 200, 200)),
            opacity=0.3, softness=1.0,
        )
        p_low = FoundationPass(params=low)
        p_high = FoundationPass(params=high)
        assert p_low.mask_strategy._softness == 0.0
        assert p_high.mask_strategy._softness == 1.0
