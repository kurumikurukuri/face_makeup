"""
Tests for FoundationPass, FoundationMaskStrategy, and FoundationBlendStrategy.
"""
import numpy as np
import pytest

from face_makeup.core.passes.foundation import (
    FoundationMaskStrategy,
    FoundationBlendStrategy,
    FoundationPass,
    blur_kernel_from_strength,
)
from face_makeup.core.types import PartParams, ColorParams


class TestFoundationMaskStrategy:
    """Tests for FoundationMaskStrategy."""

    @pytest.fixture
    def strategy(self):
        return FoundationMaskStrategy(blur_kernel=(5, 5))

    @pytest.fixture
    def oval_coords(self):
        """Simulate a face oval with coordinates forming a rough circle (36 points)."""
        h, w = 100, 100
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

    def test_output_dtype_uint8(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask.dtype == np.uint8

    def test_mask_has_values_between_0_and_255(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_center_is_filled(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask[50, 50] > 0

    def test_corners_are_zero(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        assert mask[0, 0] == 0
        assert mask[0, 99] == 0
        assert mask[99, 0] == 0
        assert mask[99, 99] == 0

    def test_blur_creates_soft_edges(self, strategy, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        mask = strategy.generate(oval_coords, (100, 100))
        unique = np.unique(mask)
        assert len(unique) > 2  # More than just 0 and 255

    def test_larger_blur_kernel(self, oval_coords, test_indices, monkeypatch):
        monkeypatch.setattr(
            "face_makeup.core.passes.foundation.FACE_OVAL_INDICES", test_indices
        )
        sharp = FoundationMaskStrategy(blur_kernel=(1, 1))
        soft = FoundationMaskStrategy(blur_kernel=(31, 31))
        mask_sharp = sharp.generate(oval_coords, (100, 100))
        mask_soft = soft.generate(oval_coords, (100, 100))
        sharp_255_count = np.sum(mask_sharp == 255)
        soft_255_count = np.sum(mask_soft == 255)
        assert soft_255_count <= sharp_255_count


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

    def test_excludes_eyes_eyelids_and_lips_via_landmarks(self, params):
        """Eye interiors, full eyelids, and lips are always excluded via
        exclusion_landmarks — no longer conditional on pass presence."""
        p = FoundationPass(params=params)
        assert len(p.exclusion_landmarks) == 6  # 2 eyes + 2 eyelids + 2 lips

    def test_has_mask_strategy(self, params):
        p = FoundationPass(params=params)
        assert isinstance(p.mask_strategy, FoundationMaskStrategy)

    def test_has_blend_strategy(self, params):
        p = FoundationPass(params=params)
        assert isinstance(p.blend_strategy, FoundationBlendStrategy)

    def test_custom_blur_kernel(self, params):
        p = FoundationPass(params=params, blur_kernel=(15, 15))
        assert p.mask_strategy._blur_kernel == (15, 15)

    def test_default_blur_kernel_matches_default_strength(self, params):
        """params.blur_strength defaults to 0.5, reproducing the original (31, 31) kernel."""
        p = FoundationPass(params=params)
        assert p.mask_strategy._blur_kernel == (31, 31)

    def test_exclusion_landmarks_set(self, params):
        """exclusion_landmarks must include eye interiors, eyelids, and lips."""
        p = FoundationPass(params=params)
        assert len(p.exclusion_landmarks) == 6
        # Each group is a list of landmark indices
        for group in p.exclusion_landmarks:
            assert isinstance(group, list)
            assert len(group) > 0
            assert all(isinstance(i, int) for i in group)

    def test_lips_in_exclusion_landmarks(self, params):
        """Lips MUST be in exclusion_landmarks — always excluded, not
        conditional on a lips pass being present."""
        p = FoundationPass(params=params)
        # At least one exclusion_landmark group should contain lip-range
        # indices (upper lip ~0-20, lower lip ~270-320).
        has_upper = False
        has_lower = False
        for group in p.exclusion_landmarks:
            if any(0 <= idx <= 25 for idx in group):
                has_upper = True
            if any(260 <= idx <= 330 for idx in group):
                has_lower = True
        assert has_upper, "No upper-lip-range indices found in exclusion_landmarks"
        assert has_lower, "No lower-lip-range indices found in exclusion_landmarks"

    def test_blur_kernel_derived_from_params_blur_strength(self):
        """params.blur_strength drives the mask blur kernel (previously ignored)."""
        low = PartParams(
            color_params=ColorParams(target_rgb=(200, 200, 200)),
            opacity=0.3, blur_strength=0.0,
        )
        high = PartParams(
            color_params=ColorParams(target_rgb=(200, 200, 200)),
            opacity=0.3, blur_strength=1.0,
        )
        p_low = FoundationPass(params=low)
        p_high = FoundationPass(params=high)
        assert p_low.mask_strategy._blur_kernel[0] < p_high.mask_strategy._blur_kernel[0]


class TestBlurKernelFromStrength:
    """Tests for the blur_strength -> Gaussian kernel size mapping."""

    def test_default_strength_matches_original_fixed_kernel(self):
        assert blur_kernel_from_strength(0.5) == (31, 31)

    def test_zero_strength_is_minimal_kernel(self):
        assert blur_kernel_from_strength(0.0) == (1, 1)

    def test_kernel_size_is_always_odd(self):
        for strength in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
            size, _ = blur_kernel_from_strength(strength)
            assert size % 2 == 1
