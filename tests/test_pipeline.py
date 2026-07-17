"""
Tests for core pipeline abstractions: MakeupPass, MaskStrategy, BlendStrategy.
"""
import numpy as np

from face_makeup.core.pipeline import (
    MakeupPass,
    MaskStrategy,
    BlendStrategy,
)
from face_makeup.core.types import PartParams, ColorParams


class _DummyMaskStrategy(MaskStrategy):
    def generate(self, coords, image_shape, image=None, landmarks=None):
        return np.zeros(image_shape, dtype=np.float32)


class _DummyBlendStrategy(BlendStrategy):
    def apply(self, image, mask, params, landmarks=None):
        return image.copy()


def _make_pass(pass_type, priority=0):
    return MakeupPass(
        pass_type=pass_type,
        priority=priority,
        params=PartParams(
            color_params=ColorParams(target_rgb=(128, 128, 128)),
            opacity=0.5,
        ),
        mask_strategy=_DummyMaskStrategy(),
        blend_strategy=_DummyBlendStrategy(),
    )


class TestMakeupPassProperties:
    """Tests for MakeupPass dataclass properties."""

    def test_default_mask_strategy_none(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
        )
        assert p.mask_strategy is None

    def test_default_blend_strategy_none(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
        )
        assert p.blend_strategy is None

    def test_default_exclusion_landmarks_empty(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
        )
        assert p.exclusion_landmarks == []

    def test_default_clip_landmarks_empty(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
        )
        assert p.clip_landmarks == []

    def test_priority_field(self):
        p1 = _make_pass("low", priority=0)
        p2 = _make_pass("high", priority=100)
        assert p1.priority < p2.priority
