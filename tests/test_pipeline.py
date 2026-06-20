"""
Tests for core pipeline abstractions: MakeupPass, topological_sort, MaskStrategy, BlendStrategy.
"""
import pytest
import numpy as np

from face_makeup.core.pipeline import (
    MakeupPass,
    MaskStrategy,
    BlendStrategy,
    topological_sort,
)
from face_makeup.core.types import PartParams, ColorParams


class _DummyMaskStrategy(MaskStrategy):
    def generate(self, coords, image_shape, image=None, landmarks=None, segmenter=None):
        return np.zeros(image_shape, dtype=np.uint8)


class _DummyBlendStrategy(BlendStrategy):
    def apply(self, image, mask, params, blender=None, segmenter=None, landmarks=None):
        return image.copy()


def _make_pass(pass_type, priority=0, exclusion_passes=None):
    return MakeupPass(
        pass_type=pass_type,
        priority=priority,
        params=PartParams(
            color_params=ColorParams(target_rgb=(128, 128, 128)),
            opacity=0.5,
        ),
        landmark_indices=[],
        mask_strategy=_DummyMaskStrategy(),
        blend_strategy=_DummyBlendStrategy(),
        exclusion_passes=exclusion_passes or [],
    )


class TestTopologicalSort:
    """Tests for topological_sort function."""

    def test_empty_list(self):
        assert topological_sort([]) == []

    def test_single_pass(self):
        p = _make_pass("a", priority=0)
        result = topological_sort([p])
        assert len(result) == 1
        assert result[0].pass_type == "a"

    def test_no_exclusions_returns_all(self):
        p1 = _make_pass("a", priority=1)
        p2 = _make_pass("b", priority=2)
        p3 = _make_pass("c", priority=0)
        result = topological_sort([p1, p2, p3])
        assert len(result) == 3
        assert {p.pass_type for p in result} == {"a", "b", "c"}

    def test_exclusion_source_comes_before_dependent(self):
        p1 = _make_pass("source", priority=1)
        p2 = _make_pass("dependent", priority=2, exclusion_passes=["source"])
        result = topological_sort([p2, p1])
        assert [p.pass_type for p in result] == ["source", "dependent"]

    def test_multiple_dependents_on_same_source(self):
        source = _make_pass("foundation", priority=0)
        dep1 = _make_pass("lips", priority=20, exclusion_passes=["foundation"])
        dep2 = _make_pass("cheek", priority=15, exclusion_passes=["foundation"])
        result = topological_sort([dep2, dep1, source])
        assert result[0].pass_type == "foundation"
        assert set(p.pass_type for p in result[1:]) == {"lips", "cheek"}

    def test_chain_dependency(self):
        a = _make_pass("a", priority=0)
        b = _make_pass("b", priority=10, exclusion_passes=["a"])
        c = _make_pass("c", priority=20, exclusion_passes=["b"])
        result = topological_sort([c, b, a])
        assert [p.pass_type for p in result] == ["a", "b", "c"]

    def test_cycle_detection(self):
        a = _make_pass("a", priority=0, exclusion_passes=["b"])
        b = _make_pass("b", priority=10, exclusion_passes=["a"])
        with pytest.raises(ValueError, match="Unresolvable exclusion"):
            topological_sort([a, b])

    def test_self_referencing_exclusion_raises(self):
        p = _make_pass("a", priority=0, exclusion_passes=["a"])
        with pytest.raises(ValueError, match="Unresolvable exclusion"):
            topological_sort([p])

    def test_missing_exclusion_source_skipped(self):
        """Exclusion referencing non-existent pass is silently ignored."""
        p = _make_pass("a", priority=0, exclusion_passes=["nonexistent"])
        result = topological_sort([p])
        assert len(result) == 1
        assert result[0].pass_type == "a"

    def test_mixed_with_and_without_exclusions(self):
        no_ex = _make_pass("noex", priority=0)
        source = _make_pass("source", priority=5)
        dependent = _make_pass("dependent", priority=10, exclusion_passes=["source"])
        result = topological_sort([dependent, source, no_ex])
        types = [p.pass_type for p in result]
        assert set(types) == {"noex", "source", "dependent"}
        # Source must come before dependent
        assert types.index("source") < types.index("dependent")


class TestMakeupPassProperties:
    """Tests for MakeupPass dataclass properties."""

    def test_default_exclusion_passes_empty(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
            landmark_indices=[],
        )
        assert p.exclusion_passes == []

    def test_default_mask_strategy_none(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
            landmark_indices=[],
        )
        assert p.mask_strategy is None

    def test_default_blend_strategy_none(self):
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
            landmark_indices=[],
        )
        assert p.blend_strategy is None

    def test_priority_field(self):
        p1 = _make_pass("low", priority=0)
        p2 = _make_pass("high", priority=100)
        assert p1.priority < p2.priority

    def test_landmark_indices_list(self):
        indices = [1, 2, 3]
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
            landmark_indices=indices,
        )
        assert p.landmark_indices == indices

    def test_landmark_indices_dict(self):
        indices = {"left": [1, 2], "right": [3, 4]}
        p = MakeupPass(
            pass_type="test", priority=0,
            params=PartParams(color_params=ColorParams(target_rgb=(128, 128, 128)), opacity=0.5),
            landmark_indices=indices,
        )
        assert p.landmark_indices == indices

    def test_exclusion_passes_stored(self):
        p = _make_pass("test", exclusion_passes=["a", "b"])
        assert p.exclusion_passes == ["a", "b"]
