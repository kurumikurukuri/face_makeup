"""
Integration tests for end-to-end makeup workflow.
"""
import pytest
import numpy as np
import cv2
import glob
import os
import tempfile

from face_makeup.processor import MakeupProcessor, ProcessorConfig
from face_makeup.core.types import (
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams,
    ColorParams,
)


@pytest.fixture
def processor():
    """MakeupProcessor instance."""
    return MakeupProcessor()


@pytest.fixture
def test_image():
    """Load first test image."""
    image_files = sorted(glob.glob("tests/test_images/*.png"))
    assert len(image_files) > 0, "No test images found"
    image = cv2.imread(image_files[0])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture
def test_image_path():
    """Get first test image path."""
    image_files = sorted(glob.glob("tests/test_images/*.png"))
    assert len(image_files) > 0, "No test images found"
    return image_files[0]


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_makeup_workflow(self, processor, test_image):
        """Test complete makeup application workflow."""
        # Open image
        ctx = processor.open(test_image)
        assert ctx.face_count >= 1

        # Apply all makeup types
        face = ctx.faces[0]
        face.lip(color=(255, 100, 100), opacity=0.5)
        face.cheek(color=(255, 180, 180), opacity=0.3)
        face.eyelid(color=(180, 130, 150), opacity=0.3)
        face.eyeliner(color=(0, 0, 0), opacity=0.8)
        face.eyeiris(color=(100, 150, 255), opacity=0.5)

        # Render
        result = ctx.render()

        # Verify result
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, test_image)

    def test_workflow_with_path(self, processor, test_image_path):
        """Test workflow using file path."""
        ctx = processor.open_path(test_image_path)
        assert ctx.face_count >= 1

        face = ctx.faces[0]
        face.lip(color=(255, 100, 100), opacity=0.5)

        result = ctx.render()
        assert result is not None

    def test_workflow_save_result(self, processor, test_image):
        """Test saving rendered result."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            ctx.save(output_path)
            assert os.path.exists(output_path)

            # Verify saved image
            saved = cv2.imread(output_path)
            assert saved is not None
            assert saved.shape[:2] == test_image.shape[:2]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_workflow_multiple_renders(self, processor, test_image):
        """Test multiple render calls with different makeup."""
        ctx = processor.open(test_image)
        face = ctx.faces[0]

        # First render with lip only
        face.lip(color=(255, 0, 0), opacity=0.5)
        result1 = ctx.render()

        # Add cheek and render again
        face.cheek(color=(255, 180, 180), opacity=0.3)
        result2 = ctx.render()

        # Results should be different
        assert not np.array_equal(result1, result2)

    def test_workflow_clear_and_reapply(self, processor, test_image):
        """Test clearing makeup and reapplying."""
        ctx = processor.open(test_image)
        face = ctx.faces[0]

        # Apply makeup
        face.lip(color=(255, 0, 0), opacity=0.5)
        result_with_makeup = ctx.render()

        # Clear and verify
        face.clear()
        result_cleared = ctx.render()
        assert np.array_equal(result_cleared, test_image)

        # Reapply different makeup
        face.lip(color=(180, 40, 100), opacity=0.5)
        result_reapplied = ctx.render()
        assert not np.array_equal(result_reapplied, result_with_makeup)


class TestFluentAPIChaining:
    """Tests for fluent API chaining."""

    def test_chain_all_methods(self, processor, test_image):
        """Test chaining all makeup methods."""
        ctx = processor.open(test_image)

        result = (
            ctx.faces[0]
            .lip(color=(255, 100, 100), opacity=0.5)
            .cheek(color=(255, 180, 180), opacity=0.3, side='both')
            .eyelid(color=(180, 130, 150), opacity=0.3, side='both')
            .eyelid_edge(color=(150, 100, 120), opacity=0.2, side='both')
            .eyeliner(color=(0, 0, 0), opacity=0.8, side='both')
            .eyeiris(color=(100, 150, 255), opacity=0.5, side='both')
        )

        assert result is ctx.faces[0]
        assert len(ctx.faces[0].makeup_layers) >= 7

    def test_chain_with_params_objects(self, processor, test_image):
        """Test chaining with parameter objects."""
        ctx = processor.open(test_image)

        lip_params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        cheek_params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 180, 180)),
            opacity=0.3,
            center=(0.0, 0.0),
            radius=(0.6, 0.5)
        )

        result = ctx.faces[0].lip(lip_params).cheek(cheek_params)
        assert result is ctx.faces[0]


class TestProcessorReusability:
    """Tests for processor reusability."""

    def test_process_multiple_images(self, processor):
        """Test processing multiple images with same processor."""
        image_files = sorted(glob.glob("tests/test_images/*.png"))

        results = []
        for img_path in image_files[:3]:
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                ctx = processor.open(image_rgb)
                ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
                result = ctx.render()
                results.append(result)
            except ValueError:
                continue

        assert len(results) > 0

    def test_processor_with_custom_config(self):
        """Test processor with custom configuration."""
        config = ProcessorConfig(
            min_detection_confidence=0.7,
            max_num_faces=5
        )
        processor = MakeupProcessor(config=config)

        image_files = sorted(glob.glob("tests/test_images/*.png"))
        image = cv2.imread(image_files[0])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ctx = processor.open(image_rgb)
        assert ctx is not None


class TestMakeupLayerCombinations:
    """Tests for different makeup layer combinations."""

    def test_lip_only(self, processor, test_image):
        """Test lip makeup only."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)
        result = ctx.render()
        assert not np.array_equal(result, test_image)

    def test_cheek_only(self, processor, test_image):
        """Test cheek makeup only."""
        ctx = processor.open(test_image)
        ctx.faces[0].cheek(color=(255, 180, 180), opacity=0.3)
        result = ctx.render()
        assert not np.array_equal(result, test_image)

    def test_eyelid_only(self, processor, test_image):
        """Test eyelid makeup only."""
        ctx = processor.open(test_image)
        ctx.faces[0].eyelid(color=(180, 130, 150), opacity=0.3)
        result = ctx.render()
        assert not np.array_equal(result, test_image)

    def test_eyeliner_only(self, processor, test_image):
        """Test eyeliner makeup only."""
        ctx = processor.open(test_image)
        ctx.faces[0].eyeliner(color=(0, 0, 0), opacity=0.8)
        result = ctx.render()
        assert not np.array_equal(result, test_image)

    def test_eyeiris_only(self, processor, test_image):
        """Test eye iris makeup only."""
        ctx = processor.open(test_image)
        ctx.faces[0].eyeiris(color=(100, 150, 255), opacity=0.5)
        result = ctx.render()
        assert not np.array_equal(result, test_image)

    def test_left_side_only(self, processor, test_image):
        """Test makeup on left side only."""
        ctx = processor.open(test_image)
        ctx.faces[0].cheek(color=(255, 180, 180), opacity=0.3, side='left')
        ctx.faces[0].eyelid(color=(180, 130, 150), opacity=0.3, side='left')
        result = ctx.render()
        assert not np.array_equal(result, test_image)

    def test_right_side_only(self, processor, test_image):
        """Test makeup on right side only."""
        ctx = processor.open(test_image)
        ctx.faces[0].cheek(color=(255, 180, 180), opacity=0.3, side='right')
        ctx.faces[0].eyelid(color=(180, 130, 150), opacity=0.3, side='right')
        result = ctx.render()
        assert not np.array_equal(result, test_image)


class TestColorVariations:
    """Tests for different color variations."""

    def test_natural_lip_colors(self, processor, test_image):
        """Test natural lip colors."""
        ctx = processor.open(test_image)
        natural_colors = [
            (200, 130, 130),  # Natural pink
            (180, 100, 100),  # Nude
            (220, 150, 150),  # Light pink
        ]

        for color in natural_colors:
            ctx.faces[0].clear()
            ctx.faces[0].lip(color=color, opacity=0.4)
            result = ctx.render()
            assert result.shape == test_image.shape

    def test_bold_lip_colors(self, processor, test_image):
        """Test bold lip colors."""
        ctx = processor.open(test_image)
        bold_colors = [
            (255, 0, 0),      # Red
            (200, 0, 100),    # Berry
            (150, 50, 100),   # Plum
        ]

        for color in bold_colors:
            ctx.faces[0].clear()
            ctx.faces[0].lip(color=color, opacity=0.6)
            result = ctx.render()
            assert result.shape == test_image.shape

    def test_eyeshadow_colors(self, processor, test_image):
        """Test different eyeshadow colors."""
        ctx = processor.open(test_image)
        eyeshadow_colors = [
            (180, 130, 150),  # Mauve
            (150, 130, 100),  # Brown
            (100, 100, 150),  # Blue-gray
            (200, 180, 150),  # Gold
        ]

        for color in eyeshadow_colors:
            ctx.faces[0].clear()
            ctx.faces[0].eyelid(color=color, opacity=0.3)
            result = ctx.render()
            assert result.shape == test_image.shape


class TestOpacityVariations:
    """Tests for different opacity levels."""

    def test_subtle_makeup(self, processor, test_image):
        """Test very subtle makeup (low opacity)."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.1)
        ctx.faces[0].cheek(color=(255, 180, 180), opacity=0.1)
        result = ctx.render()
        # Should be slightly different from original
        diff = np.mean(np.abs(result.astype(float) - test_image.astype(float)))
        assert diff < 20  # Small change

    def test_heavy_makeup(self, processor, test_image):
        """Test heavy makeup (high opacity)."""
        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 0, 0), opacity=0.8)
        ctx.faces[0].cheek(color=(255, 150, 150), opacity=0.5)
        result = ctx.render()
        # Should be noticeably different from original
        assert not np.array_equal(result, test_image)


class TestDebugMode:
    """Tests for debug output mode."""

    def test_render_with_debug(self, processor, test_image):
        """Test that render(debug=True) produces valid debug output files."""
        import shutil

        debug_dir = "outputs/debug"
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)

        ctx = processor.open(test_image)
        assert ctx.face_count >= 1

        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)

        result = ctx.render(debug=True)

        # Verify result shape and type
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

        # Verify debug files were written to outputs/debug/
        assert os.path.exists(debug_dir), (
            f"Debug directory {debug_dir} was not created"
        )
        debug_files = os.listdir(debug_dir)
        assert len(debug_files) > 0, "No debug files were written"

        # Verify expected debug file categories exist
        expected_suffixes = [
            "0_original", "1_bbox", "2_landmarks", "5_result", "6_diff"
        ]
        for suffix in expected_suffixes:
            found = any(suffix in f for f in debug_files)
            assert found, f"Missing debug file with suffix: {suffix}"

    def test_save_with_debug(self, processor, test_image):
        """Test that save(path, debug=True) saves result and debug files."""
        import shutil

        debug_dir = "outputs/debug"
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)

        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 100, 100), opacity=0.5)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            ctx.save(output_path, debug=True)

            # Verify result file was created
            assert os.path.exists(output_path)
            saved = cv2.imread(output_path)
            assert saved is not None

            # Verify debug files were written
            assert os.path.exists(debug_dir)
            debug_files = os.listdir(debug_dir)
            assert len(debug_files) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_render_debug_bypasses_cache(self, processor, test_image):
        """Test that render(debug=True) does not return stale cached data."""
        import shutil

        debug_dir = "outputs/debug"
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)

        ctx = processor.open(test_image)
        ctx.faces[0].lip(color=(255, 0, 0), opacity=0.8)

        # Render normally to populate the render cache
        _result_cached = ctx.render()

        # Change makeup (invalidates cache) and render with debug
        ctx.faces[0].clear()
        ctx.faces[0].lip(color=(180, 40, 100), opacity=0.8)
        result_debug = ctx.render(debug=True)

        # Debug render must return the fresh (green lip) result
        assert isinstance(result_debug, np.ndarray)
        assert result_debug.shape == test_image.shape
        assert result_debug.dtype == np.uint8
