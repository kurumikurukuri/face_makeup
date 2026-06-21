"""
Unit tests for glitter debug visualization methods in DebugVisualizer.
"""
import os

import cv2
import numpy as np
import pytest

from face_makeup.core.blend.glitter import (
    ParticleSet,
    compute_iridescent_colors,
    compute_static_twinkle,
    generate_particles,
    render_sparkle_overlay,
)
from face_makeup.core.debug_visualizer import DebugVisualizer


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    """Seeded random generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def eye_mask():
    """Float32 mask covering the centre of a 100×100 image."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[25:75, 25:75] = 1.0
    return mask


@pytest.fixture
def test_image():
    """RGB uint8 image."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def debug_dir(tmp_path):
    """Temporary directory for debug output."""
    return str(tmp_path / "debug")


@pytest.fixture
def visualizer(debug_dir):
    """DebugVisualizer with temporary output directory."""
    return DebugVisualizer(debug_dir=debug_dir)


@pytest.fixture
def particles(eye_mask, rng):
    """ParticleSet generated from a valid mask."""
    return generate_particles(eye_mask, 0.5, rng)


@pytest.fixture
def sparkle_overlay(test_image, eye_mask, rng):
    """Sparkle overlay rendered from generated particles."""
    p = generate_particles(eye_mask, 0.5, rng)
    brightness = compute_static_twinkle(p.phases)
    colors = compute_iridescent_colors(p.positions, p.phases, p.base_color)
    return render_sparkle_overlay((100, 100), p, brightness, colors=colors)


# ── save_glitter_particles ────────────────────────────────────────────────────


class TestSaveGlitterParticles:
    """Tests for DebugVisualizer.save_glitter_particles()."""

    def test_saves_file_with_valid_particles(
        self, visualizer, test_image, particles, debug_dir
    ):
        """File should be created when particles are present."""
        brightness = compute_static_twinkle(particles.phases)
        visualizer.save_glitter_particles(
            test_image, particles, brightness, session_id="test_p"
        )
        path = os.path.join(debug_dir, "test_p_glitter_particles.png")
        assert os.path.exists(path)

    def test_saves_file_with_empty_particles(
        self, visualizer, test_image, debug_dir
    ):
        """Empty ParticleSet should still save a file (no crash)."""
        empty = ParticleSet(
            positions=np.empty((0, 2), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.float32),
            phases=np.empty((0,), dtype=np.float32),
            base_color=np.array([200.0, 150.0, 150.0], dtype=np.float32),
        )
        brightness = np.array([], dtype=np.float32)
        visualizer.save_glitter_particles(
            test_image, empty, brightness, session_id="test_empty_p"
        )
        path = os.path.join(debug_dir, "test_empty_p_glitter_particles.png")
        assert os.path.exists(path)

    def test_image_not_mutated(self, visualizer, test_image, particles):
        """Original image should not be modified."""
        original = test_image.copy()
        brightness = compute_static_twinkle(particles.phases)
        visualizer.save_glitter_particles(
            test_image, particles, brightness, session_id="test_imm"
        )
        assert np.array_equal(test_image, original)

    def test_custom_colors_used(self, visualizer, test_image, particles):
        """Custom colors should be accepted without error."""
        brightness = compute_static_twinkle(particles.phases)
        custom_colors = np.full(
            (len(particles.positions), 3), 200.0, dtype=np.float32
        )
        visualizer.save_glitter_particles(
            test_image, particles, brightness,
            session_id="test_colors", colors=custom_colors,
        )

    def test_brightness_length_mismatch_raises(
        self, visualizer, test_image, particles
    ):
        """Mismatched brightness length should raise ValueError."""
        wrong_brightness = np.array([0.5], dtype=np.float32)  # len 1, not P
        with pytest.raises(ValueError, match="brightness length"):
            visualizer.save_glitter_particles(
                test_image, particles, wrong_brightness, session_id="test_err"
            )

    def test_colors_length_mismatch_raises(
        self, visualizer, test_image, particles
    ):
        """Mismatched colors length should raise ValueError."""
        brightness = compute_static_twinkle(particles.phases)
        wrong_colors = np.array([[200.0, 150.0, 150.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="colors length"):
            visualizer.save_glitter_particles(
                test_image, particles, brightness,
                session_id="test_err2", colors=wrong_colors,
            )

    def test_invalid_session_id_raises(self, visualizer, test_image, particles):
        """Invalid session_id containing path separators should raise."""
        brightness = compute_static_twinkle(particles.phases)
        with pytest.raises(ValueError, match="Invalid session_id"):
            visualizer.save_glitter_particles(
                test_image, particles, brightness, session_id="bad/../path"
            )


# ── save_glitter_sparkle ─────────────────────────────────────────────────────


class TestSaveGlitterSparkle:
    """Tests for DebugVisualizer.save_glitter_sparkle()."""

    def test_saves_file_with_valid_overlay(
        self, visualizer, sparkle_overlay, debug_dir
    ):
        """File should be created for a valid sparkle overlay."""
        visualizer.save_glitter_sparkle(sparkle_overlay, session_id="test_sp")
        path = os.path.join(debug_dir, "test_sp_glitter_sparkle.png")
        assert os.path.exists(path)

    def test_saves_empty_overlay(self, visualizer, debug_dir):
        """Empty (all-zero) overlay should save without error."""
        empty_overlay = np.zeros((100, 100, 3), dtype=np.float32)
        visualizer.save_glitter_sparkle(empty_overlay, session_id="test_empty_sp")
        path = os.path.join(debug_dir, "test_empty_sp_glitter_sparkle.png")
        assert os.path.exists(path)

    def test_overlay_not_mutated(self, visualizer, sparkle_overlay):
        """Sparkle overlay input should not be modified."""
        original = sparkle_overlay.copy()
        visualizer.save_glitter_sparkle(sparkle_overlay, session_id="test_imm_sp")
        assert np.array_equal(sparkle_overlay, original)

    def test_invalid_session_id_raises(self, visualizer, sparkle_overlay):
        """Invalid session_id should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid session_id"):
            visualizer.save_glitter_sparkle(
                sparkle_overlay, session_id="bad/../path"
            )


# ── save_glitter_before_after ────────────────────────────────────────────────


class TestSaveGlitterBeforeAfter:
    """Tests for DebugVisualizer.save_glitter_before_after()."""

    def test_saves_three_files(self, visualizer, test_image, debug_dir):
        """Should save before, after, and diff images."""
        # Create a slightly different "after" image
        after = test_image.copy()
        after[30:70, 30:70] = np.clip(
            after[30:70, 30:70].astype(np.int32) + 30, 0, 255
        ).astype(np.uint8)

        visualizer.save_glitter_before_after(
            test_image, after, session_id="test_ba"
        )

        assert os.path.exists(os.path.join(debug_dir, "test_ba_glitter_before.png"))
        assert os.path.exists(os.path.join(debug_dir, "test_ba_glitter_after.png"))
        assert os.path.exists(os.path.join(debug_dir, "test_ba_glitter_diff.png"))

    def test_identical_images_produce_black_diff(
        self, visualizer, test_image, debug_dir
    ):
        """Diff of identical images should be all black."""
        visualizer.save_glitter_before_after(
            test_image, test_image.copy(), session_id="test_same"
        )
        diff_path = os.path.join(debug_dir, "test_same_glitter_diff.png")
        diff_img = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
        assert diff_img is not None
        assert diff_img.max() == 0  # All black — no difference

    def test_images_not_mutated(self, visualizer, test_image):
        """Input images should not be modified."""
        before_orig = test_image.copy()
        after = test_image.copy()
        after[10:20, 10:20] = 200
        after_orig = after.copy()

        visualizer.save_glitter_before_after(
            test_image, after, session_id="test_imm_ba"
        )
        assert np.array_equal(test_image, before_orig)
        assert np.array_equal(after, after_orig)

    def test_invalid_session_id_raises(self, visualizer, test_image):
        """Invalid session_id should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid session_id"):
            visualizer.save_glitter_before_after(
                test_image, test_image.copy(), session_id="bad/../path"
            )

    def test_different_images_produce_visible_diff(
        self, visualizer, test_image, debug_dir
    ):
        """Meaningfully different images should produce a non-black diff."""
        after = test_image.copy()
        after[40:60, 40:60] = 255  # Bright region added

        visualizer.save_glitter_before_after(
            test_image, after, session_id="test_diff_visible"
        )
        diff_path = os.path.join(debug_dir, "test_diff_visible_glitter_diff.png")
        diff_img = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
        assert diff_img.max() > 0  # Non-black — difference detected
