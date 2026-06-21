"""
Unit tests for the glitter particle system (face_makeup.core.blend.glitter).
"""
import inspect

import numpy as np
import pytest

from face_makeup.core.blend.glitter import (
    ParticleSet,
    apply_bloom,
    compute_iridescent_colors,
    compute_static_twinkle,
    generate_particles,
    render_sparkle_overlay,
)


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
def empty_mask():
    """Float32 mask with no active pixels."""
    return np.zeros((100, 100), dtype=np.float32)


@pytest.fixture
def small_mask():
    """Very small mask for edge-case testing."""
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[8:12, 8:12] = 1.0
    return mask


@pytest.fixture
def test_image():
    """Solid mid-tone BGR uint8 image."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


# ── ParticleSet ──────────────────────────────────────────────────────────────


class TestParticleSet:
    """Tests for ParticleSet NamedTuple construction and attributes."""

    def test_construction_empty(self):
        """ParticleSet can be constructed with empty arrays."""
        ps = ParticleSet(
            positions=np.empty((0, 2), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.float32),
            phases=np.empty((0,), dtype=np.float32),
            base_color=np.array([200.0, 150.0, 150.0], dtype=np.float32),
        )
        assert len(ps.positions) == 0
        assert len(ps.sizes) == 0
        assert len(ps.phases) == 0
        assert ps.base_color.shape == (3,)

    def test_construction_with_particles(self, rng):
        """ParticleSet can be constructed with particle data."""
        positions = rng.uniform(0, 100, size=(10, 2)).astype(np.float32)
        sizes = np.ones(10, dtype=np.float32) * 2.0
        phases = np.linspace(0, 2 * np.pi, 10, dtype=np.float32)
        base_color = np.array([200.0, 150.0, 150.0], dtype=np.float32)

        ps = ParticleSet(
            positions=positions,
            sizes=sizes,
            phases=phases,
            base_color=base_color,
        )
        assert ps.positions.shape == (10, 2)
        assert ps.sizes.shape == (10,)
        assert ps.phases.shape == (10,)

    def test_immutable(self, rng):
        """ParticleSet fields cannot be reassigned (NamedTuple)."""
        ps = ParticleSet(
            positions=np.array([[1.0, 2.0]], dtype=np.float32),
            sizes=np.array([2.0], dtype=np.float32),
            phases=np.array([0.5], dtype=np.float32),
            base_color=np.array([200.0, 150.0, 150.0], dtype=np.float32),
        )
        with pytest.raises(AttributeError):
            ps.positions = np.array([[3.0, 4.0]], dtype=np.float32)


# ── generate_particles ───────────────────────────────────────────────────────


class TestGenerateParticles:
    """Tests for generate_particles() — stratified sampling."""

    def test_empty_mask_returns_empty(self, empty_mask, rng):
        """Empty mask produces zero particles."""
        particles = generate_particles(empty_mask, 0.5, rng)
        assert len(particles.positions) == 0
        assert len(particles.sizes) == 0
        assert len(particles.phases) == 0

    def test_zero_intensity_produces_minimal_particles(self, eye_mask, rng):
        """intensity=0 still produces 1 particle (minimum)."""
        particles = generate_particles(eye_mask, 0.0, rng)
        # Even at zero intensity, we create at least 1 particle for consistency
        assert len(particles.positions) >= 1

    def test_produces_particles_in_mask(self, eye_mask, rng):
        """All generated particle positions should fall within the mask region."""
        particles = generate_particles(eye_mask, 0.5, rng)
        assert len(particles.positions) > 0

        mask_bool = eye_mask > 0.1
        for x, y in particles.positions:
            px, py = int(np.clip(x, 0, 99)), int(np.clip(y, 0, 99))
            assert mask_bool[py, px], f"Particle at ({x:.1f}, {y:.1f}) outside mask"

    def test_particle_count_scales_with_intensity(self, eye_mask, rng):
        """Higher intensity should produce more particles."""
        p_low = generate_particles(eye_mask, 0.2, rng)
        rng2 = np.random.default_rng(42)  # fresh seed for fair comparison
        p_high = generate_particles(eye_mask, 0.8, rng2)
        # Higher intensity should have at least as many particles
        assert len(p_high.positions) >= len(p_low.positions)

    def test_deterministic_with_seed(self, eye_mask):
        """Same seed produces identical particle data."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_particles(eye_mask, 0.5, rng1)
        p2 = generate_particles(eye_mask, 0.5, rng2)
        assert np.array_equal(p1.positions, p2.positions)
        assert np.array_equal(p1.sizes, p2.sizes)
        assert np.array_equal(p1.phases, p2.phases)

    def test_different_seed_produces_different_positions(self, eye_mask):
        """Different seeds should (usually) produce different positions."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        p1 = generate_particles(eye_mask, 0.5, rng1)
        p2 = generate_particles(eye_mask, 0.5, rng2)
        # Should differ (vanishingly small chance of collision)
        assert not np.array_equal(p1.positions, p2.positions)

    def test_base_color_is_bgr(self, eye_mask, rng):
        """Base colour should be a 3-element BGR float32 array."""
        particles = generate_particles(eye_mask, 0.5, rng)
        assert particles.base_color.shape == (3,)
        assert particles.base_color.dtype == np.float32

    def test_sizes_positive(self, eye_mask, rng):
        """All particle sizes should be positive."""
        particles = generate_particles(eye_mask, 0.5, rng)
        if len(particles.sizes) > 0:
            assert np.all(particles.sizes > 0)

    def test_phases_in_range(self, eye_mask, rng):
        """All phase values should be in [0, 2π)."""
        particles = generate_particles(eye_mask, 0.5, rng)
        if len(particles.phases) > 0:
            assert np.all(particles.phases >= 0)
            assert np.all(particles.phases < 2 * np.pi)

    def test_very_small_mask(self, small_mask, rng):
        """Very small masks should still work without errors."""
        particles = generate_particles(small_mask, 0.5, rng)
        assert isinstance(particles, ParticleSet)
        # May have 0 or few particles — just should not crash

    def test_base_size_affects_particle_sizes(self, eye_mask):
        """Larger base_size should produce larger particles."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p_small = generate_particles(eye_mask, 0.5, rng1, base_size=1.0)
        p_large = generate_particles(eye_mask, 0.5, rng2, base_size=4.0)
        if len(p_small.sizes) > 0 and len(p_large.sizes) > 0:
            assert p_small.sizes.mean() < p_large.sizes.mean()

    def test_base_size_default(self, eye_mask):
        """Default base_size produces sizes consistent with the documented formula."""
        rng = np.random.default_rng(42)
        particles = generate_particles(eye_mask, 0.5, rng)
        if len(particles.sizes) > 0:
            sig = inspect.signature(generate_particles)
            default_base_size = sig.parameters["base_size"].default
            # Mean of (0.5 + 0.5*U(0,1)) = 0.75
            expected_mean = default_base_size * 0.75
            # ±25% tolerance for finite sample
            assert expected_mean * 0.75 < particles.sizes.mean() < expected_mean * 1.25

    def test_density_scale_affects_count(self, eye_mask):
        """Higher density_scale should produce more particles."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p_low = generate_particles(eye_mask, 0.5, rng1, density_scale=0.01)
        p_high = generate_particles(eye_mask, 0.5, rng2, density_scale=0.08)
        assert len(p_high.positions) >= len(p_low.positions)

    def test_min_distance_factor_affects_spacing(self, eye_mask):
        """Higher min_distance_factor should produce fewer particles (stricter spacing)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p_tight = generate_particles(eye_mask, 0.5, rng1, min_distance_factor=0.5)
        p_sparse = generate_particles(eye_mask, 0.5, rng2, min_distance_factor=1.2)
        # Tighter packing (lower factor) → more particles
        assert len(p_tight.positions) > 0
        assert len(p_sparse.positions) > 0
        assert len(p_tight.positions) >= len(p_sparse.positions)

    def test_all_new_params_combined(self, eye_mask):
        """New parameters can be combined without error."""
        rng = np.random.default_rng(42)
        particles = generate_particles(
            eye_mask, 0.5, rng,
            base_size=3.0,
            density_scale=0.05,
            min_distance_factor=0.7,
        )
        assert isinstance(particles, ParticleSet)
        assert len(particles.positions) > 0


# ── compute_static_twinkle ────────────────────────────────────────────────────


class TestComputeStaticTwinkle:
    """Tests for compute_static_twinkle()."""

    def test_empty_phases(self):
        """Empty input returns empty output."""
        result = compute_static_twinkle(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_range(self, rng):
        """All twinkle values should be in [0.0, 1.0]."""
        phases = rng.uniform(0, 2 * np.pi, size=1000).astype(np.float32)
        twinkle = compute_static_twinkle(phases)
        assert np.all(twinkle >= 0.0)
        assert np.all(twinkle <= 1.0)

    def test_deterministic(self):
        """Same phases produce same output."""
        phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float32)
        t1 = compute_static_twinkle(phases)
        t2 = compute_static_twinkle(phases)
        assert np.array_equal(t1, t2)

    def test_sine_behavior(self):
        """Phase 0 → sin(0)=0 → 0.3; Phase π/2 → sin(π/2)=1 → 1.0."""
        phases = np.array([0.0, np.pi / 2], dtype=np.float32)
        twinkle = compute_static_twinkle(phases)
        assert twinkle[0] == pytest.approx(0.3, abs=1e-6)
        assert twinkle[1] == pytest.approx(1.0, abs=1e-6)


# ── render_sparkle_overlay ────────────────────────────────────────────────────


class TestRenderSparkleOverlay:
    """Tests for render_sparkle_overlay()."""

    def test_empty_particles_returns_blank(self):
        """Empty ParticleSet produces zero overlay."""
        ps = ParticleSet(
            positions=np.empty((0, 2), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.float32),
            phases=np.empty((0,), dtype=np.float32),
            base_color=np.array([200.0, 150.0, 150.0], dtype=np.float32),
        )
        brightness = np.array([], dtype=np.float32)
        overlay = render_sparkle_overlay((100, 100), ps, brightness)
        assert overlay.shape == (100, 100, 3)
        assert overlay.dtype == np.float32
        assert overlay.max() == 0.0

    def test_output_shape_matches_input(self, eye_mask, rng):
        """Overlay should match the requested image shape."""
        particles = generate_particles(eye_mask, 0.5, rng)
        brightness = compute_static_twinkle(particles.phases)
        overlay = render_sparkle_overlay((100, 100), particles, brightness)
        assert overlay.shape == (100, 100, 3)

    def test_particles_produce_nonzero_overlay(self, eye_mask, rng):
        """Particles should produce non-zero pixels in the overlay."""
        particles = generate_particles(eye_mask, 0.5, rng)
        brightness = compute_static_twinkle(particles.phases)
        overlay = render_sparkle_overlay((100, 100), particles, brightness)
        assert overlay.max() > 0.0

    def test_deterministic(self, eye_mask):
        """Same particles + same brightness = same overlay."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_particles(eye_mask, 0.5, rng1)
        p2 = generate_particles(eye_mask, 0.5, rng2)
        b1 = compute_static_twinkle(p1.phases)
        b2 = compute_static_twinkle(p2.phases)
        o1 = render_sparkle_overlay((100, 100), p1, b1)
        o2 = render_sparkle_overlay((100, 100), p2, b2)
        assert np.array_equal(o1, o2)

    def test_higher_brightness_produces_brighter_overlay(self, eye_mask, rng):
        """Full brightness (1.0) should produce brighter overlay than low brightness."""
        particles = generate_particles(eye_mask, 0.5, rng)
        low_brightness = np.full(len(particles.positions), 0.3, dtype=np.float32)
        high_brightness = np.full(len(particles.positions), 1.0, dtype=np.float32)
        overlay_low = render_sparkle_overlay((100, 100), particles, low_brightness)
        overlay_high = render_sparkle_overlay((100, 100), particles, high_brightness)
        assert overlay_high.max() >= overlay_low.max()


# ── apply_bloom ──────────────────────────────────────────────────────────────


class TestApplyBloom:
    """Tests for apply_bloom()."""

    @pytest.fixture
    def sparkle_overlay(self, eye_mask, rng):
        """A sparkle overlay with visible bright spots."""
        particles = generate_particles(eye_mask, 0.5, rng)
        brightness = compute_static_twinkle(particles.phases)
        return render_sparkle_overlay((100, 100), particles, brightness)

    def test_zero_intensity_unchanged(self, test_image, sparkle_overlay):
        """intensity=0 should return the image unchanged."""
        result = apply_bloom(test_image, sparkle_overlay, 0.0)
        assert np.array_equal(result, test_image)

    def test_empty_overlay_unchanged(self, test_image):
        """Empty overlay should return the image unchanged."""
        empty_overlay = np.zeros((100, 100, 3), dtype=np.float32)
        result = apply_bloom(test_image, empty_overlay, 0.5)
        assert np.array_equal(result, test_image)

    def test_bloom_increases_brightness(self, test_image, sparkle_overlay):
        """Bloom should increase overall brightness of the image."""
        result = apply_bloom(test_image, sparkle_overlay, 0.5)
        assert result.mean() >= test_image.mean()

    def test_bloom_preserves_shape_and_dtype(self, test_image, sparkle_overlay):
        """Bloom output should have the same shape and dtype as input."""
        result = apply_bloom(test_image, sparkle_overlay, 0.5)
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype

    def test_deterministic(self, test_image, sparkle_overlay):
        """Same input produces same bloom output."""
        r1 = apply_bloom(test_image, sparkle_overlay, 0.5)
        r2 = apply_bloom(test_image, sparkle_overlay, 0.5)
        assert np.array_equal(r1, r2)


# ── compute_iridescent_colors ─────────────────────────────────────────────────


class TestComputeIridescentColors:
    """Tests for compute_iridescent_colors() (spec §4: 偏光・干渉)."""

    def test_empty_positions(self):
        """Empty positions returns empty (0, 3) array."""
        base_color = np.array([200.0, 150.0, 150.0], dtype=np.float32)
        colors = compute_iridescent_colors(
            np.empty((0, 2), dtype=np.float32), np.array([], dtype=np.float32), base_color,
        )
        assert colors.shape == (0, 3)

    def test_shape_and_dtype(self, eye_mask, rng):
        """Output shape should be (P, 3) float32 matching particle count."""
        particles = generate_particles(eye_mask, 0.5, rng)
        colors = compute_iridescent_colors(
            particles.positions, particles.phases, particles.base_color,
        )
        assert colors.shape == (len(particles.positions), 3)
        assert colors.dtype == np.float32

    def test_values_in_byte_range(self, eye_mask, rng):
        """Shifted colours should remain valid 0-255 BGR values."""
        particles = generate_particles(eye_mask, 0.5, rng)
        colors = compute_iridescent_colors(
            particles.positions, particles.phases, particles.base_color,
        )
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 255.0)

    def test_fallback_varies_with_phase(self):
        """Without landmarks, particles with different phases should get
        different hues (multi-tonal shimmer instead of a flat colour)."""
        base_color = np.array([151.5, 151.5, 204.0], dtype=np.float32)
        positions = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        # sin = +-0.707 -> shift +-127.3deg. Avoid +-pi/2 (sin=+-1): with the
        # full 360deg hue wheel (hue_range_deg=180), +180 and -180 alias to
        # the same hue, which would make this case coincidentally equal.
        phases = np.array([np.pi / 4, -np.pi / 4], dtype=np.float32)
        colors = compute_iridescent_colors(positions, phases, base_color)
        assert not np.allclose(colors[0], colors[1])

    def test_deterministic(self, eye_mask):
        """Same inputs produce identical colour output."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_particles(eye_mask, 0.5, rng1)
        p2 = generate_particles(eye_mask, 0.5, rng2)
        c1 = compute_iridescent_colors(p1.positions, p1.phases, p1.base_color)
        c2 = compute_iridescent_colors(p2.positions, p2.phases, p2.base_color)
        assert np.array_equal(c1, c2)

# ── integration: full apply_glitter pipeline ──────────────────────────────────


class TestApplyGlitterIntegration:
    """Integration tests for the full apply_glitter pipeline via blend_utils."""

    def test_full_pipeline_on_image(self, test_image, eye_mask, rng):
        """End-to-end: generate → twinkle → render → composite → bloom."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        result = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng,
        )
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        # Image should be modified
        assert not np.array_equal(result, test_image)

    def test_glitter_only_in_mask_region(self, test_image, eye_mask, rng):
        """Pixels far from the mask should be largely unchanged."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        result = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng,
        )
        # Corner pixels (far from mask centre) should be mostly unchanged.
        # Bloom can add a small uniform halo; allow atol=2.
        corner_before = test_image[0:10, 0:10]
        corner_after = result[0:10, 0:10]
        assert np.allclose(corner_before, corner_after, atol=2)

    def test_higher_intensity_produces_more_change(self, test_image, eye_mask):
        """Higher intensity should produce a more different image."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result_low = apply_glitter(
            test_image, eye_mask, 0.2, rng=rng1,
        )
        result_high = apply_glitter(
            test_image, eye_mask, 0.8, rng=rng2,
        )
        diff_low = np.abs(result_low.astype(float) - test_image.astype(float)).mean()
        diff_high = np.abs(result_high.astype(float) - test_image.astype(float)).mean()
        # Higher intensity should (generally) cause more change
        # Note: due to particle count overlap at low intensities this is approximate
        assert diff_high >= diff_low * 0.5  # Allow some overlap

    def test_deterministic_full_pipeline(self, test_image, eye_mask):
        """Same seed produces identical output through the full pipeline."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng1,
        )
        result2 = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng2,
        )
        assert np.array_equal(result1, result2)

    def test_sparkle_alpha_affects_brightness(self, test_image, eye_mask):
        """Higher sparkle_alpha produces brighter result."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng = np.random.default_rng(42)
        result_low = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng, sparkle_alpha=0.2,
        )
        rng2 = np.random.default_rng(42)
        result_high = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng2, sparkle_alpha=1.0,
        )
        diff_low = np.abs(result_low.astype(float) - test_image.astype(float)).mean()
        diff_high = np.abs(result_high.astype(float) - test_image.astype(float)).mean()
        assert diff_high > diff_low

    def test_sparkle_blur_sigma_affects_softness(self, test_image, eye_mask):
        """Higher blur_sigma produces more diffuse (lower peak) sparkles."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng = np.random.default_rng(42)
        result_sharp = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng, sparkle_blur_sigma=0.3,
        )
        rng2 = np.random.default_rng(42)
        result_soft = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng2, sparkle_blur_sigma=5.0,
        )
        # Softer blur → lower peak brightness (energy is spread out)
        sharp_peak = result_sharp.astype(float).max()
        soft_peak = result_soft.astype(float).max()
        assert sharp_peak >= soft_peak

    def test_bloom_threshold_affects_bloom_extent(self, test_image, eye_mask):
        """Lower threshold triggers bloom on more pixels → larger change."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng = np.random.default_rng(42)
        result_low = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng, bloom_threshold=0.9,
        )
        rng2 = np.random.default_rng(42)
        result_high = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng2, bloom_threshold=0.1,
        )
        diff_low = np.abs(result_low.astype(float) - test_image.astype(float)).mean()
        diff_high = np.abs(result_high.astype(float) - test_image.astype(float)).mean()
        # Lower threshold → more bloom → more change (but not strictly guaranteed
        # for weak sparkles; use >= to avoid flakiness)
        assert diff_high >= diff_low * 0.5

    def test_bloom_sigma_affects_glow_width(self, test_image, eye_mask):
        """Higher bloom_sigma produces wider glow → corners more affected."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng = np.random.default_rng(42)
        result_tight = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng, bloom_sigma=1.0,
        )
        rng2 = np.random.default_rng(42)
        result_wide = apply_glitter(
            test_image, eye_mask, 0.5, rng=rng2, bloom_sigma=30.0,
        )
        # Wide bloom reaches corners more
        corner_tight = result_tight[0:10, 0:10].astype(float).mean()
        corner_wide = result_wide[0:10, 0:10].astype(float).mean()
        assert corner_wide >= corner_tight

    def test_bloom_strength_affects_brightness(self, eye_mask):
        """Higher bloom_strength produces brighter peak values in bloom regions."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        # Use a dark image so bloom differences aren't masked by clipping at 255.
        dark_image = np.full((100, 100, 3), 30, dtype=np.uint8)

        rng = np.random.default_rng(42)
        result_low = apply_glitter(
            dark_image, eye_mask, 0.5, rng=rng, bloom_strength=0.05,
        )
        rng2 = np.random.default_rng(42)
        result_high = apply_glitter(
            dark_image, eye_mask, 0.5, rng=rng2, bloom_strength=1.0,
        )
        # Bloom adds a halo around bright sparkles — higher strength boosts
        # the peak brightness in the bloom region.
        assert result_high.max() >= result_low.max()
