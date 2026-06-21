"""
Glitter (lamé) particle system for eye shadow makeup.

Provides particle generation, sparkle rendering, twinkle/colour shimmer,
and bloom post-processing as composable stages. Designed for CPU-only
execution via OpenCV + NumPy.

Brightness and colour are both driven by each particle's random static
phase rather than 3D view angle — an earlier angle-tracking approach (via
face landmark normals) added significant code for a visually negligible
effect, so it was removed in favour of this simpler model. Particle colour
is a fixed silver-white, independent of the eyeshadow base colour — real
lamé reads as its own sparkle layer, not a lit-up tint of the shadow
underneath.

Usage::

    from face_makeup.core.blend.glitter import (
        ParticleSet,
        apply_bloom,
        compute_iridescent_colors,
        compute_static_twinkle,
        generate_particles,
        render_sparkle_overlay,
    )

    from face_makeup.core.blend.blend_utils import apply_glitter

    result = apply_glitter(image, mask, intensity, rng=rng)
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


class ParticleSet(NamedTuple):
    """Collection of glitter particles.

    Attributes:
        positions: (P, 2) float32 array of image-space coordinates (x, y).
        sizes: (P,) float32 array of radii in pixels.
        phases: (P,) float32 array of static phase offsets in [0, 2π).
        base_color: (3,) float32 BGR colour (0-255) — fixed silver-white,
            independent of the eyeshadow base colour.
    """

    positions: np.ndarray
    sizes: np.ndarray
    phases: np.ndarray
    base_color: np.ndarray


# ── particle generation ──────────────────────────────────────────────────────


def generate_particles(
    mask: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
    base_size: float = 0.5,
    density_scale: float = 0.03,
    min_distance_factor: float = 1,
) -> ParticleSet:
    """Generate glitter particles via Poisson-disk (dart-throwing) sampling.

    Repeatedly drops candidate points on mask pixels and rejects any that
    fall closer than ``min_distance`` to an already-placed particle. This
    keeps particles spread apart without the visible grid alignment that
    stratified (cell-based) sampling produces — a regular grid of circles
    reads as scale-like once blurred and bloomed.

    Args:
        mask: Float32 soft mask (0.0-1.0), shape (H, W).
        intensity: Glitter intensity 0.0-1.0 — controls particle density.
        rng: Seeded random generator for deterministic output.
        base_size: Base particle radius in pixels (default 0.5).
        density_scale: Multiplier on mask_area * intensity for target
            particle count (default 0.03).
        min_distance_factor: Poisson-disk rejection radius as a fraction
            of the average-cell spacing (default 1). Lower values allow
            tighter packing; higher values enforce stricter spacing.

    Returns:
        ParticleSet with positions, sizes, phases, and fixed silver-white
        base colour — glitter reads as its own cosmetic-grade sparkle
        rather than a lit-up version of the eyeshadow colour underneath.
    """
    mask_bool = mask > 0.1

    if not mask_bool.any():
        return ParticleSet(
            positions=np.empty((0, 2), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.float32),
            phases=np.empty((0,), dtype=np.float32),
            base_color=_GLITTER_BASE_COLOR_BGR.copy(),
        )

    # Determine target particle count
    mask_area = int(mask_bool.sum())
    target_count = max(1, int(mask_area * intensity * density_scale))

    ys_masked, xs_masked = np.where(mask_bool)
    num_mask_px = len(xs_masked)
    h, w = mask_bool.shape

    # Minimum spacing derived from the average area per particle. A 0.65
    # factor was tried first but left too many particles sitting right at
    # the rejection floor, reading as small touching clumps with sparse
    # gaps elsewhere. Values closer to 1.0 keep spacing close to true
    # Poisson-disk strictness for a visibly more even (blue-noise) distribution.
    min_distance = float(np.sqrt(mask_area / target_count) * min_distance_factor)

    positions_list: list[tuple[float, float]] = []
    placed = np.empty((0, 2), dtype=np.float32)
    max_attempts = target_count * 60
    attempts = 0

    while len(positions_list) < target_count and attempts < max_attempts:
        idx = rng.integers(0, num_mask_px)
        cx = float(xs_masked[idx]) + float(rng.uniform(-0.5, 0.5))
        cy = float(ys_masked[idx]) + float(rng.uniform(-0.5, 0.5))

        # Sub-pixel jitter can push a candidate across the mask boundary
        # (e.g. into pixel 24 from edge pixel 25) — reject those so every
        # particle stays strictly inside the intended region.
        px_int = int(np.clip(cx, 0, w - 1))
        py_int = int(np.clip(cy, 0, h - 1))
        if not mask_bool[py_int, px_int]:
            attempts += 1
            continue

        if len(positions_list) > 0:
            dists = np.hypot(placed[:, 0] - cx, placed[:, 1] - cy)
            if dists.min() < min_distance:
                attempts += 1
                continue

        positions_list.append((cx, cy))
        placed = np.array(positions_list, dtype=np.float32)
        attempts += 1

    positions = placed if len(positions_list) > 0 else np.empty((0, 2), dtype=np.float32)
    num_particles = len(positions)

    # Particle sizes: fixed range, ±50% random variation from base_size.
    # Intensity controls density (particle count) only, not size — larger
    # flakes at high intensity read as chunky glitter rather than a denser
    # fine sparkle.
    sizes = base_size * (0.5 + 0.5 * rng.uniform(size=num_particles)).astype(np.float32)

    # Static phase for twinkle: [0, 2π)
    phases = rng.uniform(0, 2.0 * np.pi, size=num_particles).astype(np.float32)

    # Fixed silver-white base colour (BGR for OpenCV drawing)
    base_color = _GLITTER_BASE_COLOR_BGR.copy()

    return ParticleSet(
        positions=positions,
        sizes=sizes,
        phases=phases,
        base_color=base_color,
    )


# ── brightness & colour models ──────────────────────────────────────────────


def compute_static_twinkle(phases: np.ndarray) -> np.ndarray:
    """Compute per-particle brightness from frozen sine phase.

    Maps each particle's phase to a brightness in [0.0, 1.0] via
    ``clip(0.3 + 0.7 * sin(phase), 0, 1)``, giving a natural mix
    of dim and bright particles at any snapshot.

    Args:
        phases: (P,) float32 array of phase values in [0, 2π).

    Returns:
        (P,) float32 brightness multipliers in [0.0, 1.0].
    """
    if len(phases) == 0:
        return np.array([], dtype=np.float32)
    raw = 0.3 + 0.7 * np.sin(phases)
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def compute_iridescent_colors(
    positions: np.ndarray,
    phases: np.ndarray,
    base_color: np.ndarray,
    hue_range_deg: float = 180.0,
) -> np.ndarray:
    """Compute per-particle BGR colour with interference-style hue shift.

    Real lamé pigment shifts colour with viewing angle (e.g. pink → blue →
    gold) due to thin-film interference (spec §4). Rather than tracking
    actual view angle, each particle's static twinkle phase drives a hue
    jitter, so glitter reads as multi-tonal rather than a single flat
    colour without the cost of 3D normal lookups.

    Args:
        positions: (P, 2) float32 particle positions (only used for count).
        phases: (P,) float32 static twinkle phases — hue jitter driver.
        base_color: (3,) float32 BGR base colour (0-255).
        hue_range_deg: Maximum hue rotation in either direction, degrees.
            ``sin(phase)`` spans [-1, 1], so the shift spans
            [-hue_range_deg, +hue_range_deg]; 180° makes that span the
            full hue wheel (-180° and +180° are the same hue).

    Returns:
        (P, 3) float32 BGR colours, one per particle.
    """
    num = len(positions)
    if num == 0:
        return np.empty((0, 3), dtype=np.float32)

    angle_metric = np.sin(phases)
    hue_shift_deg = (angle_metric * hue_range_deg).astype(np.float32)

    # Per-particle saturation gain instead of one fixed value for every
    # particle — driven by a different harmonic of the same phase (3x
    # frequency) so it varies independently of hue_shift_deg rather than
    # scaling in lockstep with it. Stays fully deterministic for a given
    # seed since it's derived from the particle's own random phase.
    saturation_gain = (0.75 + 0.25 * np.cos(phases * 3.0)).astype(np.float32)

    return _shift_hue(base_color, hue_shift_deg, saturation_gain)


# ── sparkle rendering ─────────────────────────────────────────────────────────


def render_sparkle_overlay(
    image_shape: tuple[int, int],
    particles: ParticleSet,
    brightness: np.ndarray,
    colors: np.ndarray | None = None,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Render faceted sparkles to a blank overlay.

    Draws a rotated quadrilateral (diamond) at each particle position —
    real glitter is made of flat faceted flakes, and plain circles read
    as soft, characterless dots. Each particle's rotation is derived from
    its static phase so flakes don't all point the same way, then a light
    Gaussian blur softens the hard polygon edges for a sparkle look.

    Args:
        image_shape: (H, W) of the target image.
        particles: ParticleSet from ``generate_particles``.
        brightness: (P,) float32 per-particle brightness multipliers in [0, 1].
        colors: Optional (P, 3) float32 per-particle BGR colours (0-255), e.g.
            from ``compute_iridescent_colors``. Defaults to the shared
            ``particles.base_color`` for every particle.
        blur_sigma: Gaussian blur sigma for sparkle softening (default 1.0).
            Higher = softer, more diffuse sparkles.

    Returns:
        (H, W, 3) float32 BGR sparkle overlay (no alpha — additive blend later).
    """
    h, w = image_shape
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    if len(particles.positions) == 0:
        return overlay

    if len(brightness) != len(particles.positions):
        raise ValueError(
            f"brightness length {len(brightness)} != positions length "
            f"{len(particles.positions)}"
        )

    positions = particles.positions    # (P, 2) float32 (x, y)
    sizes = particles.sizes            # (P,) float32
    phases = particles.phases          # (P,) float32 — reused as facet rotation

    if colors is None:
        colors = np.tile(particles.base_color, (len(positions), 1))
    elif len(colors) != len(positions):
        raise ValueError(
            f"colors length {len(colors)} != positions length {len(positions)}"
        )

    for i in range(len(positions)):
        px, py = positions[i]
        if px < 0 or py < 0 or px >= w or py >= h:
            continue
        # Dim the particle colour by per-particle brightness
        particle_color = colors[i] * brightness[i]
        # Clamp to valid BGR range
        particle_color = np.clip(particle_color, 0, 255)
        radius = float(sizes[i])
        quad = _quad_points(px, py, radius, float(phases[i]))
        cv2.fillPoly(overlay, [quad], particle_color.tolist(), cv2.LINE_AA)

    # Soften hard polygon edges into faceted sparkles
    if len(positions) > 0:
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=blur_sigma)

    return overlay


# ── bloom ─────────────────────────────────────────────────────────────────────


def apply_bloom(
    image: np.ndarray,
    sparkle_overlay: np.ndarray,
    intensity: float,
    threshold: float = 0.3,
    bloom_strength: float | None = None,
    bloom_sigma: float | None = None,
) -> np.ndarray:
    """Apply bloom (glow halo) around bright sparkle regions.

    Downsamples by 4× for performance, extracts bright regions, applies a wide
    Gaussian blur, then upsamples and additively blends back.

    Args:
        image: (H, W, 3) uint8 BGR image (already composited with sparkles).
        sparkle_overlay: (H, W, 3) float32 BGR sparkle overlay (pre-composite).
        intensity: Glitter intensity 0.0-1.0.
        threshold: Fraction of max overlay luminance above which bloom is triggered.
        bloom_strength: Override bloom strength. Defaults to ``0.3 * intensity``.
        bloom_sigma: Gaussian blur sigma for bloom halo. Defaults to
            ``8.0 + intensity * 16.0``. Higher = wider, softer glow.

    Returns:
        (H, W, 3) uint8 BGR image with bloom applied.
    """
    if intensity <= 0.0 or sparkle_overlay.max() <= 0.0:
        return image

    if bloom_strength is None:
        bloom_strength = 0.3 * intensity

    h, w = image.shape[:2]

    # BT.601 luma from overlay (BGR channel order — OpenCV convention).
    # Weights are 0.299*R + 0.587*G + 0.114*B, applied to BGR-indexed channels.
    luminance = (
        0.114 * sparkle_overlay[:, :, 0]
        + 0.587 * sparkle_overlay[:, :, 1]
        + 0.299 * sparkle_overlay[:, :, 2]
    )

    max_lum = luminance.max()
    if max_lum <= 0.0:
        return image

    # Bright pass
    bright = np.where(luminance > threshold * max_lum, luminance, 0.0).astype(np.float32)

    if bright.max() <= 0.0:
        return image

    # Downscale → blur → upscale (4× performance optimization)
    small_h, small_w = max(1, h // 4), max(1, w // 4)
    small = cv2.resize(bright, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    bloom_sigma = bloom_sigma if bloom_sigma is not None else (8.0 + intensity * 16.0)
    small_blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=bloom_sigma)

    bloom_full = cv2.resize(small_blurred, (w, h), interpolation=cv2.INTER_LINEAR)

    # Additive blend (BGR)
    result = image.astype(np.float32)
    for c in range(3):
        result[:, :, c] += bloom_full * bloom_strength

    return np.clip(result, 0, 255).astype(np.uint8)


# ── helpers ───────────────────────────────────────────────────────────────────


# Fixed silver-white sparkle colour (BGR, 0-255), independent of the
# eyeshadow base colour — real lamé pigment reads as its own cool-white
# highlight, not a lightened tint of the shadow underneath.
_GLITTER_BASE_COLOR_BGR = np.array([255.0, 250.0, 245.0], dtype=np.float32)


def _quad_points(cx: float, cy: float, radius: float, rotation: float) -> np.ndarray:
    """Build the 4 vertices of a rotated quadrilateral (diamond) facet.

    Args:
        cx: Centre x coordinate.
        cy: Centre y coordinate.
        radius: Distance from centre to each vertex.
        rotation: Rotation offset in radians, applied to all 4 vertices.

    Returns:
        (4, 2) int32 array of vertex coordinates for ``cv2.fillPoly``.
    """
    angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float32) + rotation
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _shift_hue(
    base_color: np.ndarray,
    hue_shift_deg: np.ndarray,
    saturation_gain: float | np.ndarray = 0.75,
) -> np.ndarray:
    """Rotate the hue of a shared base BGR colour per particle.

    Converts to HSV, rotates the hue channel (OpenCV's 8-bit HSV stores hue
    in units of 2°, hence the ``/2`` factor), and converts back. The base
    silver-white colour has very low saturation (~4%), so a pure hue
    rotation at fixed saturation is visually imperceptible — every hue
    reads as the same near-white. Saturation is therefore boosted in
    proportion to how far a particle's hue has rotated from the neutral
    base: particles at phase-driven shift 0 stay silver-white, while
    particles rotated toward ±hue_range_deg pick up a visibly iridescent
    tint. Value is preserved so the shift doesn't read as a brightness
    change.

    Args:
        base_color: (3,) float32 BGR colour (0-255), shared base for all particles.
        hue_shift_deg: (P,) float32 hue rotation in degrees.
        saturation_gain: Saturation (0-255 scale) added per degree of hue
            rotation magnitude — scalar (shared) or (P,) float32 (per-particle).

    Returns:
        (P, 3) float32 BGR colours, one per particle.
    """
    num = len(hue_shift_deg)
    if num == 0:
        return np.empty((0, 3), dtype=np.float32)

    bgr_u8 = np.clip(base_color, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    h0, s0, v0 = hsv[0, 0]

    hue_shift_units = hue_shift_deg / 2.0  # degrees → OpenCV's 0-179 hue scale
    hues = np.mod(h0 + hue_shift_units, 180.0)
    saturations = np.clip(s0 + np.abs(hue_shift_deg) * saturation_gain, 0.0, 255.0)

    hsv_particles = np.empty((num, 1, 3), dtype=np.uint8)
    hsv_particles[:, 0, 0] = hues.astype(np.uint8)
    hsv_particles[:, 0, 1] = saturations.astype(np.uint8)
    hsv_particles[:, 0, 2] = np.uint8(v0)

    bgr_particles = cv2.cvtColor(hsv_particles, cv2.COLOR_HSV2BGR).astype(np.float32)
    return bgr_particles.reshape(num, 3)
