"""
Shared blend utilities for MakeupPass BlendStrategy implementations.

Provides mask softening, target color application, hue-shift blending,
and the standard HSV-aware blend pipeline used by all makeup passes.
"""
from __future__ import annotations

import cv2
import numpy as np


# Opacity retained at the very mask edge. The feather ramps from this floor up
# to 1.0 across the feather band, so the boundary keeps a little colour (no hard
# 0→opacity step) while the old floor-rim all but disappears. This is the
# fallback default; passes override it per-part via the ``PartParams`` subclass
# ``_SOFTEN_FLOOR`` ClassVar (e.g. cheek / eyelid = 0.05).
_SOFTEN_FLOOR = 0.1

# softness (0..1) → feather width as a fraction of the inter-ocular span:
#   feather_px = softness * feather_scale * inter_ocular_distance(coords)
# ``_FEATHER_SCALE`` is the fallback default shared by every pass that does
# not override it. Per-part calibration now lives on the ``PartParams``
# subclasses as ``_FEATHER_SCALE`` ClassVars (e.g. cheek = 0.15, eyelid = 0.10),
# threaded into this function via the ``feather_scale`` argument. Resolution-
# independent by construction.
_FEATHER_SCALE = 0.04

# Outer eye-corner landmark IDs (MediaPipe 468 face mesh). Their Euclidean
# separation is a tilt- and crop-robust facial-scale reference, always present
# in the standard mesh (unlike the refined iris points 468-477).
_OUTER_EYE_CORNER_LEFT = 33
_OUTER_EYE_CORNER_RIGHT = 263


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """Ken Perlin's smootherstep ``6t⁵ − 15t⁴ + 10t³`` on t ∈ [0, 1].

    Zero first and second derivatives at both ends, so the feather ramp meets
    the flat interior and the outer edge without a Mach-band ridge.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def inter_ocular_distance(coords: np.ndarray) -> float:
    """Euclidean outer-eye-corner span in pixels — a robust facial scale.

    Returns 0.0 when the corner landmarks are unavailable (e.g. the small
    synthetic coordinate arrays used in tests), letting callers fall back to an
    image-scale estimate.
    """
    if len(coords) <= _OUTER_EYE_CORNER_RIGHT:
        return 0.0
    a = coords[_OUTER_EYE_CORNER_LEFT].astype(np.float64)
    b = coords[_OUTER_EYE_CORNER_RIGHT].astype(np.float64)
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def feather_px_from_softness(
    softness: float,
    coords: np.ndarray,
    image_shape: tuple[int, int],
    feather_scale: float = _FEATHER_SCALE,
) -> float:
    """Convert a 0..1 softness into an absolute feather width in pixels.

    The width is a fixed fraction of the inter-ocular span, so the same
    softness yields a proportionally identical fade at any resolution or crop.
    When eye-corner landmarks are missing, falls back to a fraction of the
    smaller image dimension as a rough facial scale.

    Args:
        softness: Feather width as a face-size ratio (0.0-1.0); larger is
            softer. ``<= 0`` disables feathering.
        coords: Landmark pixel coordinates (used for the inter-ocular span).
        image_shape: (height, width) — the fallback scale source.
        feather_scale: Multiplier mapping softness → feather fraction of the
            inter-ocular span. Defaults to the module-level ``_FEATHER_SCALE``
            fallback; passes override it with their per-part calibration
            (read off the ``PartParams`` subclass ``_FEATHER_SCALE`` ClassVar).

    Returns:
        Feather band width in pixels (0.0 when softness <= 0).
    """
    if softness <= 0.0:
        return 0.0
    reference = inter_ocular_distance(coords)
    if reference <= 1.0:
        reference = 0.5 * float(min(image_shape[0], image_shape[1]))
    return softness * feather_scale * reference


def soften_mask(
    mask: np.ndarray,
    feather_px: float,
    soften_floor: float = _SOFTEN_FLOOR,
) -> np.ndarray:
    """Feather mask edges over a fixed pixel width, size-independent.

    The interior stays a flat 1.0 plateau; only the outer ``feather_px`` band
    ramps — via smootherstep — from ``soften_floor`` at the boundary up to 1.0.
    Unlike the old ``dist / dist.max()`` attenuation, the ramp width no longer
    scales with the mask's thickness, so a thin eyeshadow band and a broad cheek
    receive the same physical softness and an asymmetric distance field can no
    longer thin one side.

    ``feather_px`` is clamped to the mask's half-thickness (its maximum distance
    transform value), so a band thinner than the feather still reaches full
    opacity along its centreline instead of washing out entirely.

    Args:
        mask: Input mask (uint8 0/255 or float32 0.0-1.0).
        feather_px: Feather band width in pixels. ``<= 0`` returns the mask as a
            hard-edged float32 (no feather).
        soften_floor: Opacity retained at the very mask edge; the ramp rises
            from this floor up to 1.0. Defaults to the module-level
            ``_SOFTEN_FLOOR`` fallback; passes override it with their per-part
            calibration (read off the ``PartParams`` subclass ``_SOFTEN_FLOOR``
            ClassVar).

    Returns:
        Softened mask as float32 (0.0-1.0).
    """
    if mask.dtype == np.uint8:
        soft_mask = mask.astype(np.float32) / 255.0
    else:
        soft_mask = mask.astype(np.float32)

    if feather_px <= 0.0:
        return soft_mask

    binary = (soft_mask > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return soft_mask

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Never feather wider than the shape's half-thickness, or a thin band would
    # never reach its full-opacity centreline.
    effective = min(float(feather_px), float(dist.max()))
    effective = max(effective, 1e-6)

    t = np.clip(dist / effective, 0.0, 1.0)
    ramp = soften_floor + (1.0 - soften_floor) * _smootherstep(t)
    return np.clip(soft_mask * ramp, 0.0, 1.0)


def apply_target_color(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """Apply target color via HSV-aware alpha blend.

    Args:
        rgb: Input image in RGB float32 (0-1).
        mask: Soft mask float32 (0-1).
        target_rgb: Target color as (R, G, B) 0-255.
        opacity: Opacity (0-1).

    Returns:
        RGB float32 (0-1) with color applied.
    """
    target_color = np.array(target_rgb, dtype=np.float32) / 255.0
    target_hsv = cv2.cvtColor(target_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]

    blend_weight = mask[:, :, np.newaxis] * opacity

    # Achromatic fallback: lerp toward the target's own V (not toward black),
    # so a white/gray/black target color reads correctly instead of always
    # darkening.
    if target_hsv[1] < 0.05:
        target_v = float(target_hsv[2])
        gray = np.full(3, target_v, dtype=np.float32)
        return rgb * (1.0 - blend_weight) + gray * blend_weight

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_bool = mask > 0

    h_channel = np.where(mask_bool, target_hsv[0], hsv[:, :, 0])
    s_channel = np.where(mask_bool, target_hsv[1], hsv[:, :, 1])
    v_channel = np.where(mask_bool, hsv[:, :, 2] * 0.8 + target_hsv[2] * 0.2, hsv[:, :, 2])
    result_hsv = np.stack([h_channel, s_channel, v_channel], axis=2).astype(np.float32)

    rgb_target = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
    return rgb * (1 - blend_weight) + rgb_target * blend_weight


def blend_hue_shift(
    image: np.ndarray,
    soft_mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """Hue-shift blend for iris, given an already-softened mask.

    Shifts only the H channel of the iris region to the target hue,
    preserving original S and V, then alpha-blends back. The caller is
    responsible for softening (fading) and excluding the mask beforehand.

    Args:
        image: RGB uint8 image (H, W, 3).
        soft_mask: Ready soft mask float32 (0.0-1.0).
        target_rgb: Target color as (R, G, B) 0-255.
        opacity: Opacity (0.0-1.0).
    """
    target_rgb_f = np.array(target_rgb, dtype=np.float32) / 255.0
    target_hsv = cv2.cvtColor(target_rgb_f.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    target_h = float(target_hsv[0])
    target_s = float(target_hsv[1])

    image_float = image.astype(np.float32) / 255.0
    blend_weight = soft_mask[:, :, np.newaxis] * opacity

    if target_s < 0.05:
        target_v = float(target_hsv[2])
        gray = np.full(3, target_v, dtype=np.float32)
        result_float = image_float * (1.0 - blend_weight) + gray * blend_weight
        return np.clip(result_float * 255.0, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_float, cv2.COLOR_RGB2HSV)
    hsv_shifted = hsv.copy()
    # Short angular path lerp for circular H channel (0-360°).
    # S interpolates linearly (non-circular).
    h_diff = target_h - hsv[:, :, 0]
    h_diff = (h_diff + 180.0) % 360.0 - 180.0  # normalize to [-180, 180)
    hsv_shifted[:, :, 0] = (hsv[:, :, 0] + soft_mask * h_diff) % 360.0
    hsv_shifted[:, :, 1] = soft_mask * target_s + (1.0 - soft_mask) * hsv[:, :, 1]

    rgb_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)
    result_float = rgb_shifted * blend_weight + image_float * (1.0 - blend_weight)

    return np.clip(result_float * 255.0, 0, 255).astype(np.uint8)


def apply_hue_shift_blend(
    image: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
    feather_px: float,
) -> np.ndarray:
    """Soften a hard mask then hue-shift blend (fade + color in one call).

    Convenience wrapper kept for direct callers; the composable pipeline
    softens in the mask stage and calls :func:`blend_hue_shift` directly.
    ``feather_px`` is the feather band width in pixels (see :func:`soften_mask`).
    """
    return blend_hue_shift(
        image, soften_mask(mask, feather_px), target_rgb, opacity
    )


def apply_glitter(
    image: np.ndarray,
    mask: np.ndarray,
    glitter_intensity: float,
    rng: np.random.Generator | None = None,
    base_size: float = 0.5,
    density_scale: float | None = None,
    min_distance_factor: float = 1,
    sparkle_alpha: float = 0.8,
    sparkle_blur_sigma: float = 1.0,
    bloom_threshold: float = 0.3,
    bloom_strength: float | None = None,
    bloom_sigma: float | None = None,
) -> np.ndarray:
    """Apply glitter (lamé) effect over a matte-blended region.

    Uses Poisson-disk particle generation, Gaussian sparkle rendering,
    phase-based twinkle, and phase-based interference-style hue shift for a
    cosmetic-grade glitter look. Both brightness and colour are driven by
    each particle's random static phase rather than 3D view angle — angle
    tracking added significant code for a visually negligible effect.
    Particle colour is a fixed silver-white independent of the eyeshadow
    colour underneath, rather than a lit-up tint of it.

    Args:
        image: RGB uint8 image (H, W, 3) — matte blend already applied.
        mask: Soft mask float32 (0.0-1.0).
        glitter_intensity: 0.0-1.0 — controls particle density and bloom.
        rng: Optional random generator (deterministic with seed for tests).
        base_size: Base particle radius in pixels (default 0.5).
        density_scale: Multiplier on mask_area * intensity for target
            particle count. ``None`` defers to ``generate_particles``'
            default (``_GLITTER_DENSITY_SCALE``, 0.03).
        min_distance_factor: Poisson-disk rejection radius as a fraction
            of the average-cell spacing (default 1).
        sparkle_alpha: Additive blend strength for sparkle overlay (default 0.8).
        sparkle_blur_sigma: Gaussian blur sigma for sparkle softening (default 1.0).
        bloom_threshold: Fraction of peak overlay luminance for bloom trigger
            (default 0.3).
        bloom_strength: Override bloom blend strength. Defaults to 0.3 * intensity.
        bloom_sigma: Gaussian blur sigma for bloom halo. Defaults to
            8.0 + intensity * 16.0.

    Returns:
        RGB uint8 image with glitter overlay applied. Always returns a fresh
        ndarray; the input image is never mutated or returned by reference.
    """
    from face_makeup.core.blend.glitter import (
        apply_bloom,
        compute_iridescent_colors,
        compute_static_twinkle,
        generate_particles,
        render_sparkle_overlay,
    )

    if glitter_intensity <= 0.0:
        return image.copy()

    rng = rng or np.random.default_rng()

    mask_bool = mask > 0.1
    if not mask_bool.any():
        return image.copy()

    # 1. Generate particles
    # density_scale=None defers to generate_particles' own default
    # (_GLITTER_DENSITY_SCALE) — forward it only when the caller set an
    # explicit value, so the default binds inside generate_particles.
    particle_kwargs: dict = {
        "base_size": base_size,
        "min_distance_factor": min_distance_factor,
    }
    if density_scale is not None:
        particle_kwargs["density_scale"] = density_scale
    particles = generate_particles(
        mask, glitter_intensity, rng, **particle_kwargs
    )

    if len(particles.positions) == 0:
        return image.copy()

    # 2. Compute per-particle brightness (random static twinkle)
    brightness = compute_static_twinkle(particles.phases)

    # 3. Compute per-particle iridescent colour (spec §4: 偏光・干渉)
    colors = compute_iridescent_colors(
        particles.positions, particles.phases, particles.base_color,
    )

    # 4. Render sparkles to overlay
    sparkle_overlay = render_sparkle_overlay(
        image.shape[:2], particles, brightness, colors=colors,
        blur_sigma=sparkle_blur_sigma,
    )

    # 5. Composite additively
    result = image.astype(np.float32) + sparkle_overlay * sparkle_alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 6. Bloom post-processing
    result = apply_bloom(
        result, sparkle_overlay, glitter_intensity,
        threshold=bloom_threshold,
        bloom_strength=bloom_strength,
        bloom_sigma=bloom_sigma,
    )

    return result


def blend_matte(
    image: np.ndarray,
    soft_mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
) -> np.ndarray:
    """HSV-aware matte color blend given an already-softened mask.

    The caller is responsible for softening (fading) and excluding the mask
    beforehand — this applies only the colour. Used by the composable pipeline,
    where the fade lives in the mask stage so exclusions can be applied to the
    soft mask before blending.

    Args:
        image: Input RGB uint8 image (H, W, 3).
        soft_mask: Ready soft mask float32 (0.0-1.0).
        target_rgb: Target color as (R, G, B) 0-255.
        opacity: Opacity (0.0-1.0).

    Returns:
        RGB uint8 image with makeup applied.
    """
    image_float = image.astype(np.float32) / 255.0
    rgb_result = apply_target_color(image_float, soft_mask, target_rgb, opacity)
    return np.clip(rgb_result * 255, 0, 255).astype(np.uint8)


def standard_blend(
    image: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
    feather_px: float,
) -> np.ndarray:
    """Soften a hard mask then matte blend (fade + color in one call).

    Convenience wrapper kept for direct callers; the composable pipeline
    softens in the mask stage and calls :func:`blend_matte` directly.

    Args:
        image: Input RGB uint8 image (H, W, 3).
        mask: Mask float32 (0.0-1.0) or uint8 (0/255).
        target_rgb: Target color as (R, G, B) 0-255.
        opacity: Opacity (0.0-1.0).
        feather_px: Feather band width in pixels (see :func:`soften_mask`).

    Returns:
        RGB uint8 image with makeup applied.
    """
    return blend_matte(image, soften_mask(mask, feather_px), target_rgb, opacity)
