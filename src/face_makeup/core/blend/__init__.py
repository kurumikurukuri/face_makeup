"""
Reusable blend / image-compositing primitives.

These pure functions are consumed by the
:class:`~face_makeup.core.pipeline.BlendStrategy` implementations that live in
:mod:`face_makeup.core.passes`. The ``BlendStrategy``
*abstract* class itself lives in :mod:`face_makeup.core.pipeline`, and its *concrete*
implementations (e.g. ``EyeShadowBlendStrategy``) sit alongside their passes in
:mod:`face_makeup.core.passes`.

This package holds only pass-agnostic image operations — mask softening, colour
application, hue-shift blending, the standard HSV-aware blend pipeline, and the
glitter (lamé) particle effect. It imports nothing else from ``face_makeup``
(only ``cv2`` / ``numpy``), so it forms a leaf utility layer that the passes
depend on, never the reverse.
"""
from face_makeup.core.blend.blend_utils import (
    apply_glitter,
    apply_hue_shift_blend,
    apply_target_color,
    blend_hue_shift,
    blend_matte,
    soften_mask,
    standard_blend,
)
from face_makeup.core.blend.glitter import (
    ParticleSet,
    apply_bloom,
    compute_iridescent_colors,
    compute_static_twinkle,
    generate_particles,
    render_sparkle_overlay,
)

__all__ = [
    "apply_bloom",
    "apply_glitter",
    "apply_hue_shift_blend",
    "apply_target_color",
    "blend_hue_shift",
    "blend_matte",
    "compute_iridescent_colors",
    "compute_static_twinkle",
    "generate_particles",
    "ParticleSet",
    "render_sparkle_overlay",
    "soften_mask",
    "standard_blend",
]
