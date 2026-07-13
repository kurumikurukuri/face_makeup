"""
Regression tests for the exclusion-after-fade pipeline contract.

Root cause fixed here: hair (and pass-to-pass) exclusions used to be subtracted
from the *hard* mask in Phase 1, before the distance-based fade. A hole punched
into the mask corrupted the distance transform and collapsed the fade on one
side (the cheek "crescent / missing chunk").

The contract now is: mask strategies return the FINAL soft mask (fade baked in
on the solid geometry), and the renderer gates it by a feathered exclusion
afterwards — ``final = soft * (1 - exclusion)``. These tests lock that:

  1. a non-excluded pixel keeps the mask strategy's faded value unchanged
     (an exclusion hole can never deform the fade elsewhere), and
  2. the exclusion boundary is feathered (a soft ramp, not a 0/1 step).
"""
from __future__ import annotations

import cv2
import numpy as np

from face_makeup.core.blend.blend_utils import soften_mask
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.types import ColorParams, PartParams
from face_makeup.renderer import Renderer, _feather_exclusion


class _SoftDiskMaskStrategy(MaskStrategy):
    """Returns a fixed, already-softened disk mask (float32 0-1)."""

    def __init__(self, soft_mask: np.ndarray):
        self._soft_mask = soft_mask

    def generate(self, coords, image_shape, image=None, landmarks=None, segmenter=None):
        return self._soft_mask


class _MaskProbeBlend(BlendStrategy):
    """Encodes the final mask handed to the blend into channel 2 (×200)."""

    def apply(self, image, mask, params, blender=None, segmenter=None, landmarks=None):
        result = image.copy()
        result[:, :, 2] = np.clip(mask * 200.0, 0, 255).astype(np.uint8)
        return result


class _FakeHairSegmenter:
    def __init__(self, mask):
        self._mask = mask

    def segment(self, image):
        return self._mask.copy()


def _make_face(passes):
    from face_makeup.face import Face
    from face_makeup.core.types import FaceBBox, FaceLandmarks

    face = Face(0, FaceBBox(0, 0, 64, 64, 1.0),
                FaceLandmarks(np.zeros((478, 2), dtype=np.float32)))
    for p in passes:
        face._makeup_passes.append(p)
    return face


def _probe_pass(soft_mask):
    return MakeupPass(
        pass_type="probe",
        priority=0,
        params=PartParams(color_params=ColorParams(target_rgb=(0, 0, 0)), opacity=1.0),
        landmark_indices=[],
        mask_strategy=_SoftDiskMaskStrategy(soft_mask),
        blend_strategy=_MaskProbeBlend(),
    )


def _soft_disk(size=64, radius=22):
    hard = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(hard, (size // 2, size // 2), radius, 255, -1)
    return soften_mask(hard, 8.0)   # 8 px feather band on a ~22 px radius disk


def _render_probe(soft_mask, hair_mask=None):
    seg = _FakeHairSegmenter(hair_mask) if hair_mask is not None else None
    renderer = Renderer(hair_segmenter=seg)
    face = _make_face([_probe_pass(soft_mask)])
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    return renderer.render_face(image, face)[:, :, 2].astype(np.int32)


def test_exclusion_hole_does_not_deform_fade_elsewhere():
    """A hole in the exclusion source must not change the fade away from it.

    This is the crescent regression: with pre-fade subtraction, a central hole
    warped the distance field and darkened one whole side. With exclusion after
    fade, pixels outside the hair patch are byte-for-byte identical to the
    no-hair render.
    """
    soft = _soft_disk()

    hair = np.zeros((64, 64), dtype=np.uint8)
    hair[28:36, 28:36] = 255  # a hole near the disk centre (the medial axis)

    baseline = _render_probe(soft)             # no exclusion
    with_hair = _render_probe(soft, hair)      # central hole excluded

    # Everywhere the exclusion is exactly zero (beyond the feather kernel), the
    # fade must be byte-for-byte identical to the no-hair render.
    hair_soft = _feather_exclusion(hair)
    untouched = hair_soft == 0.0
    assert np.array_equal(baseline[untouched], with_hair[untouched])

    # And the disk edge fade is genuinely graded (not a binary disk), proving
    # the probe is observing the softened mask.
    assert 0 < baseline.max()
    assert np.any((baseline > 0) & (baseline < baseline.max()))


def test_exclusion_boundary_is_feathered():
    """The exclusion gate ramps continuously — no hard 0/1 cut at the edge."""
    soft = np.full((64, 64), 1.0, dtype=np.float32)  # flat, fully-covering mask

    hair = np.zeros((64, 64), dtype=np.uint8)
    hair[20:44, 20:44] = 255

    probed = _render_probe(soft, hair) / 200.0  # back to ~mask scale

    # Deep interior fully excluded, far exterior untouched…
    assert probed[32, 32] < 0.05
    assert probed[2, 2] > 0.95
    # …and at least one boundary pixel sits strictly between the two (feather).
    assert np.any((probed > 0.1) & (probed < 0.9))
