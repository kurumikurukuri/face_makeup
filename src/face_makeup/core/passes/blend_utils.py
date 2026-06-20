"""
Shared blend utilities for MakeupPass BlendStrategy implementations.

Provides mask softening, target color application, hue-shift blending,
and the standard HSV-aware blend pipeline used by all makeup passes.
"""
from __future__ import annotations

import cv2
import numpy as np


def soften_mask(mask: np.ndarray, blur_strength: float) -> np.ndarray:
    """Soften mask edges using distance-based attenuation.

    Args:
        mask: Input mask (uint8 0/255 or float32 0.0-1.0).
        blur_strength: Edge attenuation factor (0.0-1.0).
            Boundary pixels retain this fraction of opacity,
            center pixels stay at 1.0.

    Returns:
        Softened mask as float32 (0.0-1.0).
    """
    if mask.dtype == np.uint8:
        soft_mask = mask.astype(np.float32) / 255.0
    else:
        soft_mask = mask.astype(np.float32)

    if blur_strength <= 0:
        return soft_mask

    binary = (soft_mask > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return soft_mask

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_dist = dist.max()
    dist_norm = dist / max_dist if max_dist > 0 else dist
    attenuation = blur_strength + (1.0 - blur_strength) * dist_norm

    return soft_mask * attenuation


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

    # Achromatic fallback
    if target_hsv[1] < 0.05:
        darken_factor = np.clip(1.0 - blend_weight, 0.0, 1.0)
        return rgb * darken_factor

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_bool = mask > 0

    h_channel = np.where(mask_bool, target_hsv[0], hsv[:, :, 0])
    s_channel = np.where(mask_bool, target_hsv[1], hsv[:, :, 1])
    v_channel = np.where(mask_bool, hsv[:, :, 2] * 0.8 + target_hsv[2] * 0.2, hsv[:, :, 2])
    result_hsv = np.stack([h_channel, s_channel, v_channel], axis=2).astype(np.float32)

    rgb_target = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
    return rgb * (1 - blend_weight) + rgb_target * blend_weight


def apply_hue_shift_blend(
    image: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
    blur_strength: float,
) -> np.ndarray:
    """Hue-shift blend for iris.

    Shifts only the H channel of the iris region to the target hue,
    preserving original S and V, then alpha-blends back.
    """
    soft_mask = soften_mask(mask, blur_strength)

    target_rgb_f = np.array(target_rgb, dtype=np.float32) / 255.0
    target_hsv = cv2.cvtColor(target_rgb_f.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    target_h = float(target_hsv[0])
    target_s = float(target_hsv[1])

    image_float = image.astype(np.float32) / 255.0
    blend_weight = soft_mask[:, :, np.newaxis] * opacity

    if target_s < 0.05:
        darken_factor = np.clip(1.0 - blend_weight, 0.0, 1.0)
        return np.clip(image_float * darken_factor * 255.0, 0, 255).astype(np.uint8)

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


def standard_blend(
    image: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    opacity: float,
    blur_strength: float,
) -> np.ndarray:
    """Standard HSV-aware blend: soften_mask + apply_target_color.

    Args:
        image: Input BGR uint8 image (H, W, 3).
        mask: Mask float32 (0.0-1.0) or uint8 (0/255).
        target_rgb: Target color as (R, G, B) 0-255.
        opacity: Opacity (0.0-1.0).
        blur_strength: Edge attenuation factor (0.0-1.0).

    Returns:
        BGR uint8 image with makeup applied.
    """
    soft_mask = soften_mask(mask, blur_strength)
    image_float = image.astype(np.float32) / 255.0
    rgb_result = apply_target_color(image_float, soft_mask, target_rgb, opacity)
    return np.clip(rgb_result * 255, 0, 255).astype(np.uint8)
