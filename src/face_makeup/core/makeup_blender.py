"""
Makeup blending operations.
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from .types import ColorParams, LipsParams, CheekParams, EyeShadowParams, EyeLinerParams, EyeIrisParams


class MakeupBlender:
    """Makeup blending processor."""

    def _soften_mask(
        self,
        mask: np.ndarray,
        blur_strength: float,
    ) -> np.ndarray:
        """
        Soften mask edges using distance-based attenuation.

        Args:
            mask: Input mask (uint8 0/255 or float32 0.0-1.0).
            blur_strength: Edge attenuation factor (0.0-1.0).
                Boundary pixels retain this fraction of opacity,
                center pixels stay at 1.0. Higher values produce
                less falloff at edges.

        Returns:
            Softened mask as float32 (0.0-1.0).
        """

        # Convert to float32
        if mask.dtype == np.uint8:
            soft_mask = mask.astype(np.float32) / 255.0
        else:
            soft_mask = mask.astype(np.float32)

        # Early exit
        if blur_strength <= 0:
            return soft_mask

        # Binary mask for distance transform
        binary = (soft_mask > 0.5).astype(np.uint8)

        if binary.max() == 0:
            return soft_mask

        # Distance from each pixel to nearest boundary (inside mask)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Normalize distance: center -> 1, boundary -> 0
        max_dist = dist.max()
        if max_dist > 0:
            dist_norm = dist / max_dist
        else:
            dist_norm = dist

        # Attenuation: center=1, boundary=blur_strength
        attenuation = blur_strength + (1.0 - blur_strength) * dist_norm

        # Apply attenuation only inside mask
        softened = soft_mask * attenuation
        return softened


    def blend(self,
              image: np.ndarray,
              mask: np.ndarray,
              color_params: ColorParams,
              opacity: float,
              blur_strength: float = 0.0) -> np.ndarray:
        """
        Apply makeup color to an image.

        Args:
            image: Input image (uint8, H, W, 3).
            mask: Mask (float32 0.0-1.0 or uint8 0/255).
            color_params: Color parameters.
            opacity: Opacity (0.0-1.0).
            blur_strength: Blur strength for mask softening (0.0-1.0).

        Returns:
            Image with makeup applied.
        """
        # Input validation
        if image.dtype != np.uint8:
            raise ValueError(f"image must be uint8, got {image.dtype}")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3), got {image.shape}")
        if len(mask.shape) != 2:
            raise ValueError(f"mask must be (H, W), got {mask.shape}")
        if image.shape[:2] != mask.shape:
            raise ValueError(f"image shape {image.shape[:2]} != mask shape {mask.shape}")

        # Apply blur to soften mask
        soft_mask = self._soften_mask(mask, blur_strength)

        # Validate mask dtype after softening
        if soft_mask.dtype != np.float32:
            raise ValueError(f"mask must be float32, got {soft_mask.dtype}")

        # Apply color (returns RGB float32 0-1)
        image_float = image.astype(np.float32) / 255.0
        rgb_result = self._apply_target_color(
            image_float, soft_mask, color_params.target_rgb, opacity
        )

        # Clip to uint8 and return
        result = np.clip(rgb_result * 255, 0, 255).astype(np.uint8)
        return result

    def _apply_target_color(self,
                            rgb: np.ndarray,
                            mask: np.ndarray,
                            target_rgb: tuple,
                            opacity: float) -> np.ndarray:
        """
        Apply the color specified by target_rgb.
        Black/achromatic colors are treated as brightness operations
        to avoid blending with the mask.

        Args:
            rgb: Input image in RGB float32 (0-1).
            mask: Soft mask float32 (0-1).
            target_rgb: Target color as (R, G, B) 0-255.
            opacity: Opacity (0-1).

        Returns:
            RGB float32 (0-1) with color applied.
        """
        target_color = np.array(target_rgb, dtype=np.float32) / 255.0
        target_hsv = cv2.cvtColor(
            target_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV
        )[0, 0]

        # Blend weight
        blend_weight = mask[:, :, np.newaxis] * opacity

        # === Black/achromatic color detection ===
        is_neutral = target_hsv[1] < 0.05  # Saturation is nearly 0

        if is_neutral:
            # Scaling all RGB channels by the same factor is mathematically
            # equivalent to reducing V in HSV while preserving H and S.
            darken_factor = np.clip(1.0 - blend_weight, 0.0, 1.0)
            return rgb * darken_factor

        # === Normal colored makeup ===
        # Convert to HSV and apply target H/S only to masked pixels
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask_bool = mask > 0

        h_channel = np.where(mask_bool, target_hsv[0], hsv[:, :, 0])
        s_channel = np.where(mask_bool, target_hsv[1], hsv[:, :, 1])
        v_channel = np.where(mask_bool, hsv[:, :, 2] * 0.8 + target_hsv[2] * 0.2, hsv[:, :, 2])
        result_hsv = np.stack([h_channel, s_channel, v_channel], axis=2).astype(np.float32)

        rgb_target = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)

        return rgb * (1 - blend_weight) + rgb_target * blend_weight

    def blend_lips(self,
                   image: np.ndarray,
                   mask: np.ndarray,
                   params: LipsParams) -> np.ndarray:
        """
        Apply makeup to lips.
        """
        return self.blend(image, mask, params.color_params, params.opacity, params.blur_strength)

    def blend_cheek(self,
                    image: np.ndarray,
                    mask: np.ndarray,
                    cheek_mask: np.ndarray,
                    params: CheekParams) -> np.ndarray:
        """
        Apply makeup to cheek (using pre-computed ellipse mask).

        Args:
            image: Input image.
            mask: Base cheek mask from segmenter.
            cheek_mask: Ellipse cheek mask from FaceSegmenter.create_cheek_mask().
            params: Cheek parameters.
        """
        combined_mask = mask * cheek_mask
        return self.blend(image, combined_mask, params.color_params, params.opacity, params.blur_strength)

    def blend_eyeshadow(self,
                        image: np.ndarray,
                        mask: np.ndarray,
                        params: EyeShadowParams) -> np.ndarray:
        """
        Apply eyeshadow makeup.

        Future: gradient functionality planned.
        Currently only normal blend processing.
        """
        return self.blend(image, mask, params.color_params, params.opacity, params.blur_strength)

    def blend_eyeliner(self,
                        image: np.ndarray,
                        mask: np.ndarray,
                        params: EyeLinerParams) -> np.ndarray:
        """
        Apply eyeliner makeup.

        Args:
            image: Input image.
            mask: Eyeliner mask.
            params: Eyeliner parameters (thickness adjusts line width).
        """
        # Dilate mask according to thickness
        effective_mask = mask
        if params.thickness > 1:
            kernel = np.ones((params.thickness, params.thickness), np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            effective_mask = mask_dilated.astype(np.float32) / 255.0

        return self.blend(image, effective_mask, params.color_params, params.opacity, params.blur_strength)

    def _blend_eyeiris_hue_shift(self,
                                 image: np.ndarray,
                                 mask: np.ndarray,
                                 params: EyeIrisParams) -> np.ndarray:
        """
        Hue-shift blend for iris : cut → shift → paste).

        Extracts the iris region, shifts its H channel to the target hue
        while keeping original S and V, then alpha-blends back.

        Args:
            image: Input RGB uint8 image.
            mask: Iris mask (float32 0-1 or uint8 0/255).
            params: Eye iris parameters.

        Returns:
            RGB uint8 image with hue-shifted iris blended in.
        """
        soft_mask = self._soften_mask(mask, params.blur_strength)

        # Target color in HSV (float32 image → H 0-360, S/V 0-1)
        target_rgb_f = np.array(params.color_params.target_rgb, dtype=np.float32) / 255.0
        target_hsv = cv2.cvtColor(target_rgb_f.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
        target_h = float(target_hsv[0])
        target_s = float(target_hsv[1])

        image_float = image.astype(np.float32) / 255.0
        blend_weight = soft_mask[:, :, np.newaxis] * params.opacity

        # Achromatic fallback: mirrors legacy is_neutral branch to avoid
        # spurious red tint when target color is black/grey/white (S≈0)
        if target_s < 0.05:
            darken_factor = np.clip(1.0 - blend_weight, 0.0, 1.0)
            return np.clip(image_float * darken_factor * 255.0, 0, 255).astype(np.uint8)

        # Convert image to float32 HSV
        hsv = cv2.cvtColor(image_float, cv2.COLOR_RGB2HSV)

        # Interpolate H toward target using short angular path (circular lerp).
        # S interpolates linearly (non-circular), V stays at original value.
        # This avoids a hard hue boundary at mask edges (vs binary mask_bool > 0).
        hsv_shifted = hsv.copy()
        h_diff = target_h - hsv[:, :, 0]
        h_diff = (h_diff + 180.0) % 360.0 - 180.0  # normalize to [-180, 180)
        hsv_shifted[:, :, 0] = (hsv[:, :, 0] + soft_mask * h_diff) % 360.0
        hsv_shifted[:, :, 1] = soft_mask * target_s + (1.0 - soft_mask) * hsv[:, :, 1]
        # V stays at original value to preserve luminance texture

        # Convert back to RGB
        rgb_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)

        # Alpha blend: shifted region composited over original
        result_float = rgb_shifted * blend_weight + image_float * (1.0 - blend_weight)

        return np.clip(result_float * 255.0, 0, 255).astype(np.uint8)

    def blend_eyeiris(self,
                      image: np.ndarray,
                      eyeiris_mask: np.ndarray,
                      eye_mask: np.ndarray,
                      params: EyeIrisParams) -> np.ndarray:
        """
        Apply colored contact lens effect to iris area.

        The iris mask from fitEllipse on iris landmarks always forms a full
        ellipse, which overlaps the eyelid skin when the eye is half-closed
        or blinking. eye_mask (the eye-opening polygon) clips the ellipse
        back down to the actual eye region so color never bleeds onto skin.

        Dispatches to hue-shift blend when params.hue_shift is True,
        otherwise uses legacy HSV replace blend.

        Args:
            image: Input image.
            eyeiris_mask: Iris ellipse mask (H, W) float32 or uint8 0/255.
            eye_mask: Eye region mask to constrain iris (H, W) float32 or uint8 0/255.
            params: Eye iris parameters.

        Raises:
            ValueError: If eye_mask is None.
        """
        if eye_mask is None:
            raise ValueError("eye_mask is required for iris makeup")
        combined_mask = self._create_eyeiris_mask(eyeiris_mask, eye_mask)
        if params.hue_shift:
            return self._blend_eyeiris_hue_shift(image, combined_mask, params)
        return self.blend(image, combined_mask, params.color_params, params.opacity, params.blur_strength)

    def _create_eyeiris_mask(self,
                             eyeiris_mask: np.ndarray,
                             eye_mask: np.ndarray) -> np.ndarray:
        """
        Clip the iris ellipse mask to the eye-opening region.

        Args:
            eyeiris_mask: Iris mask (H, W) uint8 0/255 or float32.
            eye_mask: Eye region mask (H, W) uint8 0/255 or float32.

        Returns:
            Combined mask with iris limited to eye region (H, W) float32.
        """
        eyeiris_soft = eyeiris_mask.astype(np.float32) / 255.0 if eyeiris_mask.dtype == np.uint8 else eyeiris_mask
        eye_soft = eye_mask.astype(np.float32) / 255.0 if eye_mask.dtype == np.uint8 else eye_mask
        return eyeiris_soft * eye_soft


