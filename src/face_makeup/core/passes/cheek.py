"""
Cheek makeup pass — convex hull mask + 3D-projected ellipse blend.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from face_makeup.constants import CHEEK_LEFT_INDICES, CHEEK_RIGHT_INDICES
from face_makeup.core.pipeline import BlendStrategy, MakeupPass, MaskStrategy
from face_makeup.core.blend.blend_utils import blend_matte, soften_mask
from face_makeup.core.types import CheekParams, FaceLandmarks


_INDICES_MAP = {
    "cheek_left": CHEEK_LEFT_INDICES,
    "cheek_right": CHEEK_RIGHT_INDICES,
}


class CheekMaskStrategy(MaskStrategy):
    """Soft cheek mask: convex hull ∩ 3D-projected ellipse, then edge-softened.

    Owns the full geometry (convex hull + tangent-plane ellipse) and returns
    the final soft mask (float32 0-1). Keeping geometry + fade here — and out of
    the blend — lets the pipeline apply exclusions to the softened mask before
    blending, so hair/pass exclusions never punch holes into the distance field.
    """

    def __init__(self, indices: list[int], cheek_side: str, params: CheekParams):
        self._indices = indices
        self._cheek_side = cheek_side
        self._params = params
        self._blur_strength = params.blur_strength

    def generate(
        self,
        coords: np.ndarray,
        image_shape: tuple[int, int],
        image: np.ndarray | None = None,
        landmarks: object | None = None,
        segmenter: object | None = None,
    ) -> np.ndarray:
        h, w = image_shape
        base = np.zeros((h, w), dtype=np.uint8)

        points = coords[self._indices]
        if len(points) < 3:
            return soften_mask(base, self._blur_strength)

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(base, hull, 255)

        ellipse = self.create_cheek_mask(
            image_shape, base, self._params,
            landmarks=landmarks, cheek_side=self._cheek_side,
        )
        combined = (base.astype(np.float32) / 255.0) * ellipse.astype(np.float32)
        return soften_mask(combined, self._blur_strength)

    # ------------------------------------------------------------------
    # 3D cheek ellipse projection (extracted from legacy FaceSegmenter)
    # ------------------------------------------------------------------

    def create_cheek_mask(
        self,
        image_shape: Tuple[int, int],
        base_mask: np.ndarray,
        params: CheekParams,
        landmarks: Optional[FaceLandmarks] = None,
        cheek_side: str = 'left',
        return_debug_info: bool = False,
    ) -> np.ndarray:
        """
        Generate hard-edge ellipse mask for cheek.

        Fadeout effect is achieved via blur_strength in blend().

        Args:
            image_shape: Image size (h, w).
            base_mask: Base mask.
            params: Cheek parameters.
            landmarks: Face landmarks (including 3D coordinates).
            cheek_side: 'left' or 'right'.
            return_debug_info: Whether to return debug info tuple.

        Returns:
            Ellipse mask (h, w) float32 with hard edges (0.0 or 1.0),
            or (mask, debug_info) tuple if return_debug_info is True.
        """
        h, w = image_shape

        mask_binary = (base_mask > 0.1).astype(np.uint8)
        if np.sum(mask_binary) == 0:
            return np.zeros((h, w), dtype=np.float32)

        # 3D manifold-based processing (when landmarks are provided)
        if landmarks is not None and landmarks.points_3d is not None:
            return self._create_cheek_mask_3d(
                image_shape, base_mask, params, landmarks, cheek_side, return_debug_info
            )

        # Legacy 2D processing (backward compatibility)
        coords = np.column_stack(np.where(mask_binary > 0))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        mask_h = y_max - y_min
        mask_w = x_max - x_min

        cx = x_min + params.center[0] * mask_w
        cy = y_min + params.center[1] * mask_h
        rx = params.radius[0] * mask_w
        ry = params.radius[1] * mask_h

        y_grid, x_grid = np.ogrid[:h, :w]
        distance = (
            ((x_grid - cx) / (rx + 1e-6)) ** 2 +
            ((y_grid - cy) / (ry + 1e-6)) ** 2
        )

        # Hard-edge ellipse: inside = 1.0, outside = 0.0
        cheek_mask = (distance <= 1.0).astype(np.float32)

        if return_debug_info:
            debug_info = {
                'mode': '2d',
                'cx': cx,
                'cy': cy,
                'rx': rx,
                'ry': ry,
            }
            return cheek_mask, debug_info

        return cheek_mask

    def _create_cheek_mask_3d(
        self,
        image_shape: Tuple[int, int],
        base_mask: np.ndarray,
        params: CheekParams,
        landmarks: FaceLandmarks,
        cheek_side: str,
        return_debug_info: bool = False,
    ) -> np.ndarray:
        """
        Draw ellipse on cheek surface using tangent plane coordinate system.

        Coordinate Systems:
            - Face 3D Space: MediaPipe normalized 3D coordinates (0-1)
            - Tangent Plane Space (UVW): Local coordinate system on cheek surface
            - Image Space: Pixel coordinates

        Args:
            image_shape: (height, width).
            base_mask: Base mask.
            params: Cheek parameters.
            landmarks: Face landmarks (3D coordinates required).
            cheek_side: 'left' or 'right'.
            return_debug_info: Whether to return debug info.

        Returns:
            Ellipse mask (h, w) float32 (0..1),
            or (mask, debug_info) tuple if return_debug_info is True.
        """
        from face_makeup.constants import CHEEK_LEFT_INDICES as L_INDICES
        from face_makeup.constants import CHEEK_RIGHT_INDICES as R_INDICES

        img_height, img_width = image_shape

        if img_height <= 0 or img_width <= 0:
            raise ValueError(
                f"Invalid image_shape: {image_shape}. Both dimensions must be positive."
            )

        face3d_all_points = landmarks.points_3d
        if face3d_all_points is None:
            return np.zeros((img_height, img_width), dtype=np.float32)

        cheek_indices = L_INDICES if cheek_side == 'left' else R_INDICES
        cheek_face3d = face3d_all_points[cheek_indices]
        cheek_img_norm = landmarks.points[cheek_indices]

        # --- Step 1-5: Build tangent plane and fit surface ---
        tangent_origin, basis_u, basis_v, basis_w, cheek_uvw, surface_coeffs = \
            self._build_cheek_tangent_plane(face3d_all_points, cheek_face3d, cheek_side)

        # --- Steps 6-9: Project ellipse to image pixel space ---
        center_px_x, center_px_y, polygon_px, ellipse_face3d_points = \
            self._project_ellipse_to_image(
                tangent_origin, basis_u, basis_v, basis_w,
                cheek_uvw, surface_coeffs, params, image_shape
            )

        # --- Steps 10-11: Rasterize to float32 mask ---
        result_mask = self._rasterize_ellipse_mask(
            center_px_x, center_px_y, polygon_px, image_shape
        )

        if return_debug_info:
            tangent_u_coords = cheek_uvw[:, 0]
            tangent_v_coords = cheek_uvw[:, 1]
            tangent_u_min = float(tangent_u_coords.min())
            tangent_u_max = float(tangent_u_coords.max())
            tangent_v_min = float(tangent_v_coords.min())
            tangent_v_max = float(tangent_v_coords.max())
            tangent_u_center = (tangent_u_max + tangent_u_min) / 2
            tangent_v_center = (tangent_v_max + tangent_v_min) / 2
            tangent_u_halfspan = (tangent_u_max - tangent_u_min) / 2
            tangent_v_halfspan = (tangent_v_max - tangent_v_min) / 2

            ellipse_radii = np.sqrt(
                (polygon_px[:, 0] - center_px_x) ** 2 +
                (polygon_px[:, 1] - center_px_y) ** 2
            )
            debug_info = {
                'tangent_origin_face3d': tangent_origin,
                'tangent_basis_u': basis_u,
                'tangent_basis_v': basis_v,
                'tangent_basis_w': basis_w,
                'ellipse_img_px_center_x': center_px_x,
                'ellipse_img_px_center_y': center_px_y,
                'ellipse_img_px_radius_x': float(np.mean(ellipse_radii)),
                'ellipse_img_px_radius_y': float(np.mean(ellipse_radii)),
                'img_u_axis_unit': np.array([1.0, 0.0]),
                'img_v_axis_unit': np.array([0.0, 1.0]),
                'tangent_u_min': tangent_u_min,
                'tangent_u_max': tangent_u_max,
                'tangent_v_min': tangent_v_min,
                'tangent_v_max': tangent_v_max,
                'tangent_u_center': tangent_u_center,
                'tangent_v_center': tangent_v_center,
                'tangent_u_halfspan': tangent_u_halfspan,
                'tangent_v_halfspan': tangent_v_halfspan,
                'ellipse_tangent_center_u': (
                    tangent_u_center + params.center[0] * tangent_u_halfspan
                ),
                'ellipse_tangent_center_v': (
                    tangent_v_center + params.center[1] * tangent_v_halfspan
                ),
                'ellipse_tangent_radius_u': abs(params.radius[0]) * tangent_u_halfspan,
                'ellipse_tangent_radius_v': abs(params.radius[1]) * tangent_v_halfspan,
                'surface_coeffs': surface_coeffs,
                'ellipse_face3d_points': ellipse_face3d_points,
                'cheek_tangent_uvw': cheek_uvw,
                'cheek_img_norm': cheek_img_norm,
            }
            return result_mask, debug_info

        return result_mask

    @staticmethod
    def _build_cheek_tangent_plane(
        face3d_all_points: np.ndarray,
        cheek_face3d: np.ndarray,
        cheek_side: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build UVW tangent-plane basis and fit a quadratic surface to cheek landmarks.

        Steps 1-5 of the 3D cheek mask algorithm:
          1. Look up axis-defining landmark indices for the requested side.
          2. Compute Gram-Schmidt orthonormal basis (U, V, W).
          3. Project cheek landmarks into the tangent-plane (UVW) coordinates.
          4. Fit a quadratic surface w = f(u, v) via least squares.

        Args:
            face3d_all_points: All 478 face landmarks in normalized 3D space.
            cheek_face3d: Subset of cheek landmark points (N, 3).
            cheek_side: 'left' or 'right'.

        Returns:
            Tuple of (origin, basis_u, basis_v, basis_w, cheek_uvw, surface_coeffs).
        """
        if cheek_side == 'left':
            u_start_idx, u_end_idx = 50, 36
            v_start_idx, v_end_idx = 118, 205
        else:
            u_start_idx, u_end_idx = 280, 266
            v_start_idx, v_end_idx = 347, 425

        # Build orthonormal UVW basis
        origin = cheek_face3d.mean(axis=0)

        u_raw = face3d_all_points[u_end_idx] - face3d_all_points[u_start_idx]
        basis_u = u_raw / (np.linalg.norm(u_raw) + 1e-6)

        v_raw = face3d_all_points[v_end_idx] - face3d_all_points[v_start_idx]
        v_raw = v_raw - np.dot(v_raw, basis_u) * basis_u  # Gram-Schmidt
        basis_v = v_raw / (np.linalg.norm(v_raw) + 1e-6)

        basis_w = np.cross(basis_u, basis_v)
        basis_w = basis_w / (np.linalg.norm(basis_w) + 1e-6)

        # Project cheek landmarks to tangent-plane coordinates
        vecs = cheek_face3d - origin  # (N, 3)
        cheek_uvw = np.column_stack([
            vecs @ basis_u,
            vecs @ basis_v,
            vecs @ basis_w,
        ])

        # Quadratic surface fit: w = c0 + c1*u + c2*v + c3*u^2 + c4*v^2 + c5*u*v
        u_c = cheek_uvw[:, 0]
        v_c = cheek_uvw[:, 1]
        w_c = cheek_uvw[:, 2]
        design = np.column_stack([
            np.ones_like(u_c), u_c, v_c, u_c ** 2, v_c ** 2, u_c * v_c
        ])
        try:
            surface_coeffs, _, _, _ = np.linalg.lstsq(design, w_c, rcond=None)
        except np.linalg.LinAlgError:
            surface_coeffs = np.zeros(6)
            surface_coeffs[0] = w_c.mean()

        return origin, basis_u, basis_v, basis_w, cheek_uvw, surface_coeffs

    @staticmethod
    def _project_ellipse_to_image(
        tangent_origin: np.ndarray,
        basis_u: np.ndarray,
        basis_v: np.ndarray,
        basis_w: np.ndarray,
        cheek_uvw: np.ndarray,
        surface_coeffs: np.ndarray,
        params: CheekParams,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute ellipse parameters in tangent space and project to image pixels.

        Steps 6-9 of the 3D cheek mask algorithm.
        """
        img_height, img_width = image_shape

        def _eval_w(u, v):
            c = surface_coeffs
            return (
                c[0] + c[1] * u + c[2] * v
                + c[3] * u ** 2 + c[4] * v ** 2 + c[5] * u * v
            )

        u_c = cheek_uvw[:, 0]
        v_c = cheek_uvw[:, 1]
        u_min = u_c.min() if len(u_c) > 0 else -0.1
        u_max = u_c.max() if len(u_c) > 0 else 0.1
        v_min = v_c.min() if len(v_c) > 0 else -0.1
        v_max = v_c.max() if len(v_c) > 0 else 0.1

        u_center = (u_max + u_min) / 2
        v_center = (v_max + v_min) / 2
        u_halfspan = (u_max - u_min) / 2
        v_halfspan = (v_max - v_min) / 2

        ell_cu = u_center + params.center[0] * u_halfspan
        ell_cv = v_center + params.center[1] * v_halfspan
        ell_ru = abs(params.radius[0]) * u_halfspan
        ell_rv = abs(params.radius[1]) * v_halfspan

        theta = np.linspace(0, 2 * np.pi, 50)
        ell_us = ell_cu + ell_ru * np.cos(theta)
        ell_vs = ell_cv + ell_rv * np.sin(theta)

        # Project ellipse boundary: tangent UVW -> Face 3D -> image pixels
        face3d_pts = np.array([
            tangent_origin + u * basis_u + v * basis_v + _eval_w(u, v) * basis_w
            for u, v in zip(ell_us, ell_vs)
        ])
        polygon_px = np.column_stack([
            face3d_pts[:, 0] * img_width,
            face3d_pts[:, 1] * img_height,
        ])

        # Project ellipse center
        center_w = _eval_w(ell_cu, ell_cv)
        center_face3d = (
            tangent_origin
            + ell_cu * basis_u
            + ell_cv * basis_v
            + center_w * basis_w
        )
        center_px_x = float(center_face3d[0] * img_width)
        center_px_y = float(center_face3d[1] * img_height)

        return center_px_x, center_px_y, polygon_px, face3d_pts

    @staticmethod
    def _rasterize_ellipse_mask(
        center_px_x: float,
        center_px_y: float,
        polygon_px: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Rasterize a polygon to a float32 hard-edge mask using distance-based fill.

        Steps 10-11 of the 3D cheek mask algorithm.
        """
        img_height, img_width = image_shape

        bbox_x_min = max(0, int(polygon_px[:, 0].min()) - 10)
        bbox_x_max = min(img_width, int(polygon_px[:, 0].max()) + 10)
        bbox_y_min = max(0, int(polygon_px[:, 1].min()) - 10)
        bbox_y_max = min(img_height, int(polygon_px[:, 1].max()) + 10)

        bbox_h = bbox_y_max - bbox_y_min
        bbox_w = bbox_x_max - bbox_x_min

        # Rasterize polygon interior
        poly_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        local_poly = np.round(
            polygon_px - np.array([[bbox_x_min, bbox_y_min]])
        ).astype(np.int32)
        cv2.fillPoly(poly_mask, [local_poly], 255)
        pixels_inside = poly_mask > 0

        # Pixel grid within bbox
        bbox_ys, bbox_xs = np.meshgrid(
            np.arange(bbox_y_min, bbox_y_max),
            np.arange(bbox_x_min, bbox_x_max),
            indexing="ij"
        )

        # Normalized distance from center (0 = center, 1 = boundary)
        dx = bbox_xs - center_px_x
        dy = bbox_ys - center_px_y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        ellipse_angles = np.arctan2(
            polygon_px[:, 1] - center_px_y,
            polygon_px[:, 0] - center_px_x,
        )
        ellipse_radii = np.sqrt(
            (polygon_px[:, 0] - center_px_x) ** 2 +
            (polygon_px[:, 1] - center_px_y) ** 2
        )
        sort_idx = np.argsort(ellipse_angles)
        sorted_angles = ellipse_angles[sort_idx]
        sorted_radii = ellipse_radii[sort_idx]
        ext_angles = np.concatenate([
            sorted_angles - 2 * np.pi, sorted_angles, sorted_angles + 2 * np.pi
        ])
        ext_radii = np.concatenate([sorted_radii, sorted_radii, sorted_radii])

        boundary_r = np.interp(
            np.arctan2(dy, dx).flatten(), ext_angles, ext_radii
        ).reshape(dy.shape)

        norm_dist = dist / (boundary_r + 1e-6)

        hard_mask = np.zeros_like(norm_dist, dtype=np.float32)
        hard_mask[pixels_inside & (norm_dist <= 1.0)] = 1.0

        cheek_mask = np.zeros((img_height, img_width), dtype=np.float32)
        cheek_mask[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max] = hard_mask
        return cheek_mask


class CheekBlendStrategy(BlendStrategy):
    """Pure matte colour application over a ready soft cheek mask.

    All geometry (convex hull + 3D ellipse) and the fade now live in
    :class:`CheekMaskStrategy`; this only applies the colour.
    """

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: CheekParams,
        blender: object | None = None,
        segmenter: object | None = None,
        landmarks: FaceLandmarks | None = None,
    ) -> np.ndarray:
        return blend_matte(
            image, mask,
            params.color_params.target_rgb,
            params.opacity,
        )


class CheekPass(MakeupPass):
    """Cheek makeup pass with 3D-projected ellipse blending."""

    def __init__(self, params: CheekParams, side: str):
        part_name = f"cheek_{side}"
        indices = CHEEK_LEFT_INDICES if side == "left" else CHEEK_RIGHT_INDICES
        super().__init__(
            pass_type=part_name,
            priority=15,
            params=params,
            landmark_indices=list(indices),
            mask_strategy=CheekMaskStrategy(list(indices), side, params),
            blend_strategy=CheekBlendStrategy(),
        )
