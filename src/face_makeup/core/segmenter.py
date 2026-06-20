"""
Face part segmentation.
Generate masks from landmarks.
"""
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2

from .types import FaceLandmarks, CheekParams
from .eyeliner_corrector import EyelinerCorrector
from ..constants import PART_LANDMARK_MAP, EYELINER_CORRECTION_MAP, resolve_indices


class FaceSegmenter:
    """Generate face part masks from landmarks."""
    def __init__(self):
        self._corrector = EyelinerCorrector()


    def segment(self,
                image_shape: Tuple[int, int],
                landmarks: FaceLandmarks,
                parts: Optional[List[str]] = None,
                image: Optional[np.ndarray] = None,
                eyeliner_gray: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Face part segmentation.

        Args:
            image_shape: (H, W).
            landmarks: FaceLandmarks.
            parts: List of parts to generate (None for all parts).
            image: Original RGB image for eyeliner edge correction.
                   Used to compute eyeliner_gray when it isn't supplied.
                   If both are None, eyeliner uses landmark-only segmentation.
            eyeliner_gray: Preprocessed grayscale image for eyeliner edge
                           detection (see preprocess_for_eyeliner). If None,
                           it is computed once from `image` and shared across
                           eyeliner_left/eyeliner_right so bilateralFilter
                           runs at most once per call.

        Returns:
            Dict[str, np.ndarray]: Part name -> hard mask (H, W) uint8 0/255.
        """
        if parts is None:
            parts = list(PART_LANDMARK_MAP.keys())

        h, w = image_shape
        coords = landmarks.to_image_coords(w, h)
        masks = {}

        # Preprocess once so both eyeliner parts share the same gray image.
        # Caller may supply a pre-computed gray to avoid a redundant bilateralFilter.
        needs_eyeliner_gray = any(part in EYELINER_CORRECTION_MAP for part in parts)
        if eyeliner_gray is None and image is not None and needs_eyeliner_gray:
            eyeliner_gray = self.preprocess_for_eyeliner(image)

        for part in parts:
            if part not in PART_LANDMARK_MAP:
                continue

            part_config = PART_LANDMARK_MAP[part]

            # Get indices and type
            if isinstance(part_config, tuple) and len(part_config) == 2:
                indices, seg_type = part_config
            else:
                indices = part_config
                seg_type = 'convex'  # Default

            # Eyeliner correction: use image-based edge detection when a gray
            # image is available. The corrected flat array is split back into
            # per-sub-array groups so that upper and lower curves are drawn as
            # separate polylines, matching the uncorrected path (avoids a
            # spurious connecting stroke between the two).
            if eyeliner_gray is not None and part in EYELINER_CORRECTION_MAP and isinstance(indices, dict):
                corrected_flat = self._correct_eyeliner_points(coords, part, gray=eyeliner_gray)
                h_m, w_m = image_shape
                mask = np.zeros((h_m, w_m), dtype=np.uint8)
                offset = 0
                for _key, sub_indices in EYELINER_CORRECTION_MAP[part].items():
                    n = len(sub_indices)
                    sub_points = corrected_flat[offset:offset + n]
                    offset += n
                    sub_mask = self._create_mask_from_points(sub_points, image_shape, seg_type)
                    mask = cv2.bitwise_or(mask, sub_mask)
            elif isinstance(indices, dict):
                # Special processing for lips/eyeliner (process upper/lower separately and combine)
                mask = self._create_combined_mask(coords, indices, image_shape, seg_type)
            else:
                # Expand midpoints and get coordinates
                part_points = resolve_indices(indices, coords)
                mask = self._create_mask_from_points(part_points, image_shape, seg_type)

            masks[part] = mask

        return masks

    def _create_combined_mask(self,
                              coords: np.ndarray,
                              indices_dict: Dict[str, List],
                              image_shape: Tuple[int, int],
                              seg_type: str = 'polygon') -> np.ndarray:
        """
        Generate combined mask from multiple parts (for lips and eyeliner).

        Args:
            coords: All landmark coordinates.
            indices_dict: Index dictionary like {'upper': [...], 'lower': [...]}.
            image_shape: (H, W).
            seg_type: Segmentation type ('polygon', 'curve', etc.).

        Returns:
            mask: (H, W) uint8 0/255.
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for key, indices in indices_dict.items():
            part_points = resolve_indices(indices, coords)
            part_mask = self._create_mask_from_points(part_points, image_shape, seg_type)
            mask = cv2.bitwise_or(mask, part_mask)

        return mask

    def _create_mask_from_points(self,
                                  points: np.ndarray,
                                  image_shape: Tuple[int, int],
                                  seg_type: str = 'polygon') -> np.ndarray:
        """
        Generate mask from point cloud.

        Args:
            points: (N, 2) landmark points.
            image_shape: (H, W).
            seg_type: Segmentation type.
                      'polygon' - polygon.
                      'convex' - convex hull.
                      'curve' - curve.

        Returns:
            mask: (H, W) uint8 0/255.
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(points) < 2:
            # Do nothing if less than 2 points
            return mask

        if seg_type == 'polygon':
            # Fill as polygon
            # Polygon connecting points in order (concave shapes allowed)
            if len(points) < 3:
                return mask
            shift = 4
            scale = 1 << shift

            points_shifted = (points * scale).astype(np.int32)

            cv2.fillPoly(mask, [points_shifted], 255, shift=shift)


        elif seg_type == 'convex':
            # Calculate and fill convex hull
            if len(points) < 3:
                return mask
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

        elif seg_type == 'curve':
            # Draw as curve with sub-pixel precision (same shift=4 technique as 'polygon')
            # Preserves the float coordinates returned by eyeliner edge correction.
            if len(points) < 2:
                return mask
            shift = 4
            scale = 1 << shift
            points_shifted = (points * scale).astype(np.int32)
            # Adjust line thickness based on image size
            thickness = max(1, min(h, w) // 512)
            cv2.polylines(mask, [points_shifted], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA, shift=shift)

        elif seg_type == 'ellipse':
            if len(points) < 5:
                return mask
            ellipse = cv2.fitEllipse(points)
            cv2.ellipse(mask, ellipse, color=255, thickness=-1)

        else:
            raise ValueError(f"Unknown segmentation type: {seg_type}")

        return mask

    # ------------------------------------------------------------------
    # Eyeliner edge correction — delegates to EyelinerCorrector
    # ------------------------------------------------------------------

    def preprocess_for_eyeliner(self, image: np.ndarray) -> np.ndarray:
        """Preprocess RGB image for eyeliner edge detection. Returns (H, W) uint8 gray."""
        return self._corrector.preprocess(image)

    def _correct_eyeliner_points(
        self,
        coords: np.ndarray,
        part_name: str,
        gray: np.ndarray,
    ) -> np.ndarray:
        """
        Apply edge-based correction to eyeliner points.

        Returns (M, 2) float32 array of corrected coordinates,
        concatenating upper and lower in correction-map order.
        """
        return self._corrector.correct_points(coords, part_name, gray)

    def get_eyeliner_correction_debug(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        image_shape: Tuple[int, int],
        part_name: str,
        side: str,
        gray: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Run eyeliner correction and return per-point debug information.

        Args:
            image: Original image (H, W, 3) uint8.
            landmarks: Face landmarks.
            image_shape: (H, W).
            part_name: Eyeliner part name (e.g., 'eyeliner_left').
            side: Eye side label ('left' or 'right') stored in each entry.
            gray: Pre-computed grayscale from preprocess_for_eyeliner.
                  If None, preprocessing is run internally.

        Returns:
            List of debug dicts, one per corrected point.
        """
        return self._corrector.get_debug_entries(
            image, landmarks, image_shape, part_name, side, gray
        )

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
        return False

    def close(self):
        """Release resources explicitly."""
        # No ML model resources to release (landmark-based implementation)
        pass

    def __del__(self):
        """Release resources on garbage collection."""
        self.close()

    def create_cheek_mask(self,
                          image_shape: Tuple[int, int],
                          base_mask: np.ndarray,
                          params: CheekParams,
                          landmarks: Optional[FaceLandmarks] = None,
                          cheek_side: str = 'left',
                          return_debug_info: bool = False) -> np.ndarray:
        """
        Generate hard-edge ellipse mask for cheek.

        Fadeout effect is achieved via blur_strength in blend().

        Args:
            image_shape: Image size (h, w).
            base_mask: Base mask.
            params: Cheek parameters.
            landmarks: Face landmarks (including 3D coordinates).
            cheek_side: 'left' or 'right'.

        Returns:
            Ellipse mask (h, w) float32 with hard edges (0.0 or 1.0).
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

    def _create_cheek_mask_3d(self,
                              image_shape: Tuple[int, int],
                              base_mask: np.ndarray,
                              params: CheekParams,
                              landmarks: FaceLandmarks,
                              cheek_side: str,
                              return_debug_info: bool = False) -> np.ndarray:
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
            Ellipse mask (h, w) float32 (0..1).
        """
        from face_makeup.constants import CHEEK_LEFT_INDICES, CHEEK_RIGHT_INDICES

        img_height, img_width = image_shape

        if img_height <= 0 or img_width <= 0:
            raise ValueError(f"Invalid image_shape: {image_shape}. Both dimensions must be positive.")

        face3d_all_points = landmarks.points_3d
        if face3d_all_points is None:
            return np.zeros((img_height, img_width), dtype=np.float32)

        cheek_indices = CHEEK_LEFT_INDICES if cheek_side == 'left' else CHEEK_RIGHT_INDICES
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
        result_mask = self._rasterize_ellipse_mask(center_px_x, center_px_y, polygon_px, image_shape)

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
                'ellipse_tangent_center_u': tangent_u_center + params.center[0] * tangent_u_halfspan,
                'ellipse_tangent_center_v': tangent_v_center + params.center[1] * tangent_v_halfspan,
                'ellipse_tangent_radius_u': abs(params.radius[0]) * tangent_u_halfspan,
                'ellipse_tangent_radius_v': abs(params.radius[1]) * tangent_v_halfspan,
                'surface_coeffs': surface_coeffs,
                'ellipse_face3d_points': ellipse_face3d_points,
                'cheek_tangent_uvw': cheek_uvw,
                'cheek_img_norm': cheek_img_norm,
            }
            return result_mask, debug_info

        return result_mask

    def _build_cheek_tangent_plane(
        self,
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
            - origin: (3,) centroid of cheek landmarks in Face 3D space.
            - basis_u/v/w: (3,) orthonormal basis vectors.
            - cheek_uvw: (N, 3) cheek landmarks in tangent-plane coordinates.
            - surface_coeffs: (6,) quadratic surface coefficients [c0..c5].
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

    def _project_ellipse_to_image(
        self,
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
        Compute ellipse parameters in tangent space and project to image pixel coordinates.

        Steps 6-9 of the 3D cheek mask algorithm.

        Args:
            tangent_origin: (3,) origin of the tangent plane in Face 3D space.
            basis_u/v/w: (3,) orthonormal basis vectors.
            cheek_uvw: (N, 3) cheek landmarks in tangent-plane coordinates.
            surface_coeffs: (6,) quadratic surface coefficients.
            params: CheekParams with center and radius in normalised tangent space.
            image_shape: (height, width) in pixels.

        Returns:
            (center_px_x, center_px_y, polygon_px, ellipse_face3d_points)
            - center_px_x/y: ellipse center in pixel coordinates.
            - polygon_px: (50, 2) ellipse boundary polygon in pixel coordinates.
            - ellipse_face3d_points: (50, 3) ellipse boundary in Face 3D space.
        """
        img_height, img_width = image_shape

        def _eval_w(u, v):
            c = surface_coeffs
            return c[0] + c[1] * u + c[2] * v + c[3] * u ** 2 + c[4] * v ** 2 + c[5] * u * v

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
        center_face3d = tangent_origin + ell_cu * basis_u + ell_cv * basis_v + center_w * basis_w
        center_px_x = float(center_face3d[0] * img_width)
        center_px_y = float(center_face3d[1] * img_height)

        return center_px_x, center_px_y, polygon_px, face3d_pts

    def _rasterize_ellipse_mask(
        self,
        center_px_x: float,
        center_px_y: float,
        polygon_px: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Rasterize a polygon to a float32 hard-edge mask using distance-based fill.

        Steps 10-11 of the 3D cheek mask algorithm.

        Args:
            center_px_x/y: Ellipse center in pixel coordinates.
            polygon_px: (N, 2) boundary polygon in pixel coordinates.
            image_shape: (height, width) in pixels.

        Returns:
            Float32 mask (h, w) with 1.0 inside the polygon and 0.0 outside.
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
        local_poly = np.round(polygon_px - np.array([[bbox_x_min, bbox_y_min]])).astype(np.int32)
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
        ext_angles = np.concatenate([sorted_angles - 2 * np.pi, sorted_angles, sorted_angles + 2 * np.pi])
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
