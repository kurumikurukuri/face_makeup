"""
Face part segmentation.
Generate masks from landmarks.
"""
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2

from .types import FaceLandmarks, CheekParams
from ..constants import PART_LANDMARK_MAP



class FaceSegmenter:
    """Generate face part masks from landmarks."""
    def __init__(self):
        pass


    def _expand_indices_with_midpoints(self,
                                       indices: List,
                                       coords: np.ndarray) -> np.ndarray:
        """
        Get coordinates from index list and expand midpoints.

        Args:
            indices: List of landmark indices.
                     Regular int, or ('mid', idx1, idx2, ...) tuple.
            coords: All landmark coordinates (N, 2).

        Returns:
            np.ndarray: Expanded coordinates (M, 2).

        Example:
            indices = [33, ('mid', 33, 133), 159]
            -> [coord[33], (coord[33]+coord[133])/2, coord[159]]
        """
        expanded_coords = []

        for idx in indices:
            if isinstance(idx, tuple) and idx[0] == 'mid':
                # Calculate midpoint
                midpoint_indices = idx[1:]
                if len(midpoint_indices) < 2:
                    raise ValueError(f"Midpoint requires at least 2 indices, got {len(midpoint_indices)}")

                # Get coordinates for specified indices
                points = coords[list(midpoint_indices)]
                # Calculate mean (centroid)
                midpoint = np.mean(points, axis=0)
                expanded_coords.append(midpoint)
            elif isinstance(idx, int):
                # Normal landmark
                expanded_coords.append(coords[idx])
            else:
                raise ValueError(f"Invalid index type: {type(idx)}. Expected int or ('mid', ...) tuple.")

        return np.array(expanded_coords, dtype=coords.dtype)

    def segment(self,
                image_shape: Tuple[int, int],
                landmarks: FaceLandmarks,
                parts: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Face part segmentation.

        Args:
            image_shape: (H, W).
            landmarks: FaceLandmarks.
            parts: List of parts to generate (None for all parts).

        Returns:
            Dict[str, np.ndarray]: Part name -> hard mask (H, W) uint8 0/255.
        """
        if parts is None:
            parts = list(PART_LANDMARK_MAP.keys())

        h, w = image_shape
        coords = landmarks.to_image_coords(w, h)
        masks = {}

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

            # Special processing for lips/eyeliner (process upper/lower separately and combine)
            if isinstance(indices, dict):
                mask = self._create_combined_mask(coords, indices, image_shape, seg_type)
            else:
                # Expand midpoints and get coordinates
                part_points = self._expand_indices_with_midpoints(indices, coords)
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
            part_points = self._expand_indices_with_midpoints(indices, coords)
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
            points_int = points.astype(np.int32)
            cv2.fillPoly(mask, [points_int], 255)

        elif seg_type == 'convex':
            # Calculate and fill convex hull
            if len(points) < 3:
                return mask
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

        elif seg_type == 'curve':
            # Draw as curve
            # For thin lines like eyeliner
            if len(points) < 2:
                return mask
            points_int = points.astype(np.int32)
            # Adjust line thickness based on image size
            thickness = max(2, min(h, w) // 500)
            cv2.polylines(mask, [points_int], isClosed=False, color=255, thickness=thickness)
            
        elif seg_type == 'ellipse':
            if len(points) < 5:
                return mask
            ellipse = cv2.fitEllipse(points)
            cv2.ellipse(mask, ellipse, color=255, thickness=-1)

        else:
            raise ValueError(f"Unknown segmentation type: {seg_type}")

        return mask

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
        Uses cv2.fillPoly for polygon rasterization and quadratic surface fitting.

        Coordinate Systems:
            - Face 3D Space (face3d_): MediaPipe normalized 3D coordinates (0-1)
            - Tangent Plane Space (tangent_): Local UVW coordinate system on cheek surface
                - U: horizontal direction along the cheek
                - V: vertical direction along the cheek
                - W: normal direction (perpendicular to UV plane)
            - Image Space (img_px_): Pixel coordinates

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

        # Input validation
        if img_height <= 0 or img_width <= 0:
            raise ValueError(f"Invalid image_shape: {image_shape}. Both dimensions must be positive.")

        face3d_all_points = landmarks.points_3d

        if face3d_all_points is None:
            # Return zero mask if no 3D info
            return np.zeros((img_height, img_width), dtype=np.float32)

        # =================================================================
        # 1. Extract cheek landmarks in both coordinate systems
        # =================================================================
        cheek_indices = CHEEK_LEFT_INDICES if cheek_side == 'left' else CHEEK_RIGHT_INDICES
        cheek_face3d = face3d_all_points[cheek_indices]  # Cheek points in Face 3D space
        cheek_img_norm = landmarks.points[cheek_indices]  # Cheek points in normalized image space (0-1)

        # =================================================================
        # 2. Define landmarks for tangent plane coordinate system
        # =================================================================
        if cheek_side == 'left':
            u_start_idx, u_end_idx = 50, 36    # Points defining U-axis direction
            v_start_idx, v_end_idx = 118, 205  # Points defining V-axis direction
        else:
            u_start_idx, u_end_idx = 280, 266
            v_start_idx, v_end_idx = 347, 425

        face3d_u_start = face3d_all_points[u_start_idx]
        face3d_u_end = face3d_all_points[u_end_idx]
        face3d_v_start = face3d_all_points[v_start_idx]
        face3d_v_end = face3d_all_points[v_end_idx]

        # =================================================================
        # 3. Build tangent plane coordinate system (UVW basis)
        # =================================================================
        # Origin: centroid of cheek landmarks in Face 3D space
        tangent_origin_face3d = cheek_face3d.mean(axis=0)

        # U-axis: horizontal direction along the cheek
        u_direction_raw = face3d_u_end - face3d_u_start
        tangent_basis_u = u_direction_raw / (np.linalg.norm(u_direction_raw) + 1e-6)

        # V-axis: vertical direction (Gram-Schmidt orthogonalization to U)
        v_direction_raw = face3d_v_end - face3d_v_start
        v_direction_raw = v_direction_raw - np.dot(v_direction_raw, tangent_basis_u) * tangent_basis_u
        tangent_basis_v = v_direction_raw / (np.linalg.norm(v_direction_raw) + 1e-6)

        # W-axis: normal direction (cross product of U and V)
        tangent_basis_w = np.cross(tangent_basis_u, tangent_basis_v)
        tangent_basis_w = tangent_basis_w / (np.linalg.norm(tangent_basis_w) + 1e-6)

        # =================================================================
        # 4. Transform cheek landmarks to tangent plane coordinates (UVW)
        # =================================================================
        cheek_tangent_uvw = []
        for pt_face3d in cheek_face3d:
            vec_from_origin = pt_face3d - tangent_origin_face3d
            tangent_u = np.dot(vec_from_origin, tangent_basis_u)
            tangent_v = np.dot(vec_from_origin, tangent_basis_v)
            tangent_w = np.dot(vec_from_origin, tangent_basis_w)
            cheek_tangent_uvw.append([tangent_u, tangent_v, tangent_w])
        cheek_tangent_uvw = np.array(cheek_tangent_uvw)

        # =================================================================
        # 5. Surface approximation (quadratic surface fitting in tangent space)
        #    w = c0 + c1*u + c2*v + c3*u^2 + c4*v^2 + c5*u*v
        # =================================================================
        tangent_u_coords = cheek_tangent_uvw[:, 0]
        tangent_v_coords = cheek_tangent_uvw[:, 1]
        tangent_w_coords = cheek_tangent_uvw[:, 2]

        # Design matrix for least squares fitting
        design_matrix = np.column_stack([
            np.ones_like(tangent_u_coords),
            tangent_u_coords,
            tangent_v_coords,
            tangent_u_coords ** 2,
            tangent_v_coords ** 2,
            tangent_u_coords * tangent_v_coords
        ])

        # Least squares fitting
        try:
            surface_coeffs, _, _, _ = np.linalg.lstsq(design_matrix, tangent_w_coords, rcond=None)
        except np.linalg.LinAlgError:
            # Fall back to planar approximation on fitting failure
            surface_coeffs = np.zeros(6)
            surface_coeffs[0] = tangent_w_coords.mean()

        # Surface evaluation function: tangent (u, v) -> w
        def evaluate_surface_w(u, v):
            return (surface_coeffs[0] + surface_coeffs[1] * u + surface_coeffs[2] * v +
                    surface_coeffs[3] * u ** 2 + surface_coeffs[4] * v ** 2 + surface_coeffs[5] * u * v)

        # =================================================================
        # 6. Calculate tangent space UV range and define normalized space
        # =================================================================
        tangent_u_min = tangent_u_coords.min() if len(tangent_u_coords) > 0 else -0.1
        tangent_u_max = tangent_u_coords.max() if len(tangent_u_coords) > 0 else 0.1
        tangent_v_min = tangent_v_coords.min() if len(tangent_v_coords) > 0 else -0.1
        tangent_v_max = tangent_v_coords.max() if len(tangent_v_coords) > 0 else 0.1

        # Tangent space center and half-span
        tangent_u_center = (tangent_u_max + tangent_u_min) / 2
        tangent_v_center = (tangent_v_max + tangent_v_min) / 2
        tangent_u_halfspan = (tangent_u_max - tangent_u_min) / 2
        tangent_v_halfspan = (tangent_v_max - tangent_v_min) / 2

        # =================================================================
        # 7. Calculate ellipse parameters in tangent space
        #    params.center in (-1, 1), params.radius in (0, 1]
        # =================================================================
        ellipse_tangent_center_u = tangent_u_center + params.center[0] * tangent_u_halfspan
        ellipse_tangent_center_v = tangent_v_center + params.center[1] * tangent_v_halfspan
        ellipse_tangent_radius_u = abs(params.radius[0]) * tangent_u_halfspan
        ellipse_tangent_radius_v = abs(params.radius[1]) * tangent_v_halfspan

        # Generate ellipse boundary points in tangent space (parametric form)
        num_ellipse_samples = 50
        theta = np.linspace(0, 2 * np.pi, num_ellipse_samples)
        ellipse_tangent_u = ellipse_tangent_center_u + ellipse_tangent_radius_u * np.cos(theta)
        ellipse_tangent_v = ellipse_tangent_center_v + ellipse_tangent_radius_v * np.sin(theta)

        # =================================================================
        # 8. Project ellipse boundary onto surface and convert to Face 3D space
        # =================================================================
        ellipse_face3d_points = []
        for u, v in zip(ellipse_tangent_u, ellipse_tangent_v):
            w = evaluate_surface_w(u, v)
            # Convert tangent (u,v,w) back to Face 3D space
            pt_face3d = (tangent_origin_face3d +
                         u * tangent_basis_u +
                         v * tangent_basis_v +
                         w * tangent_basis_w)
            ellipse_face3d_points.append(pt_face3d)
        ellipse_face3d_points = np.array(ellipse_face3d_points)

        # =================================================================
        # 9. Generate 2D mask using Face3D projection
        # =================================================================
        # Project ellipse center from tangent space to image pixel space
        center_tangent_w = evaluate_surface_w(ellipse_tangent_center_u, ellipse_tangent_center_v)
        center_face3d = (tangent_origin_face3d +
                         ellipse_tangent_center_u * tangent_basis_u +
                         ellipse_tangent_center_v * tangent_basis_v +
                         center_tangent_w * tangent_basis_w)
        ellipse_img_px_center_x = center_face3d[0] * img_width
        ellipse_img_px_center_y = center_face3d[1] * img_height

        # Convert ellipse Face3D points to image pixel coordinates
        ellipse_img_px_points = np.column_stack([
            ellipse_face3d_points[:, 0] * img_width,
            ellipse_face3d_points[:, 1] * img_height
        ])

        # Calculate bounding box for efficiency
        bbox_x_min = max(0, int(ellipse_img_px_points[:, 0].min()) - 10)
        bbox_x_max = min(img_width, int(ellipse_img_px_points[:, 0].max()) + 10)
        bbox_y_min = max(0, int(ellipse_img_px_points[:, 1].min()) - 10)
        bbox_y_max = min(img_height, int(ellipse_img_px_points[:, 1].max()) + 10)

        # Create pixel grid within bounding box
        bbox_ys, bbox_xs = np.meshgrid(
            np.arange(bbox_y_min, bbox_y_max),
            np.arange(bbox_x_min, bbox_x_max),
            indexing="ij"
        )

        # =================================================================
        # 10. Calculate normalized distance from center for each pixel
        # =================================================================
        # For each pixel, calculate distance to center relative to ellipse boundary

        # Check which pixels are inside the ellipse polygon (cv2.fillPoly)
        bbox_h = bbox_y_max - bbox_y_min
        bbox_w = bbox_x_max - bbox_x_min
        poly_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        local_poly = np.round(ellipse_img_px_points - np.array([[bbox_x_min, bbox_y_min]])).astype(np.int32)
        cv2.fillPoly(poly_mask, [local_poly], 255)
        pixels_inside_ellipse = poly_mask > 0

        # Calculate distance from center for gradient
        center_img_px = np.array([ellipse_img_px_center_x, ellipse_img_px_center_y])

        # Vector from center to each pixel
        vec_from_center_x = bbox_xs - ellipse_img_px_center_x
        vec_from_center_y = bbox_ys - ellipse_img_px_center_y

        # Distance from center
        dist_from_center = np.sqrt(vec_from_center_x ** 2 + vec_from_center_y ** 2)

        # For each pixel, find the distance to ellipse boundary along the ray from center
        # Approximate by finding the boundary point in the same direction
        angle_from_center = np.arctan2(vec_from_center_y, vec_from_center_x)

        # Calculate boundary distance for each angle
        # Ellipse boundary points are sampled uniformly in theta
        ellipse_angles = np.arctan2(
            ellipse_img_px_points[:, 1] - ellipse_img_px_center_y,
            ellipse_img_px_points[:, 0] - ellipse_img_px_center_x
        )
        ellipse_radii = np.sqrt(
            (ellipse_img_px_points[:, 0] - ellipse_img_px_center_x) ** 2 +
            (ellipse_img_px_points[:, 1] - ellipse_img_px_center_y) ** 2
        )

        # Interpolate boundary radius for each pixel's angle
        # Sort by angle for interpolation
        sort_idx = np.argsort(ellipse_angles)
        sorted_angles = ellipse_angles[sort_idx]
        sorted_radii = ellipse_radii[sort_idx]

        # Extend for wrap-around interpolation
        extended_angles = np.concatenate([
            sorted_angles - 2 * np.pi,
            sorted_angles,
            sorted_angles + 2 * np.pi
        ])
        extended_radii = np.concatenate([sorted_radii, sorted_radii, sorted_radii])

        # Interpolate boundary radius for each pixel
        boundary_radius = np.interp(
            angle_from_center.flatten(),
            extended_angles,
            extended_radii
        ).reshape(angle_from_center.shape)

        # Normalized distance (0 at center, 1 at boundary)
        normalized_distance = dist_from_center / (boundary_radius + 1e-6)

        # =================================================================
        # 11. Create hard-edge mask (fadeout via blur_strength in blend())
        # =================================================================
        hard_mask_local = np.zeros_like(normalized_distance, dtype=np.float32)

        # Inside ellipse: set to 1.0 (hard edge)
        inside_mask = pixels_inside_ellipse & (normalized_distance <= 1.0)
        hard_mask_local[inside_mask] = 1.0

        # Combine into final mask
        cheek_mask = np.zeros((img_height, img_width), dtype=np.float32)
        cheek_mask[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max] = hard_mask_local

        result_mask = cheek_mask

        # Calculate auxiliary values for debug_info
        img_u_axis_unit = np.array([1.0, 0.0])  # Placeholder for compatibility
        img_v_axis_unit = np.array([0.0, 1.0])
        ellipse_img_px_radius_x = np.mean(ellipse_radii)
        ellipse_img_px_radius_y = np.mean(ellipse_radii)

        if return_debug_info:
            debug_info = {
                # Tangent plane coordinate system (in Face 3D space)
                'tangent_origin_face3d': tangent_origin_face3d,
                'tangent_basis_u': tangent_basis_u,
                'tangent_basis_v': tangent_basis_v,
                'tangent_basis_w': tangent_basis_w,

                # Ellipse in image pixel space
                'ellipse_img_px_center_x': ellipse_img_px_center_x,
                'ellipse_img_px_center_y': ellipse_img_px_center_y,
                'ellipse_img_px_radius_x': ellipse_img_px_radius_x,
                'ellipse_img_px_radius_y': ellipse_img_px_radius_y,

                # Projected tangent axes in image space (unit vectors)
                'img_u_axis_unit': img_u_axis_unit,
                'img_v_axis_unit': img_v_axis_unit,

                # Tangent space UV range
                'tangent_u_min': tangent_u_min,
                'tangent_u_max': tangent_u_max,
                'tangent_v_min': tangent_v_min,
                'tangent_v_max': tangent_v_max,
                'tangent_u_center': tangent_u_center,
                'tangent_v_center': tangent_v_center,
                'tangent_u_halfspan': tangent_u_halfspan,
                'tangent_v_halfspan': tangent_v_halfspan,

                # Ellipse parameters in tangent space
                'ellipse_tangent_center_u': ellipse_tangent_center_u,
                'ellipse_tangent_center_v': ellipse_tangent_center_v,
                'ellipse_tangent_radius_u': ellipse_tangent_radius_u,
                'ellipse_tangent_radius_v': ellipse_tangent_radius_v,

                # Surface fitting coefficients
                'surface_coeffs': surface_coeffs,

                # For visualization
                'ellipse_face3d_points': ellipse_face3d_points,
                'cheek_tangent_uvw': cheek_tangent_uvw,
                'cheek_img_norm': cheek_img_norm,
            }
            return result_mask, debug_info

        return result_mask