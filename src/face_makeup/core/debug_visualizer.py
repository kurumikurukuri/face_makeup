"""
Debug visualization module for makeup pipeline.

Provides visualization utilities for debugging each step of the makeup pipeline.
"""
import os
import logging
from typing import Dict, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .types import FaceBBox, FaceLandmarks, CheekParams
    from .segmenter import FaceSegmenter

logger = logging.getLogger(__name__)


class DebugVisualizer:
    """Handles debug visualization for the makeup pipeline."""

    def __init__(self, debug_dir: str = "outputs/debug"):
        """
        Initialize the debug visualizer.

        Args:
            debug_dir: Directory to save debug images.
        """
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    def save_original(self, image: np.ndarray, session_id: str) -> None:
        """Save the original input image."""
        self._save_image(image, session_id, "0_original")

    def save_bbox(
        self,
        image: np.ndarray,
        bbox: "FaceBBox",
        session_id: str
    ) -> None:
        """
        Save image with face bounding box overlay.

        Args:
            image: RGB image (H, W, 3).
            bbox: Detected face bounding box.
            session_id: Session identifier for filename.
        """
        img = image.copy()
        cv2.rectangle(
            img,
            (bbox.x, bbox.y),
            (bbox.x + bbox.w, bbox.y + bbox.h),
            (0, 255, 0),
            2
        )
        self._save_image(img, session_id, "1_bbox")

    def save_landmarks(
        self,
        image: np.ndarray,
        landmarks: "FaceLandmarks",
        session_id: str
    ) -> None:
        """
        Save image with facial landmarks overlay.

        Args:
            image: RGB image (H, W, 3).
            landmarks: Extracted facial landmarks.
            session_id: Session identifier for filename.
        """
        img = image.copy()
        h, w = image.shape[:2]
        coords = landmarks.to_image_coords(w, h)

        for x, y in coords:
            cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), -1)

        self._save_image(img, session_id, "2_landmarks")

    def save_masks(
        self,
        image: np.ndarray,
        masks: Dict[str, np.ndarray],
        session_id: str,
        mask_type: str
    ) -> None:
        """
        Save mask visualizations as heatmap overlays.

        Args:
            image: RGB image (H, W, 3).
            masks: Dictionary of part_name -> mask.
            session_id: Session identifier for filename.
            mask_type: Type of mask ('hard' or 'soft').
        """
        for part, mask in masks.items():
            mask_vis = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                0.6,
                heatmap,
                0.4,
                0
            )
            path = os.path.join(
                self.debug_dir,
                f"{session_id}_3_{mask_type}_{part}.png"
            )
            cv2.imwrite(path, overlay)
            logger.debug(f"Saved {mask_type} mask: {path}")

    def save_result(
        self,
        original: np.ndarray,
        result: np.ndarray,
        session_id: str
    ) -> None:
        """
        Save final result and difference heatmap.

        Args:
            original: Original RGB image.
            result: Processed RGB image.
            session_id: Session identifier for filename.
        """
        # Save result
        self._save_image(result, session_id, "5_result")

        # Save difference heatmap
        diff = cv2.absdiff(original, result)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)
        path = os.path.join(self.debug_dir, f"{session_id}_6_diff.png")
        cv2.imwrite(path, heatmap)
        logger.debug(f"Saved diff heatmap: {path}")

    def save_cheek_3d_debug(
        self,
        image: np.ndarray,
        soft_mask: np.ndarray,
        params: "CheekParams",
        landmarks: "FaceLandmarks",
        segmenter: "FaceSegmenter",
        part_name: str,
        cheek_side: str,
        session_id: str
    ) -> None:
        """
        Save detailed 3D cheek ellipse debug visualization.

        Args:
            image: RGB image.
            soft_mask: Soft mask for the cheek.
            params: Cheek parameters.
            landmarks: Facial landmarks with 3D coordinates.
            segmenter: FaceSegmenter instance for mask generation.
            part_name: Name of the part (e.g., 'cheek_left').
            cheek_side: 'left' or 'right'.
            session_id: Session identifier.
        """
        # Get ellipse mask with debug info
        result = segmenter.create_cheek_mask(
            image.shape[:2],
            soft_mask,
            params,
            landmarks=landmarks,
            cheek_side=cheek_side,
            return_debug_info=True
        )

        if not isinstance(result, tuple):
            return

        cheek_mask, debug_info = result

        self._save_ellipse_mask_heatmap(
            image, cheek_mask, part_name, session_id
        )

        # Additional coordinate visualizations
        if 'cheek_tangent_uvw' in debug_info:
            self._save_uv_visualization(
                image, debug_info, params, part_name, cheek_side, session_id
            )

    def _save_image(
        self,
        image: np.ndarray,
        session_id: str,
        suffix: str
    ) -> None:
        """Save an RGB image to disk."""
        path = os.path.join(self.debug_dir, f"{session_id}_{suffix}.png")
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved image: {path}")


    def _save_ellipse_mask_heatmap(
        self,
        image: np.ndarray,
        cheek_mask: np.ndarray,
        part_name: str,
        session_id: str
    ) -> None:
        """Save ellipse mask as heatmap overlay."""
        mask_vis = (cheek_mask * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            0.6,
            heatmap,
            0.4,
            0
        )
        path = os.path.join(
            self.debug_dir,
            f"{session_id}_4_cheek_3d_mask_{part_name}.png"
        )
        cv2.imwrite(path, overlay)

    def _save_uv_visualization(
        self,
        image: np.ndarray,
        debug_info: dict,
        params,
        part_name: str,
        cheek_side: str,
        session_id: str
    ) -> None:
        """
        Save comprehensive visualization of tangent plane coordinate system.

        Coordinate Systems visualized:
            - Image Space (img_px_): Pixel coordinates in the image
            - Face 3D Space (face3d_): MediaPipe normalized 3D coordinates
            - Tangent Plane Space (tangent_): Local UVW coordinate system
                - U: horizontal direction on cheek surface
                - V: vertical direction on cheek surface
                - W: normal direction (perpendicular to UV plane)

        Panels:
            [left]     Image space with 3D-projected ellipse
            [right]  3D view of tangent UVW space
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
        except ImportError:
            return

        # =================================================================
        # Extract data from debug_info
        # =================================================================

        # Cheek points in normalized image space (0-1)
        cheek_img_norm = np.asarray(debug_info["cheek_img_norm"])

        # Cheek points in tangent UVW space (N,3)
        cheek_tangent_uvw = np.asarray(debug_info["cheek_tangent_uvw"])
        tangent_u_coords = cheek_tangent_uvw[:, 0]  # U coordinates in tangent space
        tangent_v_coords = cheek_tangent_uvw[:, 1]  # V coordinates in tangent space
        tangent_w_coords = cheek_tangent_uvw[:, 2]  # W coordinates in tangent space

        # Tangent plane basis vectors (in Face 3D space)
        tangent_origin_face3d = np.asarray(debug_info["tangent_origin_face3d"])
        tangent_basis_u = np.asarray(debug_info["tangent_basis_u"])
        tangent_basis_v = np.asarray(debug_info["tangent_basis_v"])
        tangent_basis_w = np.asarray(debug_info["tangent_basis_w"])


        # Surface fitting coefficients
        surface_coeffs = np.asarray(debug_info["surface_coeffs"])

        # Ellipse boundary points in Face 3D space
        ellipse_face3d_points = np.asarray(debug_info["ellipse_face3d_points"])

        img_height, img_width = image.shape[:2]
        cheek_img_px = np.column_stack([
            cheek_img_norm[:, 0] * img_width,
            cheek_img_norm[:, 1] * img_height
        ])

        bbox_x_min, bbox_y_min = cheek_img_px.min(axis=0)
        bbox_x_max, bbox_y_max = cheek_img_px.max(axis=0)

        margin_ratio = 0.25
        margin_x = (bbox_x_max - bbox_x_min) * margin_ratio
        margin_y = (bbox_y_max - bbox_y_min) * margin_ratio

        bbox_x_min -= margin_x
        bbox_x_max += margin_x
        bbox_y_min -= margin_y
        bbox_y_max += margin_y

        bbox_x_min = max(0, bbox_x_min)
        bbox_y_min = max(0, bbox_y_min)
        bbox_x_max = min(img_width, bbox_x_max)
        bbox_y_max = min(img_height, bbox_y_max)

        point_colors = plt.get_cmap("tab10")(np.arange(len(tangent_u_coords)) % 10)

        # ===== 2x2 grid only =====
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 2)

        # ==========================
        # left : Image Space (3D-projected ellipse)
        # ==========================
        ax_img_3d_proj = fig.add_subplot(gs[0, 0])
        ax_img_3d_proj.imshow(image)

        ellipse_img_px_from_face3d = np.column_stack([
            ellipse_face3d_points[:, 0] * img_width,
            ellipse_face3d_points[:, 1] * img_height
        ])

        origin_img_px = np.array([
            tangent_origin_face3d[0] * img_width,
            tangent_origin_face3d[1] * img_height
        ])

        axis_scale = max(bbox_x_max - bbox_x_min, bbox_y_max - bbox_y_min) * 0.01

        u_axis_img_px = np.array([
            (tangent_origin_face3d + tangent_basis_u * axis_scale)[0] * img_width,
            (tangent_origin_face3d + tangent_basis_u * axis_scale)[1] * img_height
        ]) - origin_img_px

        v_axis_img_px = np.array([
            (tangent_origin_face3d + tangent_basis_v * axis_scale)[0] * img_width,
            (tangent_origin_face3d + tangent_basis_v * axis_scale)[1] * img_height
        ]) - origin_img_px

        ax_img_3d_proj.arrow(
            origin_img_px[0], origin_img_px[1],
            u_axis_img_px[0], u_axis_img_px[1],
            color="r", width=1.5, length_includes_head=True
        )
        ax_img_3d_proj.arrow(
            origin_img_px[0], origin_img_px[1],
            v_axis_img_px[0], v_axis_img_px[1],
            color="g", width=1.5, length_includes_head=True
        )

        for i, ((x, y), c) in enumerate(zip(cheek_img_px, point_colors)):
            ax_img_3d_proj.scatter(x, y, s=1, color=c, edgecolor="black")
            ax_img_3d_proj.text(x + 4, y + 4, str(i), color="white", fontsize=11, weight="bold")

        ax_img_3d_proj.plot(
            ellipse_img_px_from_face3d[:, 0],
            ellipse_img_px_from_face3d[:, 1],
            color="lime",
            linewidth=2
        )

        ax_img_3d_proj.set_title("Image Space (Face3D-projected ellipse)")
        ax_img_3d_proj.set_xlim(bbox_x_min, bbox_x_max)
        ax_img_3d_proj.set_ylim(bbox_y_max, bbox_y_min)
        ax_img_3d_proj.set_aspect("equal")
        ax_img_3d_proj.grid(True, alpha=0.3)

        # ==========================
        # right : Tangent UVW Space (3D view)
        # ==========================
        ax_3d = fig.add_subplot(gs[0, 1], projection="3d")

        # Reconstruct 3D points from UVW
        cheek_face3d_reconstructed = (
            tangent_origin_face3d[None]
            + tangent_u_coords[:, None] * tangent_basis_u[None]
            + tangent_v_coords[:, None] * tangent_basis_v[None]
            + tangent_w_coords[:, None] * tangent_basis_w[None]
        )

        # Plot reconstructed points
        ax_3d.scatter(
            cheek_face3d_reconstructed[:, 0],
            cheek_face3d_reconstructed[:, 1],
            cheek_face3d_reconstructed[:, 2],
            c=point_colors, s=40
        )

        # Plot tangent surface (semi-transparent)
        tangent_u_grid = np.linspace(tangent_u_coords.min(), tangent_u_coords.max(), 25)
        tangent_v_grid = np.linspace(tangent_v_coords.min(), tangent_v_coords.max(), 25)
        tangent_u_mesh, tangent_v_mesh = np.meshgrid(tangent_u_grid, tangent_v_grid)

        tangent_w_mesh = (
            surface_coeffs[0]
            + surface_coeffs[1] * tangent_u_mesh + surface_coeffs[2] * tangent_v_mesh
            + surface_coeffs[3] * tangent_u_mesh ** 2 + surface_coeffs[4] * tangent_v_mesh ** 2
            + surface_coeffs[5] * tangent_u_mesh * tangent_v_mesh
        )

        surface_face3d = (
            tangent_origin_face3d[None, None]
            + tangent_u_mesh[..., None] * tangent_basis_u
            + tangent_v_mesh[..., None] * tangent_basis_v
            + tangent_w_mesh[..., None] * tangent_basis_w
        )

        ax_3d.plot_surface(
            surface_face3d[..., 0],
            surface_face3d[..., 1],
            surface_face3d[..., 2],
            alpha=0.35,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True
        )

        # Plot ellipse on surface (thick green line)
        ax_3d.plot(
            ellipse_face3d_points[:, 0],
            ellipse_face3d_points[:, 1],
            ellipse_face3d_points[:, 2],
            color="green", linewidth=2.5
        )

        # --------------------
        # Plot UVW basis vectors
        # --------------------
        origin = tangent_origin_face3d
        scale = max(tangent_u_coords.max()-tangent_u_coords.min(),
                    tangent_v_coords.max()-tangent_v_coords.min()) * 0.1

        ax_3d.quiver(
            origin[0], origin[1], origin[2],
            tangent_basis_u[0], tangent_basis_u[1], tangent_basis_u[2],
            length=scale, color="r", normalize=True
        )
        ax_3d.quiver(
            origin[0], origin[1], origin[2],
            tangent_basis_v[0], tangent_basis_v[1], tangent_basis_v[2],
            length=scale, color="g", normalize=True
        )
        ax_3d.quiver(
            origin[0], origin[1], origin[2],
            tangent_basis_w[0], tangent_basis_w[1], tangent_basis_w[2],
            length=scale, color="b", normalize=True
        )

        # --------------------
        # Labels
        # --------------------
        ax_3d.text(origin[0] + tangent_basis_u[0]*scale,
                   origin[1] + tangent_basis_u[1]*scale,
                   origin[2] + tangent_basis_u[2]*scale, "U", color="r", fontsize=12)

        ax_3d.text(origin[0] + tangent_basis_v[0]*scale,
                   origin[1] + tangent_basis_v[1]*scale,
                   origin[2] + tangent_basis_v[2]*scale, "V", color="g", fontsize=12)

        ax_3d.text(origin[0] + tangent_basis_w[0]*scale,
                   origin[1] + tangent_basis_w[1]*scale,
                   origin[2] + tangent_basis_w[2]*scale, "W", color="b", fontsize=12)

        # --------------------
        # Fixed camera view
        # --------------------
        ax_3d.view_init(elev=20, azim=135)

        ax_3d.set_title("Tangent UVW Space (3D view)")
        ax_3d.set_box_aspect([1, 1, 1])
        ax_3d.grid(True, alpha=0.3)


        # ---------- save ----------
        path = os.path.join(
            self.debug_dir,
            f"{session_id}_4_cheek_3d_uv_full_{part_name}.png"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()