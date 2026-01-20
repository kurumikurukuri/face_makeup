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
    from .makeup_blender import MakeupBlender

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
        blender: "MakeupBlender",
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
            blender: MakeupBlender instance for mask generation.
            part_name: Name of the part (e.g., 'cheek_left').
            cheek_side: 'left' or 'right'.
            session_id: Session identifier.
        """
        # Get ellipse mask with debug info
        result = blender._create_ellipse_mask(
            image.shape[:2],
            soft_mask,
            params,
            landmarks=landmarks,
            cheek_side=cheek_side,
            return_debug_info=True
        )

        if not isinstance(result, tuple):
            return

        ellipse_mask, debug_info = result
        self._draw_cheek_debug_image(
            image, debug_info, landmarks, part_name, cheek_side, session_id
        )
        self._save_ellipse_mask_heatmap(
            image, ellipse_mask, part_name, session_id
        )

        # Additional coordinate visualizations
        if 'p0' in debug_info:
            self._save_coordinate_system(
                image, debug_info, part_name, cheek_side, session_id
            )

        if 'cheek_local' in debug_info:
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

    def _draw_cheek_debug_image(
        self,
        image: np.ndarray,
        debug_info: dict,
        landmarks: "FaceLandmarks",
        part_name: str,
        cheek_side: str,
        session_id: str
    ) -> None:
        """Draw comprehensive cheek debug visualization."""
        h, w = image.shape[:2]
        debug_image = image.copy()

        cx = int(debug_info.get('cx', 0))
        cy = int(debug_info.get('cy', 0))
        rx = int(debug_info.get('rx', 0))
        ry = int(debug_info.get('ry', 0))

        # Draw ellipse center and boundary
        cv2.circle(debug_image, (cx, cy), 8, (0, 255, 0), -1)
        cv2.ellipse(debug_image, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 0), 2)

        # Draw coordinate axes if available
        if 'ex' in debug_info and 'ey' in debug_info:
            ex = debug_info['ex']
            ey = debug_info['ey']
            scale = 50

            ex_end = (int(cx + ex[0] * scale), int(cy + ex[1] * scale))
            ey_end = (int(cx + ey[0] * scale), int(cy + ey[1] * scale))

            cv2.arrowedLine(debug_image, (cx, cy), ex_end, (255, 0, 0), 2, tipLength=0.2)
            cv2.arrowedLine(debug_image, (cx, cy), ey_end, (0, 128, 255), 2, tipLength=0.2)

        # Draw info text
        info_lines = [
            f"Part: {part_name}",
            f"Center: ({cx}, {cy})",
            f"Radius: ({rx}, {ry})",
        ]
        self._draw_text_with_background(debug_image, info_lines, (10, 30))

        path = os.path.join(
            self.debug_dir,
            f"{session_id}_4_cheek_3d_{part_name}.png"
        )
        cv2.imwrite(path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

    def _save_ellipse_mask_heatmap(
        self,
        image: np.ndarray,
        ellipse_mask: np.ndarray,
        part_name: str,
        session_id: str
    ) -> None:
        """Save ellipse mask as heatmap overlay."""
        mask_vis = (ellipse_mask * 255).astype(np.uint8)
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

    def _save_coordinate_system(
        self,
        image: np.ndarray,
        debug_info: dict,
        part_name: str,
        cheek_side: str,
        session_id: str
    ) -> None:
        """Save coordinate system visualization."""
        h, w = image.shape[:2]
        viz = np.ones((h, w, 3), dtype=np.uint8) * 255
        overlay = cv2.addWeighted(image, 0.3, viz, 0.7, 0)

        p0 = debug_info['p0']
        e_u = debug_info['e_u']
        e_v = debug_info['e_v']
        cx = int(debug_info['cx'])
        cy = int(debug_info['cy'])

        # Draw origin point
        p0_2d = (int(p0[0] * w), int(p0[1] * h))
        cv2.circle(overlay, p0_2d, 10, (0, 0, 255), -1)

        # Draw basis vectors
        scale = 0.05
        p_u_end = (int((p0[0] + e_u[0] * scale) * w), int((p0[1] + e_u[1] * scale) * h))
        p_v_end = (int((p0[0] + e_v[0] * scale) * w), int((p0[1] + e_v[1] * scale) * h))

        cv2.arrowedLine(overlay, p0_2d, p_u_end, (255, 0, 0), 3, tipLength=0.2)
        cv2.arrowedLine(overlay, p0_2d, p_v_end, (0, 0, 255), 3, tipLength=0.2)

        # Draw ellipse center
        cv2.circle(overlay, (cx, cy), 8, (0, 255, 0), -1)

        path = os.path.join(
            self.debug_dir,
            f"{session_id}_4_cheek_3d_coords_{part_name}.png"
        )
        cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def _save_uv_visualization(
        self,
        image: np.ndarray,
        debug_info: dict,
        params: "CheekParams",
        part_name: str,
        cheek_side: str,
        session_id: str
    ) -> None:
        """Save UV coordinate space visualization using matplotlib."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
        except ImportError:
            logger.warning("matplotlib not available, skipping UV visualization")
            return

        cheek_local = debug_info['cheek_local']
        u_coords = cheek_local[:, 0]
        v_coords = cheek_local[:, 1]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot cheek landmarks in UV space
        ax.scatter(u_coords, v_coords, c='blue', s=50, label='Cheek landmarks')

        # Plot ellipse
        u0 = debug_info.get('u0', 0)
        v0 = debug_info.get('v0', 0)
        a = debug_info.get('a', 0.1)
        b = debug_info.get('b', 0.1)

        ellipse = Ellipse(
            (u0, v0), 2 * a, 2 * b,
            fill=False, edgecolor='green', linewidth=2, label='Ellipse'
        )
        ax.add_patch(ellipse)
        ax.plot(u0, v0, 'go', markersize=10, label='Ellipse center')

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('u (local coordinate)')
        ax.set_ylabel('v (local coordinate)')
        ax.set_title(f'UV Space - {part_name} ({cheek_side})')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        path = os.path.join(
            self.debug_dir,
            f"{session_id}_4_cheek_3d_uv_{part_name}.png"
        )
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

    def _draw_text_with_background(
        self,
        image: np.ndarray,
        lines: list,
        position: tuple,
        font_scale: float = 0.5
    ) -> None:
        """Draw text with black outline for visibility."""
        x, y = position
        for i, line in enumerate(lines):
            pos = (x, y + i * 25)
            # Black outline
            cv2.putText(
                image, line, pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 3, cv2.LINE_AA
            )
            # White text
            cv2.putText(
                image, line, pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA
            )
