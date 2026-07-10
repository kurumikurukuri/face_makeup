"""
Tests for ellipse mask debug_info dictionary keys.

Verifies the new naming conventions for coordinate systems.
"""
import pytest
import numpy as np
from face_makeup.constants import CHEEK_LEFT_INDICES, CHEEK_RIGHT_INDICES
from face_makeup.core.passes.cheek import CheekMaskStrategy
from face_makeup.core.types import CheekParams, ColorParams, FaceLandmarks


class TestEllipseMaskDebugInfo:
    """Test debug_info dictionary returned by CheekMaskStrategy.create_cheek_mask."""

    @pytest.fixture
    def cheek_params(self):
        """Create default cheek parameters."""
        return CheekParams(
            color_params=ColorParams(target_rgb=(255, 150, 150)),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.5, 0.5)
        )

    @pytest.fixture
    def strategy(self, cheek_params):
        """Create CheekMaskStrategy instance (geometry now lives here)."""
        return CheekMaskStrategy(list(CHEEK_LEFT_INDICES), 'left', cheek_params)

    @pytest.fixture
    def base_mask(self):
        """Create a simple base mask."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0
        return mask

    @pytest.fixture
    def landmarks_3d(self):
        """Create mock FaceLandmarks with 3D coordinates."""
        num_landmarks = 478
        points_2d = np.random.rand(num_landmarks, 2).astype(np.float32)
        points_3d = np.random.rand(num_landmarks, 3).astype(np.float32)
        points_2d = points_2d * 0.5 + 0.25
        points_3d = points_3d * 0.5 + 0.25
        return FaceLandmarks(points=points_2d, points_3d=points_3d)

    @pytest.fixture
    def landmarks_2d_only(self):
        """Create mock FaceLandmarks without 3D coordinates."""
        num_landmarks = 478
        points_2d = np.random.rand(num_landmarks, 2).astype(np.float32)
        points_2d = points_2d * 0.5 + 0.25
        return FaceLandmarks(points=points_2d, points_3d=None)

    def test_debug_info_2d_mode(self, strategy, cheek_params, base_mask, landmarks_2d_only):
        """Test debug_info in 2D mode (no 3D landmarks)."""
        result = strategy.create_cheek_mask(
            image_shape=(100, 100),
            base_mask=base_mask,
            params=cheek_params,
            landmarks=landmarks_2d_only,
            cheek_side='left',
            return_debug_info=True
        )

        assert isinstance(result, tuple)
        mask, debug_info = result
        assert debug_info.get('mode') == '2d'

    def test_debug_info_3d_mode_keys(self, strategy, cheek_params, base_mask, landmarks_3d):
        """Test that 3D mode debug_info contains all required keys."""
        result = strategy.create_cheek_mask(
            image_shape=(100, 100),
            base_mask=base_mask,
            params=cheek_params,
            landmarks=landmarks_3d,
            cheek_side='left',
            return_debug_info=True
        )

        assert isinstance(result, tuple)
        mask, debug_info = result

        required_keys = [
            # Tangent plane coordinate system
            'tangent_origin_face3d',
            'tangent_basis_u',
            'tangent_basis_v',
            'tangent_basis_w',

            # Image pixel space
            'ellipse_img_px_center_x',
            'ellipse_img_px_center_y',
            'ellipse_img_px_radius_x',
            'ellipse_img_px_radius_y',

            # Projected tangent axes
            'img_u_axis_unit',
            'img_v_axis_unit',

            # Tangent space ranges
            'tangent_u_min',
            'tangent_u_max',
            'tangent_v_min',
            'tangent_v_max',
            'tangent_u_center',
            'tangent_v_center',
            'tangent_u_halfspan',
            'tangent_v_halfspan',

            # Ellipse in tangent space
            'ellipse_tangent_center_u',
            'ellipse_tangent_center_v',
            'ellipse_tangent_radius_u',
            'ellipse_tangent_radius_v',

            # Surface coefficients
            'surface_coeffs',

            # Visualization data
            'ellipse_face3d_points',
            'cheek_tangent_uvw',
            'cheek_img_norm',
        ]

        for key in required_keys:
            assert key in debug_info, f"Missing key: '{key}'"

    def test_surface_coeffs_shape(self, strategy, cheek_params, base_mask, landmarks_3d):
        """Test that surface_coeffs has correct shape for quadratic surface."""
        result = strategy.create_cheek_mask(
            image_shape=(100, 100),
            base_mask=base_mask,
            params=cheek_params,
            landmarks=landmarks_3d,
            cheek_side='left',
            return_debug_info=True
        )

        mask, debug_info = result
        surface_coeffs = debug_info['surface_coeffs']

        assert isinstance(surface_coeffs, np.ndarray)
        assert surface_coeffs.shape == (6,)

    def test_ellipse_face3d_points_shape(self, strategy, cheek_params, base_mask, landmarks_3d):
        """Test that ellipse_face3d_points has correct structure."""
        result = strategy.create_cheek_mask(
            image_shape=(100, 100),
            base_mask=base_mask,
            params=cheek_params,
            landmarks=landmarks_3d,
            cheek_side='left',
            return_debug_info=True
        )

        mask, debug_info = result
        ellipse_face3d = debug_info['ellipse_face3d_points']

        assert isinstance(ellipse_face3d, np.ndarray)
        assert len(ellipse_face3d.shape) == 2
        assert ellipse_face3d.shape[1] == 3

    def test_tangent_basis_orthogonal(self, strategy, cheek_params, base_mask, landmarks_3d):
        """Test that tangent basis vectors are approximately orthogonal."""
        result = strategy.create_cheek_mask(
            image_shape=(100, 100),
            base_mask=base_mask,
            params=cheek_params,
            landmarks=landmarks_3d,
            cheek_side='left',
            return_debug_info=True
        )

        mask, debug_info = result

        u = debug_info['tangent_basis_u']
        v = debug_info['tangent_basis_v']
        w = debug_info['tangent_basis_w']

        # Check orthogonality (dot products should be close to 0)
        assert abs(np.dot(u, v)) < 0.1, "U and V should be orthogonal"
        assert abs(np.dot(u, w)) < 0.1, "U and W should be orthogonal"
        assert abs(np.dot(v, w)) < 0.1, "V and W should be orthogonal"

    @pytest.mark.parametrize("cheek_side", ['left', 'right'])
    def test_both_cheeks(self, cheek_params, base_mask, landmarks_3d, cheek_side):
        """Test debug_info works for both left and right cheeks."""
        indices = CHEEK_LEFT_INDICES if cheek_side == 'left' else CHEEK_RIGHT_INDICES
        s = CheekMaskStrategy(list(indices), cheek_side, cheek_params)
        result = s.create_cheek_mask(
            image_shape=(100, 100),
            base_mask=base_mask,
            params=cheek_params,
            landmarks=landmarks_3d,
            cheek_side=cheek_side,
            return_debug_info=True
        )

        assert isinstance(result, tuple)
        mask, debug_info = result

        assert 'surface_coeffs' in debug_info
        assert 'ellipse_face3d_points' in debug_info
        assert 'cheek_tangent_uvw' in debug_info
