"""
Tests for Face class.
"""
import pytest
import numpy as np
from face_makeup.processor import MakeupProcessor
from face_makeup.face import Face
from face_makeup.core.types import (
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams,
    ColorParams,
    EyeIrisParams,
    PartParams
)
from face_makeup.custom_landmark import NUM_TOTAL


@pytest.fixture
def processor():
    """MakeupProcessor instance."""
    return MakeupProcessor()


@pytest.fixture
def face(processor, test_image):
    """Face object from test image."""
    ctx = processor.open(test_image)
    return ctx.faces[0]


class TestFaceProperties:
    """Tests for Face properties."""

    def test_face_id(self, face):
        """Test face_id property."""
        assert face.face_id == 0

    def test_bbox(self, face):
        """Test bbox property."""
        bbox = face.bbox
        assert bbox is not None
        assert bbox.x >= 0
        assert bbox.y >= 0
        assert bbox.w > 0
        assert bbox.h > 0

    def test_landmarks(self, face):
        """Test landmarks property."""
        landmarks = face.landmarks
        assert landmarks is not None
        assert landmarks.points.shape[0] == NUM_TOTAL  # base 478 + custom landmarks

    def test_makeup_passes_initially_empty(self, face):
        """Test that makeup_passes is initially empty."""
        assert len(face.makeup_passes) == 0


class TestFaceFluentAPI:
    """Tests for Face fluent API."""

    def test_lip_returns_self(self, face):
        """Test that lip() returns self."""
        result = face.lip(color=(255, 100, 100))
        assert result is face

    def test_cheek_returns_self(self, face):
        """Test that cheek() returns self."""
        result = face.cheek(
            color=(255, 100, 100),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2)
        )
        assert result is face

    def test_eyelid_returns_self(self, face):
        """Test that eyelid() returns self."""
        result = face.eyelid(color=(180, 130, 150), opacity=0.3)
        assert result is face

    def test_chaining(self, face):
        """Test method chaining."""
        result = (
            face.lip(color=(255, 100, 100), opacity=0.5)
            .cheek(color=(255, 100, 100), opacity=0.5, center=(0.0, 0.0), radius=(0.4, 0.2))
            .eyelid(color=(180, 130, 150), opacity=0.3)
        )
        assert result is face


class TestFaceMakeupPasses:
    """Tests for makeup pass management."""

    def test_lip_adds_pass(self, face):
        """Test that lip() adds a lips pass."""
        face.lip(color=(255, 100, 100), opacity=0.5)

        passes = face.makeup_passes
        assert len(passes) == 1
        assert passes[0].pass_type == 'lips'

    def test_cheek_both_adds_two_passes(self, face):
        """Test that cheek(side='both') adds two passes."""
        face.cheek(
            color=(255, 100, 100),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2),
            side='both'
        )

        passes = face.makeup_passes
        assert len(passes) == 2
        pass_types = [p.pass_type for p in passes]
        assert 'cheek_left' in pass_types
        assert 'cheek_right' in pass_types

    def test_cheek_left_adds_one_pass(self, face):
        """Test that cheek(side='left') adds one pass."""
        face.cheek(
            color=(255, 100, 100),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2),
            side='left'
        )

        passes = face.makeup_passes
        assert len(passes) == 1
        assert passes[0].pass_type == 'cheek_left'

    def test_eyelid_both_adds_two_passes(self, face):
        """Test that eyelid(side='both') adds two passes."""
        face.eyelid(color=(180, 130, 150), opacity=0.3, side='both')

        passes = face.makeup_passes
        assert len(passes) == 2
        pass_types = [p.pass_type for p in passes]
        assert 'eyelid_left' in pass_types
        assert 'eyelid_right' in pass_types

    def test_eyeliner_both_adds_two_passes(self, face):
        """Test that eyeliner(side='both') adds two passes."""
        face.eyeliner(color=(0, 0, 0), opacity=1.0, thickness=2, side='both')

        passes = face.makeup_passes
        assert len(passes) == 2
        pass_types = [p.pass_type for p in passes]
        assert 'eyeliner_left' in pass_types
        assert 'eyeliner_right' in pass_types

    def test_eyeliner_left_adds_one_pass(self, face):
        """Test that eyeliner(side='left') adds one pass."""
        face.eyeliner(color=(0, 0, 0), opacity=1.0, thickness=2, side='left')

        passes = face.makeup_passes
        assert len(passes) == 1
        assert passes[0].pass_type == 'eyeliner_left'

    def test_eyeliner_returns_self(self, face):
        """Test that eyeliner() returns self."""
        result = face.eyeliner(color=(0, 0, 0), opacity=1.0, thickness=2)
        assert result is face

    def test_eyeiris_returns_self(self, face):
        """Test that eyeiris() returns self."""
        result = face.eyeiris(color=(100, 150, 255), opacity=0.5)
        assert result is face

    def test_eyeiris_both_adds_two_passes(self, face):
        """Test that eyeiris(side='both') adds two passes."""
        face.eyeiris(color=(100, 150, 255), opacity=0.5, side='both')

        passes = face.makeup_passes
        assert len(passes) == 2
        pass_types = [p.pass_type for p in passes]
        assert 'eyeiris_left' in pass_types
        assert 'eyeiris_right' in pass_types

    def test_eyeiris_left_adds_one_pass(self, face):
        """Test that eyeiris(side='left') adds one pass."""
        face.eyeiris(color=(100, 150, 255), opacity=0.5, side='left')

        passes = face.makeup_passes
        assert len(passes) == 1
        assert passes[0].pass_type == 'eyeiris_left'

    def test_clear_removes_all_passes(self, face):
        """Test that clear() removes all passes."""
        face.lip(color=(255, 100, 100), opacity=0.5)
        assert len(face.makeup_passes) == 1

        face.clear()
        assert len(face.makeup_passes) == 0

    def test_clear_returns_self(self, face):
        """Test that clear() returns self."""
        result = face.clear()
        assert result is face


class TestMakeupPassesReturnsCopy:
    """Tests that makeup_passes returns a copy."""

    def test_makeup_passes_returns_copy(self, face):
        """Test that makeup_passes returns a copy, not the internal list."""
        face.lip(color=(255, 100, 100), opacity=0.5)

        passes1 = face.makeup_passes
        passes2 = face.makeup_passes

        assert passes1 is not passes2


class TestColorValidation:
    """Tests for color parameter validation."""

    def test_lip_invalid_color_not_tuple(self, face):
        """Test that lip() raises error for non-tuple color."""
        with pytest.raises(ValueError, match="Color must be"):
            face.lip(color=[255, 100, 100])

    def test_lip_invalid_color_wrong_length(self, face):
        """Test that lip() raises error for wrong length color tuple."""
        with pytest.raises(ValueError, match="Color must be"):
            face.lip(color=(255, 100))

    def test_lip_invalid_color_out_of_range(self, face):
        """Test that lip() raises error for out of range color value."""
        with pytest.raises(ValueError, match="Color values must be"):
            face.lip(color=(256, 100, 100))

    def test_lip_invalid_color_negative(self, face):
        """Test that lip() raises error for negative color value."""
        with pytest.raises(ValueError, match="Color values must be"):
            face.lip(color=(-1, 100, 100))

    def test_lip_invalid_color_float(self, face):
        """Test that lip() raises error for float color value."""
        with pytest.raises(ValueError, match="Color values must be"):
            face.lip(color=(255.5, 100.0, 100.0))

    def test_cheek_invalid_color(self, face):
        """Test that cheek() raises error for invalid color."""
        with pytest.raises(ValueError, match="Color must be"):
            face.cheek(color="red")

    def test_eyelid_invalid_color(self, face):
        """Test that eyelid() raises error for invalid color."""
        with pytest.raises(ValueError, match="Color values must be"):
            face.eyelid(color=(300, 100, 100))

    def test_eyeliner_invalid_color(self, face):
        """Test that eyeliner() raises error for invalid color."""
        with pytest.raises(ValueError, match="Color values must be"):
            face.eyeliner(color=(100, -50, 100))

    def test_valid_color_boundary_zero(self, face):
        """Test that color value of 0 is accepted."""
        face.lip(color=(0, 0, 0))
        assert len(face.makeup_passes) == 1

    def test_valid_color_boundary_255(self, face):
        """Test that color value of 255 is accepted."""
        face.lip(color=(255, 255, 255))
        assert len(face.makeup_passes) == 1


class TestOpacityValidation:
    """Tests for opacity parameter validation."""

    def test_lip_invalid_opacity_too_high(self, face):
        """Test that lip() raises error for opacity > 1.0."""
        with pytest.raises(ValueError, match="Opacity must be"):
            face.lip(color=(255, 100, 100), opacity=1.5)

    def test_lip_invalid_opacity_negative(self, face):
        """Test that lip() raises error for negative opacity."""
        with pytest.raises(ValueError, match="Opacity must be"):
            face.lip(color=(255, 100, 100), opacity=-0.1)

    def test_lip_invalid_opacity_string(self, face):
        """Test that lip() raises error for string opacity."""
        with pytest.raises(ValueError, match="Opacity must be"):
            face.lip(color=(255, 100, 100), opacity="0.5")

    def test_cheek_invalid_opacity(self, face):
        """Test that cheek() raises error for invalid opacity."""
        with pytest.raises(ValueError, match="Opacity must be"):
            face.cheek(color=(255, 180, 180), opacity=2.0)

    def test_eyelid_invalid_opacity(self, face):
        """Test that eyelid() raises error for invalid opacity."""
        with pytest.raises(ValueError, match="Opacity must be"):
            face.eyelid(color=(180, 130, 150), opacity=-1.0)

    def test_eyeliner_invalid_opacity(self, face):
        """Test that eyeliner() raises error for invalid opacity."""
        with pytest.raises(ValueError, match="Opacity must be"):
            face.eyeliner(color=(0, 0, 0), opacity=100)

    def test_valid_opacity_boundary_zero(self, face):
        """Test that opacity of 0.0 is accepted."""
        face.lip(color=(255, 100, 100), opacity=0.0)
        assert len(face.makeup_passes) == 1

    def test_valid_opacity_boundary_one(self, face):
        """Test that opacity of 1.0 is accepted."""
        face.lip(color=(255, 100, 100), opacity=1.0)
        assert len(face.makeup_passes) == 1

    def test_valid_opacity_integer(self, face):
        """Test that integer opacity (0 or 1) is accepted."""
        face.lip(color=(255, 100, 100), opacity=1)
        assert len(face.makeup_passes) == 1


class TestBlurStrengthValidation:
    """Tests for blur_strength parameter validation."""

    def test_lip_invalid_blur_strength_too_high(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.lip(color=(255, 100, 100), blur_strength=1.5)

    def test_lip_invalid_blur_strength_negative(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.lip(color=(255, 100, 100), blur_strength=-0.1)

    def test_lip_invalid_blur_strength_string(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.lip(color=(255, 100, 100), blur_strength="0.5")

    def test_cheek_invalid_blur_strength(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.cheek(color=(255, 180, 180), blur_strength=2.0)

    def test_eyelid_invalid_blur_strength(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.eyelid(color=(180, 130, 150), blur_strength=-1.0)

    def test_eyelid_edge_invalid_blur_strength(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.eyelid_edge(color=(245, 41, 41), blur_strength=1.1)

    def test_eyeliner_invalid_blur_strength(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.eyeliner(color=(0, 0, 0), blur_strength=5.0)

    def test_eyeiris_invalid_blur_strength(self, face):
        with pytest.raises(ValueError, match="blur_strength must be"):
            face.eyeiris(color=(100, 150, 255), blur_strength=-0.5)

    def test_valid_blur_strength_boundary_zero(self, face):
        face.lip(color=(255, 100, 100), blur_strength=0.0)
        assert len(face.makeup_passes) == 1

    def test_valid_blur_strength_boundary_one(self, face):
        face.lip(color=(255, 100, 100), blur_strength=1.0)
        assert len(face.makeup_passes) == 1

    def test_valid_blur_strength_integer(self, face):
        face.lip(color=(255, 100, 100), blur_strength=1)
        assert len(face.makeup_passes) == 1


class TestPartParamsValidation:
    """Tests for PartParams base class __post_init__ validation."""

    def test_invalid_opacity_via_dataclass(self):
        with pytest.raises(ValueError, match="opacity must be"):
            LipsParams(
                color_params=ColorParams(target_rgb=(255, 100, 100)),
                opacity=1.5
            )

    def test_invalid_blur_strength_via_dataclass(self):
        with pytest.raises(ValueError, match="blur_strength must be"):
            LipsParams(
                color_params=ColorParams(target_rgb=(255, 100, 100)),
                opacity=0.5,
                blur_strength=2.0
            )

    def test_negative_opacity_via_dataclass(self):
        with pytest.raises(ValueError, match="opacity must be"):
            EyeShadowParams(
                color_params=ColorParams(target_rgb=(180, 130, 150)),
                opacity=-0.1
            )

    def test_negative_blur_strength_via_dataclass(self):
        with pytest.raises(ValueError, match="blur_strength must be"):
            CheekParams(
                color_params=ColorParams(target_rgb=(255, 180, 180)),
                opacity=0.3,
                blur_strength=-0.5,
                center=(0.0, 0.0),
                radius=(0.5, 0.5)
            )


class TestCheekParamsValidation:
    """Tests for CheekParams center/radius validation."""

    def test_center_out_of_range_positive(self):
        with pytest.raises(ValueError, match="center values must be in range"):
            CheekParams(
                color_params=ColorParams(target_rgb=(255, 180, 180)),
                opacity=0.3,
                center=(1.5, 0.0),
                radius=(0.5, 0.5),
            )

    def test_center_out_of_range_negative(self):
        with pytest.raises(ValueError, match="center values must be in range"):
            CheekParams(
                color_params=ColorParams(target_rgb=(255, 180, 180)),
                opacity=0.3,
                center=(0.0, -1.5),
                radius=(0.5, 0.5),
            )

    def test_radius_zero_rejected(self):
        with pytest.raises(ValueError, match="radius values must be in range"):
            CheekParams(
                color_params=ColorParams(target_rgb=(255, 180, 180)),
                opacity=0.3,
                center=(0.0, 0.0),
                radius=(0.0, 0.5),
            )

    def test_radius_negative_rejected(self):
        with pytest.raises(ValueError, match="radius values must be in range"):
            CheekParams(
                color_params=ColorParams(target_rgb=(255, 180, 180)),
                opacity=0.3,
                center=(0.0, 0.0),
                radius=(0.5, -0.1),
            )

    def test_radius_over_one_rejected(self):
        with pytest.raises(ValueError, match="radius values must be in range"):
            CheekParams(
                color_params=ColorParams(target_rgb=(255, 180, 180)),
                opacity=0.3,
                center=(0.0, 0.0),
                radius=(1.5, 0.5),
            )

    def test_valid_params(self):
        params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 180, 180)),
            opacity=0.3,
            center=(-0.5, 0.8),
            radius=(0.3, 1.0),
        )
        assert params.center == (-0.5, 0.8)
        assert params.radius == (0.3, 1.0)

    def test_boundary_values(self):
        params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 180, 180)),
            opacity=0.3,
            center=(-1.0, 1.0),
            radius=(0.01, 1.0),
        )
        assert params.center == (-1.0, 1.0)
        assert params.radius == (0.01, 1.0)


class TestEyeLinerParamsValidation:
    """Tests for EyeLinerParams thickness validation."""

    def test_thickness_zero_rejected(self):
        with pytest.raises(ValueError, match="thickness must be a positive integer"):
            EyeLinerParams(
                color_params=ColorParams(target_rgb=(0, 0, 0)),
                opacity=1.0,
                thickness=0
            )

    def test_thickness_negative_rejected(self):
        with pytest.raises(ValueError, match="thickness must be a positive integer"):
            EyeLinerParams(
                color_params=ColorParams(target_rgb=(0, 0, 0)),
                opacity=1.0,
                thickness=-1
            )

    def test_thickness_float_rejected(self):
        with pytest.raises(ValueError, match="thickness must be a positive integer"):
            EyeLinerParams(
                color_params=ColorParams(target_rgb=(0, 0, 0)),
                opacity=1.0,
                thickness=1.5
            )

    def test_thickness_valid(self):
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=3
        )
        assert params.thickness == 3
