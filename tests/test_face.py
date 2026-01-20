"""
Tests for Face class and MakeupLayer classes.
"""
import pytest
import numpy as np
import cv2
import glob

from src.face_makeup.processor import MakeupProcessor
from src.face_makeup.face import Face, LipLayer, CheekLayer, EyeShadowLayer, EyeLinerLayer
from src.face_makeup.core.types import (
    LipsParams,
    CheekParams,
    EyeShadowParams,
    EyeLinerParams,
    ColorParams,
)

@pytest.fixture
def processor():
    """MakeupProcessor instance."""
    return MakeupProcessor()


@pytest.fixture
def test_image():
    """Load first test image."""
    image_files = sorted(glob.glob("tests/test_images/*.png"))
    assert len(image_files) > 0, "No test images found"
    image = cv2.imread(image_files[0])
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
        assert landmarks.points.shape[0] == 478  # MediaPipe refined landmarks

    def test_makeup_layers_initially_empty(self, face):
        """Test that makeup_layers is initially empty."""
        assert len(face.makeup_layers) == 0


class TestFaceFluentAPI:
    """Tests for Face fluent API."""

    def test_lip_returns_self(self, face):
        """Test that lip() returns self."""
        params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        result = face.lip(params)
        assert result is face

    def test_cheek_returns_self(self, face):
        """Test that cheek() returns self."""
        params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2),
            falloff=0.4
        )
        result = face.cheek(params)
        assert result is face

    def test_eyelid_returns_self(self, face):
        """Test that eyelid() returns self."""
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(180, 130, 150)),
            opacity=0.3
        )
        result = face.eyelid(params)
        assert result is face

    def test_chaining(self, face):
        """Test method chaining."""
        lip_params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        cheek_params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2),
            falloff=0.4
        )
        eye_params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(180, 130, 150)),
            opacity=0.3
        )

        result = face.lip(lip_params).cheek(cheek_params).eyelid(eye_params)
        assert result is face


class TestFaceMakeupLayers:
    """Tests for makeup layer management."""

    def test_lip_adds_layer(self, face):
        """Test that lip() adds a LipLayer."""
        params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        face.lip(params)

        layers = face.makeup_layers
        assert len(layers) == 1
        assert isinstance(layers[0], LipLayer)
        assert layers[0].part_name == 'lips'

    def test_cheek_both_adds_two_layers(self, face):
        """Test that cheek(side='both') adds two layers."""
        params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2),
            falloff=0.4
        )
        face.cheek(params, side='both')

        layers = face.makeup_layers
        assert len(layers) == 2
        assert all(isinstance(l, CheekLayer) for l in layers)
        part_names = [l.part_name for l in layers]
        assert 'cheek_left' in part_names
        assert 'cheek_right' in part_names

    def test_cheek_left_adds_one_layer(self, face):
        """Test that cheek(side='left') adds one layer."""
        params = CheekParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5,
            center=(0.0, 0.0),
            radius=(0.4, 0.2),
            falloff=0.4
        )
        face.cheek(params, side='left')

        layers = face.makeup_layers
        assert len(layers) == 1
        assert layers[0].part_name == 'cheek_left'

    def test_eyelid_both_adds_two_layers(self, face):
        """Test that eyelid(side='both') adds two layers."""
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(180, 130, 150)),
            opacity=0.3
        )
        face.eyelid(params, side='both')

        layers = face.makeup_layers
        assert len(layers) == 2
        part_names = [l.part_name for l in layers]
        assert 'eyelid_left' in part_names
        assert 'eyelid_right' in part_names

    def test_eyeliner_both_adds_two_layers(self, face):
        """Test that eyeliner(side='both') adds two layers."""
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=2
        )
        face.eyeliner(params, side='both')

        layers = face.makeup_layers
        assert len(layers) == 2
        assert all(isinstance(l, EyeLinerLayer) for l in layers)
        part_names = [l.part_name for l in layers]
        assert 'eyeliner_left' in part_names
        assert 'eyeliner_right' in part_names

    def test_eyeliner_left_adds_one_layer(self, face):
        """Test that eyeliner(side='left') adds one layer."""
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=2
        )
        face.eyeliner(params, side='left')

        layers = face.makeup_layers
        assert len(layers) == 1
        assert layers[0].part_name == 'eyeliner_left'

    def test_eyeliner_returns_self(self, face):
        """Test that eyeliner() returns self."""
        params = EyeLinerParams(
            color_params=ColorParams(target_rgb=(0, 0, 0)),
            opacity=1.0,
            thickness=2
        )
        result = face.eyeliner(params)
        assert result is face

    def test_clear_removes_all_layers(self, face):
        """Test that clear() removes all layers."""
        params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        face.lip(params)
        assert len(face.makeup_layers) == 1

        face.clear()
        assert len(face.makeup_layers) == 0

    def test_clear_returns_self(self, face):
        """Test that clear() returns self."""
        result = face.clear()
        assert result is face



class TestMakeupLayerReturnsListCopy:
    """Tests that makeup_layers returns a copy."""

    def test_makeup_layers_returns_copy(self, face):
        """Test that makeup_layers returns a copy, not the internal list."""
        params = LipsParams(
            color_params=ColorParams(target_rgb=(255, 100, 100)),
            opacity=0.5
        )
        face.lip(params)

        layers1 = face.makeup_layers
        layers2 = face.makeup_layers

        # Should be equal but not same object
        assert layers1 == layers2
        assert layers1 is not layers2


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
        assert len(face.makeup_layers) == 1

    def test_valid_color_boundary_255(self, face):
        """Test that color value of 255 is accepted."""
        face.lip(color=(255, 255, 255))
        assert len(face.makeup_layers) == 1


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
        assert len(face.makeup_layers) == 1

    def test_valid_opacity_boundary_one(self, face):
        """Test that opacity of 1.0 is accepted."""
        face.lip(color=(255, 100, 100), opacity=1.0)
        assert len(face.makeup_layers) == 1

    def test_valid_opacity_integer(self, face):
        """Test that integer opacity (0 or 1) is accepted."""
        face.lip(color=(255, 100, 100), opacity=1)
        assert len(face.makeup_layers) == 1
