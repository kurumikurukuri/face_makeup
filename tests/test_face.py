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
    PartParams,
    TearBagHighlightParams,
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


class TestGlitterValidation:
    """Tests for glitter parameter validation on eyelid()."""

    def test_eyelid_glitter_too_high(self, face):
        with pytest.raises(ValueError, match="glitter must be"):
            face.eyelid(color=(180, 130, 150), glitter=1.5)

    def test_eyelid_glitter_negative(self, face):
        with pytest.raises(ValueError, match="glitter must be"):
            face.eyelid(color=(180, 130, 150), glitter=-0.1)

    def test_eyelid_glitter_string(self, face):
        with pytest.raises(ValueError, match="glitter must be"):
            face.eyelid(color=(180, 130, 150), glitter="0.5")

    def test_valid_glitter_boundary_zero(self, face):
        face.eyelid(color=(180, 130, 150), glitter=0.0, side='left')
        assert len(face.makeup_passes) == 1

    def test_valid_glitter_boundary_one(self, face):
        face.eyelid(color=(180, 130, 150), glitter=1.0, side='left')
        assert len(face.makeup_passes) == 1

    def test_valid_glitter_integer(self, face):
        face.eyelid(color=(180, 130, 150), glitter=0, side='left')
        assert len(face.makeup_passes) == 1

    def test_glitter_default_is_zero(self, face):
        """eyelid() with no glitter arg should default to 0.0."""
        face.eyelid(color=(180, 130, 150))
        params = face.makeup_passes[0].params
        assert params.glitter == 0.0

    def test_glitter_propagates_to_params(self, face):
        """Verify glitter value reaches EyeShadowParams."""
        face.eyelid(color=(180, 130, 150), glitter=0.7)
        params = face.makeup_passes[0].params
        assert params.glitter == 0.7


class TestEyeShadowParamsGlitterValidation:
    """Tests for EyeShadowParams glitter validation at the dataclass level."""

    def test_glitter_too_high_via_dataclass(self):
        with pytest.raises(ValueError, match="glitter must be"):
            EyeShadowParams(
                color_params=ColorParams(target_rgb=(180, 130, 150)),
                opacity=0.3,
                glitter=1.5,
            )

    def test_glitter_negative_via_dataclass(self):
        with pytest.raises(ValueError, match="glitter must be"):
            EyeShadowParams(
                color_params=ColorParams(target_rgb=(180, 130, 150)),
                opacity=0.3,
                glitter=-0.1,
            )

    def test_glitter_default_is_zero_via_dataclass(self):
        params = EyeShadowParams(
            color_params=ColorParams(target_rgb=(180, 130, 150)),
            opacity=0.3,
        )
        assert params.glitter == 0.0


class TestApplyGlitter:
    """Tests for apply_glitter() function in blend_utils."""

    @pytest.fixture
    def glitter_image(self):
        """Synthetic image for glitter testing — solid mid-tone."""
        return np.full((100, 100, 3), 128, dtype=np.uint8)

    @pytest.fixture
    def glitter_mask(self):
        """Float32 mask covering the centre quarter."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[25:75, 25:75] = 1.0
        return mask

    def test_glitter_zero_returns_unchanged(self, glitter_image, glitter_mask):
        """glitter_intensity=0 returns the input image unchanged (same array content)."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        result = apply_glitter(
            glitter_image, glitter_mask,
            glitter_intensity=0.0,
            rng=np.random.default_rng(42),
        )
        assert np.array_equal(result, glitter_image)

    def test_glitter_positive_returns_different(self, glitter_image, glitter_mask):
        """glitter_intensity>0 produces a different image."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        result = apply_glitter(
            glitter_image, glitter_mask,
            glitter_intensity=0.5,
            rng=np.random.default_rng(42),
        )
        assert not np.array_equal(result, glitter_image)

    def test_glitter_positive_same_shape(self, glitter_image, glitter_mask):
        """glitter output has the same shape as input."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        result = apply_glitter(
            glitter_image, glitter_mask,
            glitter_intensity=0.8,
            rng=np.random.default_rng(42),
        )
        assert result.shape == glitter_image.shape
        assert result.dtype == glitter_image.dtype

    def test_glitter_positive_deterministic(self, glitter_image, glitter_mask):
        """Same seed produces identical output."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = apply_glitter(
            glitter_image, glitter_mask,
            glitter_intensity=0.5,
            rng=rng1,
        )
        result2 = apply_glitter(
            glitter_image, glitter_mask,
            glitter_intensity=0.5,
            rng=rng2,
        )
        assert np.array_equal(result1, result2)

    def test_glitter_empty_mask_returns_unchanged(self, glitter_image):
        """Empty mask returns image unchanged even with positive intensity."""
        from face_makeup.core.blend.blend_utils import apply_glitter

        empty_mask = np.zeros((100, 100), dtype=np.float32)
        result = apply_glitter(
            glitter_image, empty_mask,
            glitter_intensity=0.5,
            rng=np.random.default_rng(42),
        )
        assert np.array_equal(result, glitter_image)


class TestEyelidZones:
    """Tests for the `zone` argument on eyelid()."""

    @pytest.mark.parametrize("zone", ["outer", "center", "inner"])
    def test_eyelid_zone_both_sides(self, face, zone):
        """eyelid(zone=...) builds zone-qualified part names for both sides."""
        face.eyelid(color=(180, 130, 150), zone=zone, side='both')
        pass_types = [p.pass_type for p in face.makeup_passes]
        assert pass_types == [f"eyelid_{zone}_left", f"eyelid_{zone}_right"]

    def test_eyelid_default_zone_is_full(self, face):
        """Default zone='full' preserves backward-compatible part names."""
        face.eyelid(color=(180, 130, 150), side='both')
        pass_types = [p.pass_type for p in face.makeup_passes]
        assert pass_types == ["eyelid_left", "eyelid_right"]

    def test_eyelid_zone_single_side(self, face):
        """zone with side='left' adds exactly one zone pass."""
        face.eyelid(color=(180, 130, 150), zone='outer', side='left')
        assert [p.pass_type for p in face.makeup_passes] == ["eyelid_outer_left"]

    def test_eyelid_invalid_zone(self, face):
        with pytest.raises(ValueError, match="zone must be one of"):
            face.eyelid(color=(180, 130, 150), zone='top')

    def test_zone_part_names_resolve_in_indices_map(self):
        """Every zone part name produced by the API exists in _INDICES_MAP."""
        from face_makeup.core.passes.eyeshadow import _INDICES_MAP

        for zone in Face._VALID_ZONES:
            for side in ("left", "right"):
                assert Face._eyelid_part(zone, side) in _INDICES_MAP

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_overall_zones_tile_full_eyelid(self, face, test_image, side):
        """The three overall zones together cover (tile) the full eyelid mask."""
        from face_makeup.core.passes.eyeshadow import EyeShadowMaskStrategy
        from face_makeup import constants as C

        coords = face.landmarks.points
        shape = test_image.shape[:2]
        full_pair = (
            getattr(C, f"EYELID_CREASE_{side.upper()}_INDICES"),
            getattr(C, f"EYELID_LASH_{side.upper()}_INDICES"),
        )
        zone_pairs = [
            getattr(C, f"EYELID_OUTER_{side.upper()}_INDICES"),
            getattr(C, f"EYELID_CENTER_{side.upper()}_INDICES"),
            getattr(C, f"EYELID_INNER_{side.upper()}_INDICES"),
        ]

        full = EyeShadowMaskStrategy(*full_pair).generate(coords, shape) > 0
        union = np.zeros(shape, dtype=bool)
        for crease, lash in zone_pairs:
            zone_mask = EyeShadowMaskStrategy(crease, lash).generate(coords, shape) > 0
            assert zone_mask.sum() > 0, "zone mask must be non-empty"
            union |= zone_mask

        covered = (full & union).sum() / full.sum()
        assert covered > 0.9, f"zones cover only {covered:.1%} of full eyelid"


class TestEyelidSize:
    """Tests for the `size` argument on eyelid()."""

    def test_size_propagates_to_params(self, face):
        """Verify size value reaches EyeShadowParams."""
        face.eyelid(color=(180, 130, 150), size=0.5)
        params = face.makeup_passes[0].params
        assert params.size == 0.5

    def test_size_default_is_one(self, face):
        """eyelid() with no size arg should default to 1.0."""
        face.eyelid(color=(180, 130, 150))
        params = face.makeup_passes[0].params
        assert params.size == 1.0

    def test_size_too_high(self, face):
        with pytest.raises(ValueError, match="size must be"):
            face.eyelid(color=(180, 130, 150), size=1.5)

    def test_size_negative(self, face):
        with pytest.raises(ValueError, match="size must be"):
            face.eyelid(color=(180, 130, 150), size=-0.1)

    def test_size_reduces_mask_area(self, face, test_image):
        """Smaller size produces a smaller mask on real landmarks."""
        from face_makeup.core.passes.eyeshadow import EyeShadowMaskStrategy
        from face_makeup import constants as C

        shape = test_image.shape[:2]
        coords = face.landmarks.to_image_coords(shape[1], shape[0])
        crease = C.EYELID_CREASE_LEFT_INDICES
        lash = C.EYELID_LASH_LEFT_INDICES

        area_small = np.count_nonzero(
            EyeShadowMaskStrategy(crease, lash, size=0.3).generate(coords, shape)
        )
        area_full = np.count_nonzero(
            EyeShadowMaskStrategy(crease, lash, size=1.0).generate(coords, shape)
        )
        assert 0 < area_small < area_full

    def test_two_layers_stack(self, face):
        """Calling eyelid() twice (different sizes) stacks four passes."""
        face.eyelid(color=(224, 88, 140), size=1.0, side='both')
        face.eyelid(color=(245, 41, 41), size=0.3, side='both')
        passes = face.makeup_passes
        assert [p.pass_type for p in passes] == [
            "eyelid_left", "eyelid_right", "eyelid_left", "eyelid_right"
        ]
        assert [p.params.size for p in passes] == [1.0, 1.0, 0.3, 0.3]


class TestTearBag:
    """Tests for tear_bag_highlight() and tear_bag_shadow()."""

    def test_highlight_both_sides(self, face):
        """tear_bag_highlight(side='both') appends left + right highlight passes."""
        face.tear_bag_highlight(color=(255, 255, 255), side='both')
        assert [p.pass_type for p in face.makeup_passes] == [
            "tear_bag_highlight_left", "tear_bag_highlight_right"
        ]

    def test_shadow_both_sides(self, face):
        """tear_bag_shadow(side='both') appends left + right shadow passes."""
        face.tear_bag_shadow(color=(150, 110, 90), side='both')
        assert [p.pass_type for p in face.makeup_passes] == [
            "tear_bag_shadow_left", "tear_bag_shadow_right"
        ]

    def test_highlight_single_side(self, face):
        """side='left' adds exactly one highlight pass."""
        face.tear_bag_highlight(side='left')
        assert [p.pass_type for p in face.makeup_passes] == ["tear_bag_highlight_left"]

    def test_shadow_single_side(self, face):
        """side='right' adds exactly one shadow pass."""
        face.tear_bag_shadow(side='right')
        assert [p.pass_type for p in face.makeup_passes] == ["tear_bag_shadow_right"]

    def test_highlight_invalid_side(self, face):
        with pytest.raises(ValueError, match="side must be one of"):
            face.tear_bag_highlight(side='middle')

    def test_shadow_invalid_side(self, face):
        with pytest.raises(ValueError, match="side must be one of"):
            face.tear_bag_shadow(side='both_sides')

    def test_invalid_color(self, face):
        with pytest.raises(ValueError, match="Color values must be integers"):
            face.tear_bag_highlight(color=(300, 0, 0))
        with pytest.raises(ValueError, match="Color values must be integers"):
            face.tear_bag_shadow(color=(0, -1, 0))

    def test_invalid_opacity(self, face):
        with pytest.raises(ValueError, match="Opacity must be"):
            face.tear_bag_highlight(opacity=1.5)
        with pytest.raises(ValueError, match="Opacity must be"):
            face.tear_bag_shadow(opacity=-0.1)

    def test_part_names_resolve_in_ridge_lash_map(self):
        """Every tear bag part name exists in the tear bag _RIDGE_LASH_MAP."""
        from face_makeup.core.passes.tearbag import _RIDGE_LASH_MAP

        for element in ("highlight", "shadow"):
            for side in ("left", "right"):
                assert f"tear_bag_{element}_{side}" in _RIDGE_LASH_MAP

    def test_shadow_priority_above_highlight(self, face):
        """Shadow renders above the highlight (higher priority value)."""
        face.tear_bag_highlight()
        face.tear_bag_shadow()
        by_type = {p.pass_type: p.priority for p in face.makeup_passes}
        assert by_type["tear_bag_shadow_left"] > by_type["tear_bag_highlight_left"]

    def test_landmarks_mirror_around_face_center(self, face):
        """Left/right tear-bag points are mirror-symmetric about the face centerline.

        Center is derived from a known-good mirror pair (eye corners 33/263),
        so this is independent of where the face sits in the image.
        """
        from face_makeup import constants as C

        pts = face.landmarks.points
        center_x = (pts[33][0] + pts[263][0]) / 2
        for li, ri in zip(C.TEAR_BAG_SHADOW_LEFT_INDICES, C.TEAR_BAG_SHADOW_RIGHT_INDICES):
            lx, ly = pts[li]
            rx, ry = pts[ri]
            assert abs((lx + rx) / 2 - center_x) < 0.03, f"x-mirror off for {li}/{ri}"
            assert abs(ly - ry) < 0.03, f"y-mismatch for {li}/{ri}"

    def test_masks_non_empty(self, face, test_image):
        """Highlight (fillPoly) and shadow (polylines) masks are non-empty."""
        from face_makeup.core.passes.tearbag import (
            TearBagHighlightMaskStrategy,
            TearBagShadowMaskStrategy,
            _RIDGE_LASH_MAP,
        )

        coords = face.landmarks.to_image_coords(*test_image.shape[:2])
        shape = test_image.shape[:2]
        for side in ("left", "right"):
            ridge, lash = _RIDGE_LASH_MAP[f"tear_bag_highlight_{side}"]
            highlight = TearBagHighlightMaskStrategy(ridge, lash, size=1.0).generate(coords, shape)
            ridge_s, lash_s = _RIDGE_LASH_MAP[f"tear_bag_shadow_{side}"]
            shadow = TearBagShadowMaskStrategy(ridge_s, lash_s, size=1.0).generate(coords, shape)
            assert highlight.sum() > 0, f"highlight mask empty for {side}"
            assert shadow.sum() > 0, f"shadow mask empty for {side}"

    def test_masks_empty_at_size_zero(self, face, test_image):
        """At size=0, highlight polygon degenerates (zero or near-zero area)."""
        from face_makeup.core.passes.tearbag import (
            TearBagHighlightMaskStrategy,
            _RIDGE_LASH_MAP,
        )

        coords = face.landmarks.to_image_coords(*test_image.shape[:2])
        shape = test_image.shape[:2]
        ridge, lash = _RIDGE_LASH_MAP["tear_bag_highlight_left"]
        mask_full = TearBagHighlightMaskStrategy(ridge, lash, size=1.0).generate(coords, shape)
        mask_zero = TearBagHighlightMaskStrategy(ridge, lash, size=0.0).generate(coords, shape)
        assert mask_zero.sum() < mask_full.sum(), "size=0 must produce smaller area than size=1"

    def test_highlight_blend_modifies_masked_region(self, face, test_image):
        """Highlight modifies pixels inside the mask but leaves the rest untouched.

        Uses the Renderer integration path so the full mask→blend pipeline is
        exercised.  A mid-grey target and moderate opacity produce a measurable
        but not extreme change — the assertion is non-trivial.
        """
        from face_makeup.renderer import Renderer

        face.tear_bag_highlight(
            color=(180, 180, 180), opacity=0.5, blur_strength=0.0, side="left",
        )
        result = Renderer().render_face(test_image.copy(), face)

        # Derive a reference mask from the pass itself — still exercises the full
        # pipeline but lets us isolate the region for the assertion.
        # NOTE: to_image_coords takes (width, height), NOT (height, width).
        hl_pass = next(p for p in face.makeup_passes if "highlight" in p.pass_type)
        h, w = test_image.shape[:2]
        coords = face.landmarks.to_image_coords(w, h)
        ref_mask = hl_pass.mask_strategy.generate(coords, (h, w))
        region = ref_mask > 0

        assert region.sum() > 100, "highlight mask must cover a real area"

        inside = result[region]
        orig_inside = test_image[region]
        assert not np.array_equal(inside, orig_inside), (
            "pixels inside the mask must change"
        )

        outside = result[~region]
        orig_outside = test_image[~region]
        assert np.array_equal(outside, orig_outside), (
            "pixels outside the mask must stay unchanged"
        )

    def test_size_validation(self, face):
        """size outside [0, 1] raises ValueError."""
        import pytest
        with pytest.raises(ValueError, match="size"):
            face.tear_bag_highlight(size=1.5)
        with pytest.raises(ValueError, match="size"):
            face.tear_bag_shadow(size=-0.1)

    def test_highlight_glitter_too_high(self, face):
        with pytest.raises(ValueError, match="glitter must be"):
            face.tear_bag_highlight(glitter=1.5)

    def test_highlight_glitter_negative(self, face):
        with pytest.raises(ValueError, match="glitter must be"):
            face.tear_bag_highlight(glitter=-0.1)

    def test_highlight_glitter_string(self, face):
        with pytest.raises(ValueError, match="glitter must be"):
            face.tear_bag_highlight(glitter="0.5")

    def test_highlight_glitter_default_is_zero(self, face):
        """tear_bag_highlight() with no glitter arg defaults to 0.0."""
        face.tear_bag_highlight(side='left')
        assert face.makeup_passes[0].params.glitter == 0.0

    def test_highlight_glitter_propagates_to_params(self, face):
        """Verify glitter value reaches TearBagHighlightParams."""
        face.tear_bag_highlight(glitter=0.6, side='left')
        assert face.makeup_passes[0].params.glitter == 0.6

    def test_shadow_rejects_glitter_kwarg(self, face):
        """Glitter is highlight-only; tear_bag_shadow does not accept it."""
        with pytest.raises(TypeError):
            face.tear_bag_shadow(glitter=0.5)

    def test_highlight_glitter_changes_output(self, face, test_image):
        """glitter > 0 adds a sparkle overlay, altering the highlighted region."""
        from face_makeup.core.passes.tearbag import (
            TearBagHighlightBlendStrategy,
            TearBagHighlightMaskStrategy,
            _RIDGE_LASH_MAP,
        )
        from face_makeup.core.types import ColorParams, TearBagHighlightParams

        # NOTE: to_image_coords takes (width, height).
        h, w = test_image.shape[:2]
        coords = face.landmarks.to_image_coords(w, h)
        ridge, lash = _RIDGE_LASH_MAP["tear_bag_highlight_left"]
        mask = TearBagHighlightMaskStrategy(ridge, lash, size=1.0).generate(coords, (h, w))

        base = dict(
            color_params=ColorParams(target_rgb=(255, 255, 255)),
            opacity=1.0, blur_strength=0.0,
        )
        matte = TearBagHighlightBlendStrategy().apply(
            test_image, mask, TearBagHighlightParams(glitter=0.0, **base)
        )
        glittered = TearBagHighlightBlendStrategy().apply(
            test_image, mask, TearBagHighlightParams(glitter=0.8, **base)
        )
        assert not np.array_equal(matte, glittered)


class TestTearBagHighlightParamsGlitterValidation:
    """Tests for TearBagHighlightParams glitter validation at the dataclass level."""

    def test_glitter_too_high_via_dataclass(self):
        with pytest.raises(ValueError, match="glitter must be"):
            TearBagHighlightParams(
                color_params=ColorParams(target_rgb=(255, 255, 255)),
                opacity=0.3,
                glitter=1.5,
            )

    def test_glitter_negative_via_dataclass(self):
        with pytest.raises(ValueError, match="glitter must be"):
            TearBagHighlightParams(
                color_params=ColorParams(target_rgb=(255, 255, 255)),
                opacity=0.3,
                glitter=-0.1,
            )

    def test_glitter_default_is_zero_via_dataclass(self):
        params = TearBagHighlightParams(
            color_params=ColorParams(target_rgb=(255, 255, 255)),
            opacity=0.3,
        )
        assert params.glitter == 0.0

