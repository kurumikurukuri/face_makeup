"""
Tests for face-wide lash-line landmark correction (core/landmark_correction.py).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from face_makeup.constants import EDGE_LANDMARK_DIRECTIONS
from face_makeup.core import eyeliner_ml_corrector
from face_makeup.core.eyeliner_corrector import EyelinerCorrector
from face_makeup.core.eyeliner_ml_corrector import EyelinerMLCorrector
from face_makeup.core.eyeliner_model import ModelContractError
from face_makeup.core.landmark_correction import LashLineCorrector
from face_makeup.core.landmark_resolver import extend_landmarks
from face_makeup.core.types import FaceLandmarks


# ── fixtures ─────────────────────────────────────────────────────────────────


class _FixedTSession:
    """ONNX session stand-in returning the same ``t`` for every edge."""

    def __init__(self, t: float = 0.5):
        self._t = t
        self.runs = 0

    def run(self, output_names, feed):
        self.runs += 1
        return [np.full((len(feed["px"]), 1), self._t, dtype=np.float32)]


@pytest.fixture
def landmarks():
    rng = np.random.RandomState(0)
    points = (0.3 + 0.4 * rng.rand(478, 2)).astype(np.float32)
    return extend_landmarks(FaceLandmarks(points=points))


@pytest.fixture
def image():
    rng = np.random.RandomState(1)
    return rng.randint(0, 256, (120, 160, 3), dtype=np.uint8)


def _ml_corrector_with_t(t: float) -> LashLineCorrector:
    corrector = LashLineCorrector("ml")
    ml = corrector.resolve()
    ml._session = _FixedTSession(t)
    return corrector


# ── mode validation / resolution ─────────────────────────────────────────────


class TestModes:
    @pytest.mark.parametrize("mode", ["ml", "rule"])
    def test_accepts_modes(self, mode):
        assert LashLineCorrector(mode).mode == mode

    @pytest.mark.parametrize("mode", ["auto", "off", "magic", ""])
    def test_rejects_other_modes(self, mode):
        with pytest.raises(ValueError, match="'ml' or 'rule'"):
            LashLineCorrector(mode)

    def test_rule_mode_selects_rule_corrector(self):
        corrector = LashLineCorrector("rule")
        for _ in range(2):
            assert isinstance(corrector.resolve(), EyelinerCorrector)

    def test_ml_mode_selects_ml_corrector(self):
        assert isinstance(LashLineCorrector("ml").resolve(), EyelinerMLCorrector)

    def test_model_is_loaded_lazily(self, monkeypatch):
        """Building and resolving a corrector never loads the model."""
        monkeypatch.setattr(eyeliner_ml_corrector, "load_session", pytest.fail)
        LashLineCorrector("ml").resolve()

    def test_resolved_corrector_is_reused(self):
        corrector = LashLineCorrector("ml")
        assert corrector.resolve() is corrector.resolve()


# ── model failures ───────────────────────────────────────────────────────────


class TestModelFailure:
    """A model that cannot be loaded raises; there is no fallback."""

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("model not installed"),
            ModelContractError("contract mismatch"),
            OSError("disk full"),
        ],
    )
    def test_ml_raises_when_model_unavailable(self, monkeypatch, landmarks, image, error):
        def fail(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(eyeliner_ml_corrector, "load_session", fail)
        corrector = LashLineCorrector("ml")
        h, w = image.shape[:2]
        with pytest.raises(type(error), match=re.escape(str(error))):
            corrector.apply(image, landmarks, landmarks.to_image_coords(w, h))
        assert isinstance(corrector.resolve(), EyelinerMLCorrector)

    def test_rule_never_loads_the_model(self, monkeypatch, landmarks, image):
        monkeypatch.setattr(eyeliner_ml_corrector, "load_session", pytest.fail)
        h, w = image.shape[:2]
        LashLineCorrector("rule").apply(image, landmarks, landmarks.to_image_coords(w, h))


# ── apply(): corrected coordinates ───────────────────────────────────────────


class TestApply:
    def test_edge_rows_hold_corrected_points(self, landmarks, image):
        """Every edge landmark row becomes R = t*P + (1-t)*Q (t=0.5 → midpoint)."""
        h, w = image.shape[:2]
        coords = landmarks.to_image_coords(w, h)
        result = _ml_corrector_with_t(0.5).apply(image, landmarks, coords)

        for edge_id, (p_idx, q_idx) in EDGE_LANDMARK_DIRECTIONS.items():
            expected = (coords[p_idx].astype(np.float32) + coords[q_idx]) / 2.0
            np.testing.assert_allclose(result.coords[edge_id], expected, atol=1e-4)

    def test_other_rows_unchanged(self, landmarks, image):
        h, w = image.shape[:2]
        coords = landmarks.to_image_coords(w, h)
        result = _ml_corrector_with_t(0.5).apply(image, landmarks, coords)

        others = [i for i in range(len(coords)) if i not in EDGE_LANDMARK_DIRECTIONS]
        np.testing.assert_array_equal(result.coords[others], coords[others])

    def test_returns_new_float_array_without_mutating_input(self, landmarks, image):
        h, w = image.shape[:2]
        coords = landmarks.to_image_coords(w, h)
        before = coords.copy()
        result = _ml_corrector_with_t(0.5).apply(image, landmarks, coords)

        assert result.coords is not coords
        assert result.coords.dtype == np.float32
        assert result.coords.shape == coords.shape
        np.testing.assert_array_equal(coords, before)

    def test_t_one_keeps_uncorrected_points(self, landmarks, image):
        """t = 1 is R = P, the value extend_landmarks already stores."""
        h, w = image.shape[:2]
        coords = landmarks.to_image_coords(w, h)
        result = _ml_corrector_with_t(1.0).apply(image, landmarks, coords)
        np.testing.assert_allclose(result.coords, coords, atol=1e-4)

    def test_ml_runs_once_per_eye(self, landmarks, image):
        corrector = _ml_corrector_with_t(0.5)
        h, w = image.shape[:2]
        corrector.apply(image, landmarks, landmarks.to_image_coords(w, h))
        assert corrector.resolve()._session.runs == 2

    def test_ml_model_sees_float_coordinates(self, monkeypatch, landmarks, image):
        corrector = LashLineCorrector("ml")
        ml = corrector.resolve()
        received = []
        real = ml.correct_points

        def spy(coords, part_name, eyeliner, model_coords=None):
            received.append(model_coords)
            return real(coords, part_name, eyeliner, model_coords=model_coords)

        ml._session = _FixedTSession(0.5)
        monkeypatch.setattr(ml, "correct_points", spy)
        h, w = image.shape[:2]
        corrector.apply(image, landmarks, landmarks.to_image_coords(w, h))

        assert len(received) == 2
        for model_coords in received:
            np.testing.assert_array_equal(
                model_coords, landmarks.to_image_coords_float(w, h)
            )

    def test_rule_mode_corrects_on_int_coords(self, landmarks, image):
        h, w = image.shape[:2]
        coords = landmarks.to_image_coords(w, h)
        result = LashLineCorrector("rule").apply(image, landmarks, coords)
        assert result.coords.shape == coords.shape
        assert np.isfinite(result.coords).all()
        assert isinstance(result.corrector, EyelinerCorrector)

    def test_result_exposes_corrector_used(self, landmarks, image):
        corrector = _ml_corrector_with_t(0.5)
        h, w = image.shape[:2]
        result = corrector.apply(image, landmarks, landmarks.to_image_coords(w, h))
        assert result.corrector is corrector.resolve()
