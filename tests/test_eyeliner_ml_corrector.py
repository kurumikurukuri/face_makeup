"""
Tests for ML-based eyeliner edge correction (EyelinerMLCorrector).

The ONNX session is faked so the orchestration logic (batching, eye_scale,
ordering, reconstruction) is verified without the model; a final smoke test
exercises the real model when it is installed (always in CI).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from face_makeup.constants import (
    EDGE_LANDMARK_DIRECTIONS,
    EYELINER_CORRECTION_MAP,
    EYELINER_LEFT_INDICES,
    EYELINER_RIGHT_INDICES,
    LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407,
    LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507,
    LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607,
    LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707,
)
from face_makeup.core import eyeliner_ml_corrector
from face_makeup.core.eyeliner_ml_corrector import EyelinerMLCorrector
from face_makeup.core.eyeliner_model import model_path as model_path_of_installed_model
from face_makeup.core.eyeliner_patch import compute_eye_scale
from face_makeup.core.landmark_resolver import extend_landmarks
from face_makeup.core.types import FaceLandmarks


# The 14 edges per eye the model was trained on, by lid.
_TRAINED_EDGES = {
    "eyeliner_left": {
        "upper": [LM_1401, LM_1402, LM_1403, LM_1404, LM_1405, LM_1406, LM_1407],
        "lower": [LM_1501, LM_1502, LM_1503, LM_1504, LM_1505, LM_1506, LM_1507],
    },
    "eyeliner_right": {
        "upper": [LM_1601, LM_1602, LM_1603, LM_1604, LM_1605, LM_1606, LM_1607],
        "lower": [LM_1701, LM_1702, LM_1703, LM_1704, LM_1705, LM_1706, LM_1707],
    },
}


# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeSession:
    """Records each feed and returns a controlled ``t`` per batch item.

    ``t_upper`` (when given) is returned for items fed with ``is_upper=1``,
    so tests can tell which lid flag produced a point.
    """

    def __init__(self, t_value: float = 0.0, t_upper: float | None = None):
        self._t_value = t_value
        self._t_upper = t_value if t_upper is None else t_upper
        self.feeds: list[dict] = []

    def run(self, output_names, feed) -> list[np.ndarray]:
        self.feeds.append(feed)
        is_upper = feed["is_upper"] == 1.0
        return [np.where(is_upper, self._t_upper, self._t_value).astype(np.float32)]


def _edge_occurrences(part_name: str) -> int:
    """Number of edge landmark slots in a part's correction map (duplicates counted)."""
    correction_map = EYELINER_CORRECTION_MAP[part_name]
    return sum(
        sum(1 for idx in indices if idx in EDGE_LANDMARK_DIRECTIONS)
        for indices in correction_map.values()
    )


def _unique_edge_ids(part_name: str) -> list[int]:
    """Distinct edge landmarks of a part, in correction-map order."""
    seen: dict[int, None] = {}
    for indices in EYELINER_CORRECTION_MAP[part_name].values():
        for idx in indices:
            if idx in EDGE_LANDMARK_DIRECTIONS:
                seen.setdefault(idx)
    return list(seen)


def _expected_for_t(coords: np.ndarray, part_name: str, t: float) -> np.ndarray:
    """Expected corrected output for a constant ``t`` (mirrors the corrector)."""
    correction_map = EYELINER_CORRECTION_MAP[part_name]
    out: list[tuple[float, float]] = []
    for _key, indices in correction_map.items():
        for idx in indices:
            if idx in EDGE_LANDMARK_DIRECTIONS:
                p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[idx]
                px, py = float(coords[p_idx][0]), float(coords[p_idx][1])
                qx, qy = float(coords[q_idx][0]), float(coords[q_idx][1])
                out.append((t * px + (1.0 - t) * qx, t * py + (1.0 - t) * qy))
            else:
                out.append((float(coords[idx][0]), float(coords[idx][1])))
    return np.array(out, dtype=np.float32)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_landmarks():
    rng = np.random.RandomState(42)
    points = rng.rand(478, 2).astype(np.float32)
    return extend_landmarks(FaceLandmarks(points=points))


@pytest.fixture
def image():
    return np.full((200, 200, 3), 128, dtype=np.uint8)


# ── float image coords (model input) ────────────────────────────────────────


class TestFloatImageCoords:
    def test_float_coords_are_unrounded_and_round_to_int_coords(self, sample_landmarks):
        w, h = 200, 150
        float_coords = sample_landmarks.to_image_coords_float(w, h)
        assert float_coords.dtype == np.float32
        assert not np.array_equal(float_coords, np.round(float_coords))
        np.testing.assert_array_equal(
            np.round(float_coords).astype(np.int32), sample_landmarks.to_image_coords(w, h)
        )


# ── model location ───────────────────────────────────────────────────────────


class TestModelLocation:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, None),
            ({"model_path": "/opt/m.onnx"}, "/opt/m.onnx"),
        ],
    )
    def test_load_session_receives_location(self, monkeypatch, kwargs, expected):
        calls = []

        def fake_load(explicit=None):
            calls.append(explicit)
            return _FakeSession()

        monkeypatch.setattr(eyeliner_ml_corrector, "load_session", fake_load)
        EyelinerMLCorrector(**kwargs)._load_session()
        assert calls == [expected]


# ── correct_points orchestration ─────────────────────────────────────────────


class TestCorrectPoints:
    def test_returns_float32(self, sample_landmarks, image):
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=0.0)
        result = corrector.correct_points(coords, "eyeliner_left", image)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_point_count_matches_map(self, sample_landmarks, image):
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=0.0)

        for part_name, correction_map in EYELINER_CORRECTION_MAP.items():
            expected = sum(len(indices) for indices in correction_map.values())
            result = corrector.correct_points(coords, part_name, image)
            assert result.shape == (expected, 2), part_name

    @pytest.mark.parametrize("t_value", [0.0, 1.0, 0.5])
    def test_reconstructs_r_equals_t_p_plus_1_minus_t_q(
        self, sample_landmarks, image, t_value
    ):
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=t_value)

        result = corrector.correct_points(coords, "eyeliner_left", image)
        expected = _expected_for_t(coords, "eyeliner_left", t_value)
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("raw_t", [np.nan, np.inf, -np.inf])
    def test_non_finite_t_falls_back_to_p(self, sample_landmarks, image, caplog, raw_t):
        """A NaN/inf prediction yields R = P (t = 1) and a warning, never NaN."""
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=raw_t)

        with caplog.at_level("WARNING", logger=eyeliner_ml_corrector.__name__):
            result = corrector.correct_points(coords, "eyeliner_left", image)

        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, _expected_for_t(coords, "eyeliner_left", 1.0))
        assert "non-finite" in caplog.text

    @pytest.mark.parametrize("raw_t, clipped", [(5.0, 2.0), (-7.0, -1.0), (1.5, 1.5)])
    def test_t_clipped_to_label_range(self, sample_landmarks, image, raw_t, clipped):
        """Outputs beyond the training label range [-1, 2] are clipped to it."""
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=raw_t)

        result = corrector.correct_points(coords, "eyeliner_left", image)

        np.testing.assert_allclose(result, _expected_for_t(coords, "eyeliner_left", clipped))
        entries = corrector._debug_cache["eyeliner_left"][1]
        assert all(e["t"] == clipped for e in entries)

    def test_batches_all_edge_points(self, sample_landmarks, image):
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        session = _FakeSession(t_value=0.5)
        corrector = EyelinerMLCorrector()
        corrector._session = session

        corrector.correct_points(coords, "eyeliner_left", image)

        assert len(session.feeds) == 1
        feed = session.feeds[0]
        # Each distinct edge is inferred once (the duplicated 1507 is not re-run).
        n = len(_unique_edge_ids("eyeliner_left"))
        assert 0 < n < _edge_occurrences("eyeliner_left")
        assert feed["patch_a"].shape == (n, 3, 64, 64)
        assert feed["patch_b"].shape == (n, 3, 64, 64)
        assert feed["is_upper"].shape == (n, 1)
        assert feed["px"].shape == (n, 1)
        assert feed["py"].shape == (n, 1)
        assert feed["qx"].shape == (n, 1)
        assert feed["qy"].shape == (n, 1)

    def test_model_input_uses_model_coords_and_r_uses_drawing_coords(
        self, sample_landmarks, image
    ):
        """P/Q/eye_scale fed to the model are float; R is rebuilt on the int coords."""
        h, w = image.shape[:2]
        float_coords = sample_landmarks.to_image_coords_float(w, h)
        int_coords = sample_landmarks.to_image_coords(w, h)
        session = _FakeSession(t_value=0.3)
        corrector = EyelinerMLCorrector()
        corrector._session = session

        result = corrector.correct_points(
            int_coords, "eyeliner_left", image, model_coords=float_coords
        )

        feed = session.feeds[0]
        edge_ids = _unique_edge_ids("eyeliner_left")
        for i, idx in enumerate(edge_ids):
            p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[idx]
            assert feed["px"][i, 0] == float_coords[p_idx][0]
            assert feed["qy"][i, 0] == float_coords[q_idx][1]
        expected_scale = compute_eye_scale(
            tuple(float_coords[j]) for idx in edge_ids for j in EDGE_LANDMARK_DIRECTIONS[idx]
        )
        np.testing.assert_allclose(feed["eye_scale"][0, 0], expected_scale, rtol=1e-6)
        np.testing.assert_allclose(
            result, _expected_for_t(int_coords.astype(np.float32), "eyeliner_left", 0.3),
            rtol=0.0, atol=1e-4,
        )

    def test_model_coords_default_to_coords(self, sample_landmarks, image):
        h, w = image.shape[:2]
        int_coords = sample_landmarks.to_image_coords(w, h)
        session = _FakeSession(t_value=0.5)
        corrector = EyelinerMLCorrector()
        corrector._session = session
        corrector.correct_points(int_coords, "eyeliner_left", image)
        p_idx, _ = EDGE_LANDMARK_DIRECTIONS[_unique_edge_ids("eyeliner_left")[0]]
        assert session.feeds[0]["px"][0, 0] == int_coords[p_idx][0]

    def test_eye_scale_is_shared_scalar(self, sample_landmarks, image):
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        session = _FakeSession(t_value=0.5)
        corrector = EyelinerMLCorrector()
        corrector._session = session

        corrector.correct_points(coords, "eyeliner_left", image)

        feed = session.feeds[0]
        eye_scale = feed["eye_scale"]
        n = len(_unique_edge_ids("eyeliner_left"))
        assert eye_scale.shape == (n, 1)
        assert np.all(eye_scale == eye_scale[0, 0])
        assert eye_scale[0, 0] >= 1.0

    @pytest.mark.parametrize("part_name", sorted(_TRAINED_EDGES))
    def test_eye_scale_uses_trained_edges(self, sample_landmarks, image, part_name):
        """eye_scale matches the model's definition: the 14 trained P/Q edges."""
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        session = _FakeSession(t_value=0.5)
        corrector = EyelinerMLCorrector()
        corrector._session = session

        corrector.correct_points(coords, part_name, image)

        trained = [idx for indices in _TRAINED_EDGES[part_name].values() for idx in indices]
        assert sorted(_unique_edge_ids(part_name)) == sorted(trained)
        points = []
        for idx in trained:
            p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[idx]
            points += [tuple(coords[p_idx]), tuple(coords[q_idx])]
        expected = compute_eye_scale(points)
        assert session.feeds[0]["eye_scale"][0, 0] == pytest.approx(expected, rel=1e-6)

    def test_is_upper_flag_follows_lid(self, sample_landmarks, image):
        """Each edge takes the lid it belongs to; the shared 1507 is a lower-lid edge."""
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        session = _FakeSession(t_value=0.5)
        corrector = EyelinerMLCorrector()
        corrector._session = session

        corrector.correct_points(coords, "eyeliner_left", image)

        is_upper = dict(
            zip(_unique_edge_ids("eyeliner_left"), session.feeds[0]["is_upper"].reshape(-1))
        )
        for idx in _TRAINED_EDGES["eyeliner_left"]["upper"]:
            assert is_upper[idx] == 1.0, idx
        for idx in _TRAINED_EDGES["eyeliner_left"]["lower"]:
            assert is_upper[idx] == 0.0, idx
        assert len(is_upper) == 14

    @pytest.mark.parametrize("part_name, dup_id", [
        ("eyeliner_left", LM_1507),
        ("eyeliner_right", LM_1707),
    ])
    def test_duplicated_edge_uses_lower_lid_result(self, sample_landmarks, image, part_name, dup_id):
        """The edge shared by both polylines gets the lower-lid inference in both."""
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=0.2, t_upper=0.9)

        result = corrector.correct_points(coords, part_name, image)

        correction_map = EYELINER_CORRECTION_MAP[part_name]
        slots = [idx for indices in correction_map.values() for idx in indices]
        positions = [i for i, idx in enumerate(slots) if idx == dup_id]
        assert len(positions) == 2  # once in the upper polyline, once in the lower

        p_idx, q_idx = EDGE_LANDMARK_DIRECTIONS[dup_id]
        lower = 0.2 * coords[p_idx] + 0.8 * coords[q_idx]
        for pos in positions:
            np.testing.assert_allclose(result[pos], lower, rtol=0.0, atol=1e-4)

    def test_invalid_part_name_raises(self, sample_landmarks, image):
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession()
        with pytest.raises(ValueError, match="Unknown part_name"):
            corrector.correct_points(coords, "lips", image)


# ── get_debug_entries ────────────────────────────────────────────────────────


class TestGetDebugEntries:
    def test_returns_base_key_entries(self, sample_landmarks, image):
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=0.5)
        entries = corrector.get_debug_entries(
            image, sample_landmarks, image.shape[:2], "eyeliner_left", side="left"
        )
        assert isinstance(entries, list)
        assert len(entries) == _edge_occurrences("eyeliner_left")
        for entry in entries:
            assert {"px", "py", "qx", "qy", "detected_x", "detected_y", "side"} <= entry.keys()
            assert entry["side"] == "left"
            # ML entries carry no rule-based derivative data.
            assert "samples" not in entry

    def test_uses_int_image_coords(self, sample_landmarks, image):
        """Drawn P/Q come from the same rounded coords the renderer draws with."""
        h, w = image.shape[:2]
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=0.5)
        entries = corrector.get_debug_entries(
            image, sample_landmarks, (h, w), "eyeliner_left", side="left"
        )
        for entry in entries:
            for key in ("px", "py", "qx", "qy"):
                assert entry[key] == round(entry[key])

    def test_fallback_inference_uses_float_coords(self, sample_landmarks, image):
        """Without a cached render, the model is still fed unrounded coords."""
        h, w = image.shape[:2]
        session = _FakeSession(t_value=0.5)
        corrector = EyelinerMLCorrector()
        corrector._session = session
        corrector.get_debug_entries(
            image, sample_landmarks, (h, w), "eyeliner_left", side="left"
        )
        float_coords = sample_landmarks.to_image_coords_float(w, h)
        p_idx, _ = EDGE_LANDMARK_DIRECTIONS[_unique_edge_ids("eyeliner_left")[0]]
        assert session.feeds[0]["px"][0, 0] == float_coords[p_idx][0]

    def test_reuses_render_result_without_reinference(self, sample_landmarks, image):
        """After correct_points, debug entries come from the cache (no 2nd run)."""
        h, w = image.shape[:2]
        corrector = EyelinerMLCorrector()
        session = _FakeSession(t_value=0.25)
        corrector._session = session
        corrected = corrector.correct_points(
            sample_landmarks.to_image_coords(w, h), "eyeliner_left", image
        )

        entries = corrector.get_debug_entries(
            image, sample_landmarks, (h, w), "eyeliner_left", side="left", eyeliner=image
        )

        assert len(session.feeds) == 1
        assert len(entries) == _edge_occurrences("eyeliner_left")
        drawn = {(round(x, 4), round(y, 4)) for x, y in corrected.tolist()}
        for entry in entries:
            assert (round(entry["detected_x"], 4), round(entry["detected_y"], 4)) in drawn

    def test_entries_carry_t_and_landmark_id(self, sample_landmarks, image):
        h, w = image.shape[:2]
        coords = sample_landmarks.to_image_coords(w, h)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession(t_value=0.25)
        corrector.correct_points(coords, "eyeliner_left", image)

        entries = corrector.get_debug_entries(
            image, sample_landmarks, (h, w), "eyeliner_left", side="left", eyeliner=image
        )

        assert [e["landmark_id"] for e in entries] == [
            idx
            for indices in EYELINER_CORRECTION_MAP["eyeliner_left"].values()
            for idx in indices
            if idx in EDGE_LANDMARK_DIRECTIONS
        ]
        for entry in entries:
            assert entry["t"] == pytest.approx(0.25)
            assert entry["detected_x"] == pytest.approx(
                0.25 * entry["px"] + 0.75 * entry["qx"]
            )

    def test_recomputes_for_a_different_image(self, sample_landmarks, image):
        """The cache is tied to the source image object."""
        h, w = image.shape[:2]
        corrector = EyelinerMLCorrector()
        session = _FakeSession(t_value=0.5)
        corrector._session = session
        corrector.correct_points(
            sample_landmarks.to_image_coords(w, h), "eyeliner_left", image
        )

        other = image.copy()
        corrector.get_debug_entries(
            other, sample_landmarks, (h, w), "eyeliner_left", side="left", eyeliner=other
        )

        assert len(session.feeds) == 2

    def test_cache_does_not_keep_image_alive(self, sample_landmarks):
        import gc
        import weakref

        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        ref = weakref.ref(img)
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession()
        corrector.correct_points(
            sample_landmarks.to_image_coords(200, 200), "eyeliner_left", img
        )
        del img
        gc.collect()
        assert ref() is None

    def test_invalid_part_name_raises(self, sample_landmarks, image):
        corrector = EyelinerMLCorrector()
        corrector._session = _FakeSession()
        with pytest.raises(ValueError, match="Unknown part_name"):
            corrector.get_debug_entries(
                image, sample_landmarks, image.shape[:2], "lips", side="left"
            )


# ── correction map ───────────────────────────────────────────────────────────


class TestCorrectionMap:
    def test_correction_map_matches_drawing_polylines(self):
        """Corrected polylines are the drawing polylines (upper line runs 1407 -> 130)."""
        drawing = {"eyeliner_left": EYELINER_LEFT_INDICES, "eyeliner_right": EYELINER_RIGHT_INDICES}
        for part_name, sub_map in drawing.items():
            assert EYELINER_CORRECTION_MAP[part_name] == sub_map
        assert 1408 not in EDGE_LANDMARK_DIRECTIONS
        assert 1608 not in EDGE_LANDMARK_DIRECTIONS


# ── real-model smoke test ────────────────────────────────────────────────────


class TestRealModelSmoke:
    def test_end_to_end_when_available(self, sample_landmarks, image):
        """Run a real inference end-to-end when the model is installed."""
        model_path = model_path_of_installed_model()
        # CI installs the package, so a missing model there is a failure, not a skip.
        if not model_path.is_file() and not os.environ.get("CI"):
            pytest.skip(f"lash-line correction model not installed: {model_path}")

        corrector = EyelinerMLCorrector(model_path)
        h, w = image.shape[:2]
        coords = sample_landmarks.points * np.array([w, h], dtype=np.float32)
        result = corrector.correct_points(coords, "eyeliner_left", image)

        expected = sum(
            len(indices) for indices in EYELINER_CORRECTION_MAP["eyeliner_left"].values()
        )
        assert result.shape == (expected, 2)
        assert np.isfinite(result).all()
