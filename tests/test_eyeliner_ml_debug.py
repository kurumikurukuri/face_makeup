"""
Tests for the ML eyeliner debug overlay (P / Q / R drawn on the eye crop).
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import pytest

from face_makeup.core.debug_visualizer import DebugVisualizer
from face_makeup.core import eyeliner_ml_debug
from face_makeup.core.eyeliner_ml_debug import (
    P_COLOR,
    Q_COLOR,
    R_COLOR,
    draw_eyeliner_ml_overlay,
    save_eyeliner_ml_overlay,
)


def _entry(px, py, qx, qy, t, landmark_id=1401):
    return {
        "px": px, "py": py, "qx": qx, "qy": qy,
        "detected_x": t * px + (1 - t) * qx,
        "detected_y": t * py + (1 - t) * qy,
        "t": t,
        "landmark_id": landmark_id,
    }


@pytest.fixture
def image():
    return np.full((200, 300, 3), 128, dtype=np.uint8)


@pytest.fixture
def entries():
    return [
        _entry(100, 100, 100, 80, 0.3, 1401),
        _entry(120, 98, 122, 78, 0.6, 1402),
        _entry(140, 100, 142, 120, 0.5, 1501),
    ]


def _has_color(img: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.all(img == np.array(color, dtype=np.uint8), axis=-1).any())


class TestDrawEyelinerMlOverlay:
    def test_empty_entries_returns_none(self, image):
        assert draw_eyeliner_ml_overlay(image, []) is None

    def test_returns_rgb_uint8(self, image, entries):
        out = draw_eyeliner_ml_overlay(image, entries)
        assert out.dtype == np.uint8
        assert out.ndim == 3 and out.shape[2] == 3

    def test_draws_p_q_and_r(self, image, entries):
        out = draw_eyeliner_ml_overlay(image, entries)
        for color in (P_COLOR, Q_COLOR, R_COLOR):
            assert _has_color(out, color), color

    def test_crop_is_upscaled_for_legibility(self, image, entries):
        """The eye region is small; the output is enlarged beyond the crop."""
        out = draw_eyeliner_ml_overlay(image, entries)
        assert max(out.shape[:2]) > 100

    def test_does_not_modify_input(self, image, entries):
        before = image.copy()
        draw_eyeliner_ml_overlay(image, entries)
        np.testing.assert_array_equal(image, before)

    def test_points_outside_image_are_clipped(self, image):
        out = draw_eyeliner_ml_overlay(image, [_entry(5, 5, -20, -30, 2.0)])
        assert out is not None

    def test_degenerate_crop_returns_none(self):
        """All points fully outside the image -> nothing to show."""
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        assert draw_eyeliner_ml_overlay(tiny, [_entry(500, 500, 510, 510, 0.5)]) is None


class TestSaveEyelinerMlDebug:
    def test_writes_png_in_session_dir(self, tmp_path, image, entries):
        visualizer = DebugVisualizer(debug_dir=str(tmp_path))
        visualizer.save_eyeliner_ml_debug(image, "left", entries, session_id="s1")

        path = tmp_path / "s1" / "eyeliner_ml_debug_left.png"
        assert path.is_file()
        assert cv2.imread(str(path)) is not None

    def test_no_file_for_empty_entries(self, tmp_path, image):
        visualizer = DebugVisualizer(debug_dir=str(tmp_path))
        visualizer.save_eyeliner_ml_debug(image, "left", [], session_id="s1")
        assert not os.path.exists(tmp_path / "s1" / "eyeliner_ml_debug_left.png")

    def test_rejects_unsafe_session_id(self, tmp_path, image, entries):
        visualizer = DebugVisualizer(debug_dir=str(tmp_path))
        with pytest.raises(ValueError):
            visualizer.save_eyeliner_ml_debug(image, "left", entries, session_id="../x")

    def test_write_failure_is_reported(self, tmp_path, image, entries, monkeypatch, caplog):
        monkeypatch.setattr(eyeliner_ml_debug.cv2, "imwrite", lambda *_: False)
        path = str(tmp_path / "s1" / "eyeliner_ml_debug_left.png")
        with caplog.at_level(logging.WARNING, logger=eyeliner_ml_debug.__name__):
            assert save_eyeliner_ml_overlay(path, image, entries) is False
        assert "failed to write" in caplog.text
