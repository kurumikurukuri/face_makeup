"""Tests for the atomic model download helper (face_makeup.core.model_files)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

import face_makeup.core.model_files as mf

_URL = "https://example.invalid/model.bin"


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, headers: dict | None = None):
        super().__init__(payload)
        self.headers = {} if headers is None else headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.fixture
def serve(monkeypatch):
    """Make urlopen return *payload* (or raise it when it is an exception)."""

    def _serve(payload, headers=None):
        def fake_urlopen(req, timeout):
            assert req.full_url == _URL
            if isinstance(payload, Exception):
                raise payload
            return _FakeResponse(payload, headers)

        monkeypatch.setattr(mf.request, "urlopen", fake_urlopen)

    return _serve


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


class TestDownloadFile:
    def test_validated_payload_lands_at_dest(self, tmp_path, serve):
        serve(b"model-bytes")
        seen = []
        dest = tmp_path / "sub" / "model.bin"

        mf.download_file(_URL, dest, validate=lambda p: seen.append(p.read_bytes()))

        assert dest.read_bytes() == b"model-bytes"
        assert seen == [b"model-bytes"]
        assert _leftovers(dest.parent) == ["model.bin"]

    def test_rejected_payload_leaves_nothing(self, tmp_path, serve):
        serve(b"<html>not a model</html>")

        def reject(path: Path) -> None:
            raise ValueError("bad model")

        with pytest.raises(ValueError, match="bad model"):
            mf.download_file(_URL, tmp_path / "model.bin", validate=reject)
        assert _leftovers(tmp_path) == []

    def test_network_error_leaves_nothing(self, tmp_path, serve):
        serve(OSError("connection reset"))
        with pytest.raises(OSError, match="connection reset"):
            mf.download_file(_URL, tmp_path / "model.bin", validate=lambda p: None)
        assert _leftovers(tmp_path) == []

    def test_rejected_payload_keeps_existing_file(self, tmp_path, serve):
        dest = tmp_path / "model.bin"
        dest.write_bytes(b"old")
        serve(b"new")

        def reject(path: Path) -> None:
            raise ValueError("bad model")

        with pytest.raises(ValueError):
            mf.download_file(_URL, dest, validate=reject)
        assert dest.read_bytes() == b"old"

    def test_oversized_payload_is_rejected(self, tmp_path, serve, monkeypatch):
        monkeypatch.setattr(mf, "_MAX_DOWNLOAD_BYTES", 4)
        serve(b"12345")
        with pytest.raises(OSError, match="exceeds"):
            mf.download_file(_URL, tmp_path / "model.bin", validate=pytest.fail)
        assert _leftovers(tmp_path) == []

    def test_payload_at_size_limit_is_accepted(self, tmp_path, serve, monkeypatch):
        monkeypatch.setattr(mf, "_MAX_DOWNLOAD_BYTES", 5)
        serve(b"12345")
        mf.download_file(_URL, tmp_path / "model.bin", validate=lambda p: None)
        assert (tmp_path / "model.bin").read_bytes() == b"12345"

    def test_oversized_content_length_is_rejected_before_reading(
        self, tmp_path, serve, monkeypatch
    ):
        monkeypatch.setattr(mf, "_MAX_DOWNLOAD_BYTES", 4)
        serve(b"", headers={"Content-Length": "5"})
        with pytest.raises(OSError, match="exceeds"):
            mf.download_file(_URL, tmp_path / "model.bin", validate=pytest.fail)
        assert _leftovers(tmp_path) == []

    def test_slow_download_hits_total_deadline(self, tmp_path, serve, monkeypatch):
        """A trickling connection cannot hold the download open indefinitely."""
        monkeypatch.setattr(mf, "_DOWNLOAD_CHUNK_BYTES", 1)
        ticks = iter(range(0, 10_000, 100))
        monkeypatch.setattr(mf, "_clock", lambda: next(ticks))
        monkeypatch.setattr(mf, "_DOWNLOAD_DEADLINE_SEC", 250)
        serve(b"12345")
        with pytest.raises(TimeoutError):
            mf.download_file(_URL, tmp_path / "model.bin", validate=pytest.fail)
        assert _leftovers(tmp_path) == []

