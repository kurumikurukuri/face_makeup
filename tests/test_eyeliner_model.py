"""Tests for fetching (install / pull time), loading and contract validation of the lash-line model."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import face_makeup.core.eyeliner_model as em
import face_makeup.core.model_files as mf

# The model installed with this checkout (``pip install -e .`` / git hooks).
# CI installs the package, so a missing model there is a failure, not a skip.
_REAL_MODEL = em.model_path()
_requires_real_model = pytest.mark.skipif(
    not _REAL_MODEL.is_file() and not os.environ.get("CI"),
    reason=f"lash-line correction model not installed: {_REAL_MODEL} (run 'pip install -e .')",
)


def _node(name: str, shape: list, type_: str = "tensor(float)") -> SimpleNamespace:
    return SimpleNamespace(name=name, shape=shape, type=type_)


def _valid_inputs() -> list[SimpleNamespace]:
    return [
        _node("patch_a", ["batch", 3, 64, 64]),
        _node("patch_b", ["batch", 3, 64, 64]),
        *(_node(n, ["batch", 1]) for n in ("is_upper", "px", "py", "qx", "qy", "eye_scale")),
    ]


class _FakeSession:
    def __init__(self, inputs=None, outputs=None):
        self._inputs = _valid_inputs() if inputs is None else inputs
        self._outputs = [_node("t", ["batch", 1])] if outputs is None else outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Install the model into tmp_path and start with an empty session cache."""
    monkeypatch.setattr(em, "PACKAGE_MODEL_DIR", tmp_path)
    monkeypatch.setattr(em, "_SESSION_CACHE", {})


class TestValidateContract:
    def test_valid_session_passes(self):
        em.validate_contract(_FakeSession())

    def test_missing_input_rejected(self):
        with pytest.raises(em.ModelContractError, match="inputs mismatch"):
            em.validate_contract(_FakeSession(inputs=_valid_inputs()[:-1]))

    def test_renamed_output_rejected(self):
        with pytest.raises(em.ModelContractError, match="outputs mismatch"):
            em.validate_contract(_FakeSession(outputs=[_node("t_raw", ["batch", 1])]))

    def test_wrong_patch_size_rejected(self):
        inputs = _valid_inputs()
        inputs[0] = _node("patch_a", ["batch", 3, 32, 32])
        with pytest.raises(em.ModelContractError, match="patch_a"):
            em.validate_contract(_FakeSession(inputs=inputs))

    def test_fixed_batch_rejected(self):
        inputs = _valid_inputs()
        inputs[2] = _node("is_upper", [1, 1])
        with pytest.raises(em.ModelContractError, match="dynamic batch"):
            em.validate_contract(_FakeSession(inputs=inputs))

    def test_non_float_input_rejected(self):
        inputs = _valid_inputs()
        inputs[3] = _node("px", ["batch", 1], "tensor(double)")
        with pytest.raises(em.ModelContractError, match=r"tensor\(double\)"):
            em.validate_contract(_FakeSession(inputs=inputs))

    def test_wrong_rank_rejected(self):
        with pytest.raises(em.ModelContractError, match="'t'"):
            em.validate_contract(_FakeSession(outputs=[_node("t", ["batch"])]))


class TestFetch:
    """Install / pull time: the latest ``main`` model is downloaded and validated."""

    def test_downloads_into_dest_dir_with_contract_validation(self, tmp_path, monkeypatch):
        calls, validated = [], []

        def fake_download(url, dest, validate):
            calls.append((url, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
            validate(dest)

        monkeypatch.setattr(em, "download_file", fake_download)
        monkeypatch.setattr(em, "_create_session", lambda p: validated.append(p) or _FakeSession())

        dest = em.fetch(tmp_path / "pkg")

        assert dest == tmp_path / "pkg" / em.MODEL_FILENAME
        assert calls == [(em.MODEL_URL, dest)]
        assert validated == [dest]

    def test_contract_mismatch_rejects_the_download(self, tmp_path, monkeypatch):
        def fake_download(url, dest, validate):
            validate(dest)

        def bad_create(path):
            raise em.ModelContractError("lash-line correction model inputs mismatch")

        monkeypatch.setattr(em, "download_file", fake_download)
        monkeypatch.setattr(em, "_create_session", bad_create)
        with pytest.raises(em.ModelContractError):
            em.fetch(tmp_path)

    def test_hook_without_onnxruntime_defers_validation(self, tmp_path, monkeypatch):
        """A git hook may run under a Python without onnxruntime; loading checks it."""
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        monkeypatch.setattr(em, "_create_session", pytest.fail)
        em._validate_fetched(tmp_path / "model.onnx", required=False)

    def test_required_validation_without_onnxruntime_rejects(self, tmp_path, monkeypatch):
        """pip install must validate before installing; no onnxruntime is a failure."""
        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        def fake_download(url, dest, validate):
            validate(dest)

        monkeypatch.setattr(em, "download_file", fake_download)
        with pytest.raises(ImportError):
            em.fetch(tmp_path, require_validation=True)

    def test_loads_standalone_without_the_package(self, tmp_path):
        """setup.py and the git hooks load eyeliner_model.py by path, outside face_makeup."""
        code = "\n".join([
            "import importlib.util, sys",
            f"spec = importlib.util.spec_from_file_location('fetcher', {str(Path(em.__file__))!r})",
            "m = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(m)",
            "assert 'face_makeup' not in sys.modules",
            "assert 'model_files' not in sys.modules",
            "print(m.MODEL_URL)",
        ])
        out = subprocess.run(
            [sys.executable, "-I", "-c", code], cwd=tmp_path,
            capture_output=True, text=True, check=True,
        )
        assert out.stdout.strip() == em.MODEL_URL


class TestHookEntryPoint:
    def test_success_returns_zero(self, monkeypatch):
        monkeypatch.setattr(em, "fetch", lambda: Path("model.onnx"))
        assert em.main() == 0

    def test_failure_warns_without_raising(self, monkeypatch, capsys):
        """A failed fetch must not break git pull; the previous model is kept."""
        def failing():
            raise OSError("network down")

        monkeypatch.setattr(em, "fetch", failing)
        assert em.main() == 1
        assert "not refreshed" in capsys.readouterr().err


class TestLoadSession:
    """Render time: the installed model is loaded, never downloaded."""

    @pytest.fixture(autouse=True)
    def _no_download(self, monkeypatch):
        monkeypatch.setattr(em, "download_file", pytest.fail)

    def test_loads_the_installed_model_once(self, tmp_path, monkeypatch):
        (tmp_path / em.MODEL_FILENAME).write_bytes(b"x")
        created = []
        monkeypatch.setattr(em, "_create_session", lambda p: created.append(p) or _FakeSession())

        first = em.load_session()

        assert em.load_session() is first
        assert created == [tmp_path / em.MODEL_FILENAME]

    def test_missing_model_asks_to_reinstall(self):
        with pytest.raises(RuntimeError, match="Reinstall face_makeup") as excinfo:
            em.load_session()
        assert "correction='rule'" in str(excinfo.value)

    def test_unloadable_model_asks_to_reinstall(self, tmp_path, monkeypatch):
        (tmp_path / em.MODEL_FILENAME).write_bytes(b"x")

        def broken(path):
            raise OSError("protobuf parsing failed")

        monkeypatch.setattr(em, "_create_session", broken)
        with pytest.raises(RuntimeError, match="Reinstall face_makeup") as excinfo:
            em.load_session()
        assert isinstance(excinfo.value.__cause__, OSError)
        assert em._SESSION_CACHE == {}

    def test_contract_mismatch_raises(self, tmp_path, monkeypatch):
        (tmp_path / em.MODEL_FILENAME).write_bytes(b"x")

        def bad_create(path):
            raise em.ModelContractError("lash-line correction model inputs mismatch")

        monkeypatch.setattr(em, "_create_session", bad_create)
        with pytest.raises(em.ModelContractError, match="Update face_makeup"):
            em.load_session()

    def test_explicit_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            em.load_session(tmp_path / "missing.onnx")

    def test_explicit_file_is_loaded(self, tmp_path, monkeypatch):
        model = tmp_path / "mine.onnx"
        model.write_bytes(b"x")
        monkeypatch.setattr(em, "_create_session", lambda p: _FakeSession())
        assert isinstance(em.load_session(model), _FakeSession)

    def test_concurrent_loads_create_one_session(self, tmp_path, monkeypatch):
        (tmp_path / em.MODEL_FILENAME).write_bytes(b"x")
        created = []
        release = threading.Event()

        def slow_create(path):
            created.append(path)
            release.wait(timeout=5)
            return _FakeSession()

        monkeypatch.setattr(em, "_create_session", slow_create)
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(em.load_session()))
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=5)

        assert len(created) == 1
        assert len(results) == 3
        assert all(session is results[0] for session in results)


@_requires_real_model
class TestRealModel:
    @pytest.fixture(autouse=True)
    def _real_dir(self, monkeypatch):
        monkeypatch.setattr(em, "PACKAGE_MODEL_DIR", _REAL_MODEL.parent)

    def test_installed_model_matches_contract(self):
        em.load_session()

    def test_fetch_validates_and_replaces_existing_file(self, tmp_path, monkeypatch):
        """Validating with a real ORT session must not lock the temp file (Windows)."""
        payload = _REAL_MODEL.read_bytes()

        class _Response:
            headers: dict = {}

            def __init__(self):
                self._buf = memoryview(payload)

            def read(self, n):
                chunk, self._buf = self._buf[:n], self._buf[n:]
                return bytes(chunk)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        monkeypatch.setattr(mf.request, "urlopen", lambda req, timeout: _Response())
        dest = tmp_path / em.MODEL_FILENAME
        dest.write_bytes(b"stale")

        em.fetch(tmp_path)

        assert dest.read_bytes() == payload
        assert sorted(p.name for p in tmp_path.iterdir()) == [em.MODEL_FILENAME]
