"""
Build hook: fetch the lash-line correction model when face_makeup is installed.

Project metadata lives in pyproject.toml; this file only extends ``build_py``
so every ``pip install`` (``git+...``, ``-e .``, a Community Cloud deploy)
downloads the latest EyeContourRefiner ``main`` model into ``face_makeup/models/``.
Rendering never downloads it (see ``face_makeup/core/eyeliner_model.py``).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_FETCHER = Path(__file__).resolve().parent / "src" / "face_makeup" / "core" / "eyeliner_model.py"


def _load_fetcher():
    """Load eyeliner_model.py by path: the package itself is not importable at build time."""
    spec = importlib.util.spec_from_file_location("_face_makeup_model_fetcher", _FETCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildPyWithModel(build_py):
    """``build_py`` that also fetches the lash-line correction model."""

    def run(self) -> None:
        super().run()
        fetcher = _load_fetcher()
        # An editable install imports the package from src/, so the model goes
        # next to the sources; a regular build ships it inside the wheel.
        if getattr(self, "editable_mode", False):
            dest_dir = fetcher.PACKAGE_MODEL_DIR
        else:
            dest_dir = Path(self.build_lib) / "face_makeup" / "models"
        try:
            # onnxruntime is a build requirement (pyproject.toml), so the
            # contract is always checked before the model is installed.
            fetcher.fetch(dest_dir, require_validation=True)
        except Exception as exc:
            raise SystemExit(
                f"face_makeup: failed to fetch the lash-line correction model from "
                f"{fetcher.MODEL_URL}: {exc}. Check the network connection and retry."
            ) from exc


setup(cmdclass={"build_py": BuildPyWithModel})
