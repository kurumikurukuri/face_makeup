"""
Fetching, location and contract validation of the lash-line correction ONNX model.

The model tracks the ``main`` branch of EyeContourRefiner and is not committed
here. It is fetched when face_makeup is installed and when a checkout is
updated, never while rendering:

* ``pip install`` (``git+...``, ``-e .``, a Streamlit Community Cloud deploy):
  the ``build_py`` hook in ``setup.py`` runs :func:`fetch` and the model is
  installed as package data (``face_makeup/models/``).
* ``git pull`` in a checkout with ``git config core.hooksPath .githooks``: the
  ``post-merge`` / ``post-rewrite`` hooks run this file as a script.

Both run outside the installed package, so this module imports only the
standard library and the sibling :mod:`model_files` at module level;
onnxruntime is imported lazily. There is no pinned checksum; every file is
checked against the input/output contract this package was written for when
it is fetched and whenever a session is loaded. ``pip install`` always checks
it at fetch time (onnxruntime is a build requirement); a git hook running
under a Python without onnxruntime leaves the check to load time.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from .model_files import download_file
except ImportError:  # run by setup.py or a git hook, outside the package
    import importlib.util

    # Loaded by path like setup.py loads this file: putting core/ on sys.path
    # would expose its modules (types.py, pipeline.py, ...) as top-level names.
    _spec = importlib.util.spec_from_file_location(
        "_face_makeup_model_files", Path(__file__).resolve().parent / "model_files.py"
    )
    _model_files = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_model_files)
    download_file = _model_files.download_file

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_FILENAME = "eye_contour_refiner.onnx"
MODEL_URL = (
    "https://github.com/kurumikurukuri/EyeContourRefiner/raw/main/models/"
    + MODEL_FILENAME
)
# face_makeup/models/: in a checkout src/face_makeup/models/ (git-ignored),
# once installed site-packages/face_makeup/models/.
PACKAGE_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

PATCH_SIZE = (64, 64)

# name -> shape after the (dynamic) batch axis.
_EXPECTED_INPUTS: dict[str, tuple[int, ...]] = {
    "patch_a": (3, *PATCH_SIZE),
    "patch_b": (3, *PATCH_SIZE),
    "is_upper": (1,),
    "px": (1,),
    "py": (1,),
    "qx": (1,),
    "qy": (1,),
    "eye_scale": (1,),
}
_EXPECTED_OUTPUTS: dict[str, tuple[int, ...]] = {"t": (1,)}

# Sessions keyed by resolved model path. An InferenceSession is read-only, so
# sharing it across corrector instances (one per eye) avoids reloading the
# model on every lash-line correction. MakeupProcessor is documented as non-thread-safe.
_SESSION_CACHE: dict[str, "ort.InferenceSession"] = {}
# Serialises session creation across threads (e.g. concurrent Streamlit
# sessions), so the 15 MB model is loaded only once.
_LOAD_LOCK = threading.Lock()

_FLOAT_TENSOR = "tensor(float)"

_REINSTALL_HINT = (
    "Reinstall face_makeup to fetch the model again (or run 'pip install -e .' "
    "in a source checkout), or use ProcessorConfig(landmark_correction='rule')."
)


class ModelContractError(ValueError):
    """The ONNX file does not match the expected input/output contract."""


def _check_io(kind: str, actual: list, expected: dict[str, tuple[int, ...]]) -> None:
    names = [node.name for node in actual]
    if sorted(names) != sorted(expected):
        raise ModelContractError(
            f"lash-line correction model {kind} mismatch: expected {sorted(expected)}, got {sorted(names)}"
        )
    for node in actual:
        shape = list(node.shape)
        batch_is_dynamic = bool(shape) and not isinstance(shape[0], int)
        if not batch_is_dynamic or tuple(shape[1:]) != expected[node.name]:
            raise ModelContractError(
                f"lash-line correction model {kind} '{node.name}' has shape {shape}, "
                f"expected [<dynamic batch>, {', '.join(map(str, expected[node.name]))}]"
            )
        if node.type != _FLOAT_TENSOR:
            raise ModelContractError(
                f"lash-line correction model {kind} '{node.name}' has type {node.type}, "
                f"expected {_FLOAT_TENSOR}"
            )


def validate_contract(session: "ort.InferenceSession") -> None:
    """Raise :class:`ModelContractError` unless *session* matches the contract.

    Args:
        session: A loaded ONNX Runtime session (anything exposing
            ``get_inputs()`` / ``get_outputs()`` with ``name`` and ``shape``).
    """
    _check_io("inputs", session.get_inputs(), _EXPECTED_INPUTS)
    _check_io("outputs", session.get_outputs(), _EXPECTED_OUTPUTS)


def _create_session(path: Path) -> "ort.InferenceSession":
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    validate_contract(session)
    return session


def _validate_fetched(path: Path, required: bool) -> None:
    """Check a fetched file against the contract.

    Args:
        path: The downloaded (not yet installed) model file.
        required: Fail when onnxruntime is missing instead of deferring the
            check to load time. ``pip install`` always has onnxruntime (it is
            a build requirement); a git hook may run under a Python without it.

    Raises:
        ImportError: if *required* and onnxruntime is not installed.
        ModelContractError: if the model does not match the contract.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        if required:
            raise
        logger.info("onnxruntime not available; the model contract is checked at load time")
        return
    _create_session(path)


# ── fetching (install / pull time) ───────────────────────────────────────────


def fetch(
    dest_dir: str | os.PathLike = PACKAGE_MODEL_DIR,
    *,
    require_validation: bool = False,
) -> Path:
    """Download the latest model from EyeContourRefiner ``main`` into *dest_dir*.

    The file is replaced atomically and only after it passed validation, so a
    failed or rejected fetch keeps the previous model.

    Args:
        dest_dir: Directory to place ``eye_contour_refiner.onnx`` in.
        require_validation: Refuse the model when onnxruntime is unavailable
            to check its contract (used by the ``pip install`` build hook).

    Returns:
        Path of the fetched model.

    Raises:
        OSError: on network or filesystem failure.
        ImportError: if *require_validation* and onnxruntime is not installed.
        ModelContractError: if the published model does not match the contract.
    """
    dest = Path(dest_dir) / MODEL_FILENAME
    download_file(
        MODEL_URL, dest, validate=lambda path: _validate_fetched(path, require_validation)
    )
    return dest


# ── loading (render time; never downloads) ───────────────────────────────────


def model_path() -> Path:
    """Return where the installed lash-line correction model lives."""
    return PACKAGE_MODEL_DIR / MODEL_FILENAME


def load_session(explicit: str | os.PathLike | None = None) -> "ort.InferenceSession":
    """Load (or reuse) the validated ONNX session for the lash-line correction model.

    Never downloads: the model is fetched at install / pull time (see the
    module docstring).

    Args:
        explicit: A caller-supplied model file used instead of the installed one.

    Raises:
        FileNotFoundError: if *explicit* does not exist.
        ModelContractError: if the model does not match the contract.
        RuntimeError: if the installed model is missing or cannot be loaded.
    """
    path = Path(explicit) if explicit is not None else model_path()
    key = str(path.resolve())
    with _LOAD_LOCK:
        cached = _SESSION_CACHE.get(key)
        if cached is not None:
            return cached
        session = _load(path, explicit is not None)
        _SESSION_CACHE[key] = session
        return session


def _load(path: Path, explicit: bool) -> "ort.InferenceSession":
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(f"lash-line correction model not found at explicit path: {path}")
        raise RuntimeError(f"The lash-line correction model is not installed. {_REINSTALL_HINT}")
    try:
        return _create_session(path)
    except ModelContractError as exc:
        if explicit:
            raise
        raise ModelContractError(
            f"The installed lash-line correction model does not match the interface this "
            f"face_makeup version expects ({exc}). Update face_makeup, or use "
            "ProcessorConfig(landmark_correction='rule')."
        ) from exc
    except Exception as exc:
        where = f" from {path}" if explicit else f". {_REINSTALL_HINT}"
        raise RuntimeError(f"Failed to load the lash-line correction model{where}") from exc


def main() -> int:
    """Entry point for the git hooks: refresh the model in this checkout."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        fetch()
    except Exception as exc:  # a hook must never break the pull
        print(f"warning: lash-line correction model not refreshed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
