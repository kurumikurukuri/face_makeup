"""
Model file location and download helpers.

Models are never committed to the repository. :func:`resolve_model_dir`
resolves the hair segmentation model's directory through a priority chain;
:func:`download_file` downloads the lash-line correction model atomically.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable, Union
from urllib import request

if TYPE_CHECKING:
    from http.client import HTTPResponse

logger = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 16
_DOWNLOAD_TIMEOUT_SEC = 60
# Whole-transfer limits: the per-read timeout above does not bound a trickling
# connection, and nothing else bounds the payload written to disk.
_DOWNLOAD_DEADLINE_SEC = 600
_MAX_DOWNLOAD_BYTES = 64 << 20
# mkstemp creates 0600 files; models must stay readable in shared caches.
_MODEL_FILE_MODE = 0o644

# Indirection so tests can control time without patching the time module.
_clock = time.monotonic

# Repository root when running from a source checkout (src/face_makeup/core/ -> root).
_SOURCE_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def resolve_model_dir(explicit: Union[str, os.PathLike, None] = None) -> Path:
    """Resolve the directory that holds downloaded model files.

    Priority chain (highest first):

    1. *explicit* — caller-supplied path (``ProcessorConfig.model_dir``).
    2. ``FACE_MAKEUP_MODEL_DIR`` environment variable.
    3. Source-checkout ``models/`` directory, detected by checking whether the
       ``__file__``-relative path exists. This keeps ``pip install -e .`` and
       direct ``PYTHONPATH`` usage working without extra configuration.
    4. Platform cache directory (always writable for auto-download):

       - **Windows**: ``%LOCALAPPDATA%/face_makeup/models/``
       - **macOS**: ``~/Library/Caches/face_makeup/models/``
       - **Linux**: ``$XDG_CACHE_HOME/face_makeup/models/`` (falls back to
         ``~/.cache/face_makeup/models/``).

    Returns:
        Absolute path to the model directory (created if it doesn't exist
        for the platform-cache fallback).
    """
    if explicit:
        return Path(explicit).resolve()

    env_dir = os.environ.get("FACE_MAKEUP_MODEL_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    if _SOURCE_MODELS_DIR.is_dir():
        return _SOURCE_MODELS_DIR

    if sys.platform == "win32":
        base = Path(
            os.environ.get(
                "LOCALAPPDATA",
                str(Path.home() / "AppData" / "Local"),
            )
        )
        cache_dir = base / "face_makeup" / "models"
    elif sys.platform == "darwin":
        cache_dir = Path.home() / "Library" / "Caches" / "face_makeup" / "models"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            cache_dir = Path(xdg_cache) / "face_makeup" / "models"
        else:
            cache_dir = Path.home() / ".cache" / "face_makeup" / "models"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _check_size(url: str, size: int) -> None:
    if size > _MAX_DOWNLOAD_BYTES:
        raise OSError(
            f"Download from {url} exceeds the {_MAX_DOWNLOAD_BYTES}-byte limit"
        )


def _stream(resp: "HTTPResponse", out: IO[bytes], url: str) -> None:
    """Copy *resp* into *out*, enforcing the size limit and the total deadline."""
    length = resp.headers.get("Content-Length")
    if length is not None and length.isdigit():
        _check_size(url, int(length))
    deadline = _clock() + _DOWNLOAD_DEADLINE_SEC
    written = 0
    for chunk in iter(lambda: resp.read(_DOWNLOAD_CHUNK_BYTES), b""):
        written += len(chunk)
        _check_size(url, written)
        if _clock() > deadline:
            raise TimeoutError(
                f"Download from {url} did not finish within {_DOWNLOAD_DEADLINE_SEC} s"
            )
        out.write(chunk)


def download_file(
    url: str,
    dest: Path,
    validate: Callable[[Path], None],
) -> None:
    """Download *url* to *dest* atomically, validating before it becomes visible.

    The payload is streamed into a uniquely named temp file in the destination
    directory (so concurrent downloads never clobber each other), passed to
    *validate*, and only then moved into place with ``os.replace``. A failed
    or rejected download never leaves a partial file at *dest*.

    Args:
        url: Source URL (https).
        dest: Final path of the downloaded file.
        validate: Callback that raises if the downloaded file is unusable.

    Raises:
        OSError: on network or filesystem failure (``URLError`` is an OSError),
            or when the payload exceeds ``_MAX_DOWNLOAD_BYTES``.
        TimeoutError: when the transfer exceeds ``_DOWNLOAD_DEADLINE_SEC``.
        Exception: whatever *validate* raises for a rejected payload.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".download", dir=dest.parent
    )
    tmp_path = Path(tmp_name)
    logger.info("Downloading %s from %s ...", dest.name, url)
    try:
        with os.fdopen(fd, "wb") as out, request.urlopen(
            request.Request(url), timeout=_DOWNLOAD_TIMEOUT_SEC
        ) as resp:
            _stream(resp, out, url)
        validate(tmp_path)
        os.chmod(tmp_path, _MODEL_FILE_MODE)
        os.replace(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
