"""Installed-dependency checks for the OpenCV < 5 pin.

OpenCV 5 changes ``cv2.warpAffine`` rounding, which shifts the lash-line
correction model's input patches (``core/eyeliner_patch.py``). The ``cv2``
module can come from several distributions at once: face_makeup depends on
``opencv-python-headless`` and mediapipe pulls ``opencv-contrib-python``
unpinned. Whichever is installed last wins, so every one of them has to stay
below 5, and face_makeup has to declare that cap for each.
"""
from __future__ import annotations

from importlib.metadata import (
    PackageNotFoundError,
    packages_distributions,
    requires,
    version,
)

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_FIRST_BLOCKED_OPENCV = Version("5.0.0")
_CAPPED_DISTRIBUTIONS = ("opencv-python-headless", "opencv-contrib-python")


def test_every_installed_cv2_distribution_is_below_5() -> None:
    """No installed distribution that ships cv2 may be OpenCV 5 or later."""
    dists = sorted(set(packages_distributions().get("cv2", [])))
    assert dists, "no installed distribution provides cv2"
    too_new = {d: version(d) for d in dists if Version(version(d)) >= _FIRST_BLOCKED_OPENCV}
    assert not too_new, f"OpenCV 5+ installed: {too_new}"


@pytest.mark.parametrize("name", _CAPPED_DISTRIBUTIONS)
def test_face_makeup_declares_opencv_cap(name: str) -> None:
    """face_makeup's installed metadata caps each cv2 distribution below 5."""
    try:
        declared = [Requirement(r) for r in requires("face_makeup") or []]
    except PackageNotFoundError:
        pytest.fail("face_makeup is not installed; run 'pip install -e .'")
    matching = [
        r for r in declared
        if canonicalize_name(r.name) == name and r.marker is None
    ]
    assert matching, f"face_makeup does not depend on {name}"
    for req in matching:
        assert not req.specifier.contains(_FIRST_BLOCKED_OPENCV, prereleases=True), (
            f"{req} allows OpenCV 5"
        )
