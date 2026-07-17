"""
Pytest configuration and shared fixtures.
Test images must be placed manually in tests/test_images/.
Use CC0 / Public Domain images, or images you own the rights to.
Free sources: https://www.pexels.com, https://pixabay.com, https://unsplash.com
Recommended: 640x480 or larger color portrait (PNG or JPEG).
``test_image`` / ``test_image_path`` will fail with a clear message if no
images are found; ``sample_face_image`` falls back to a synthetic image.
"""
import sys
from pathlib import Path

# Add src directory to path for test imports when package is not installed
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture(scope="session")
def test_images_dir():
    """Path to test images directory."""
    return Path(__file__).parent / "test_images"


@pytest.fixture(scope="session", autouse=True)
def ensure_test_images(test_images_dir):
    """Ensure test images directory exists.

    Runs once per session. If no test images are found in
    ``tests/test_images/``, tests that depend on ``test_image`` /
    ``test_image_path`` / ``sample_face_image`` will skip or fall back
    gracefully.
    """
    test_images_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def test_image(ensure_test_images, test_images_dir):
    """Load first available test image as an RGB numpy array."""
    import cv2

    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files = sorted(test_images_dir.glob(pattern))
        if files:
            img = cv2.imread(str(files[0]))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pytest.fail("No test images found")


@pytest.fixture
def test_image_path(ensure_test_images, test_images_dir):
    """Get path string to the first available test image."""
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files = sorted(test_images_dir.glob(pattern))
        if files:
            return str(files[0])

    pytest.fail("No test images found")


@pytest.fixture
def sample_face_image(test_images_dir):
    """Load a sample face image for testing."""
    import cv2
    import glob

    # Find any jpg/png in test_images
    patterns = [
        str(test_images_dir / "*.jpg"),
        str(test_images_dir / "*.png"),
        str(test_images_dir / "*.jpeg"),
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            img = cv2.imread(files[0])
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pytest.skip("No test images found in %s" % test_images_dir)


@pytest.fixture
def synthetic_image():
    """Generate a synthetic RGB image for testing."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = [200, 150, 150]  # Face-colored region
    return image


@pytest.fixture
def sample_mask():
    """Generate a sample float32 mask for testing."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[40:60, 40:60] = 1.0
    return mask
