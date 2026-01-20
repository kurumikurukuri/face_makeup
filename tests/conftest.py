"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

# Add src directory to path for test imports when package is not installed
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
import numpy as np


@pytest.fixture
def test_images_dir():
    """Path to test images directory."""
    return Path(__file__).parent / "test_images"


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

    # Return a synthetic image if no test images found
    return np.zeros((480, 640, 3), dtype=np.uint8)


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
