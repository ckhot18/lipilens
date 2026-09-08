from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load a manuscript image as grayscale.

    Returns:
        Grayscale image as a NumPy uint8 array.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"OpenCV could not read: {image_path}")

    return image


def denoise_median(
    image: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Remove small impulse-like noise while preserving handwriting edges.

    kernel_size must be an odd positive integer.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    return cv2.medianBlur(image, kernel_size)