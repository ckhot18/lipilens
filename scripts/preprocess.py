from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover - fallback for environments missing OpenCV
    cv2 = None


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if cv2 is None:
        return np.array(Image.fromarray(image).convert("L"))
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return np.array(Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3)))
    return cv2.fastNlMeansDenoising(
        gray, None, h=15, templateWindowSize=7, searchWindowSize=21
    )


def binarize(gray: np.ndarray) -> np.ndarray:
    if cv2 is None:
        threshold = int(np.mean(gray) * 0.9)
        return np.where(gray > threshold, 255, 0).astype(np.uint8)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )


def deskew(binary: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return binary
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 20:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 15:
        return binary
    h, w = binary.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        binary, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255
    )


def segment_lines(binary: np.ndarray, min_line_height: int = 12) -> list[np.ndarray]:
    ink = binary < 200
    projection = ink.sum(axis=1)
    threshold = max(2, int(binary.shape[1] * 0.01))
    active = projection > threshold
    ranges = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            if y - start >= min_line_height:
                ranges.append((start, y))
            start = None
    if start is not None and len(active) - start >= min_line_height:
        ranges.append((start, len(active)))
    if not ranges:
        return [binary]
    lines = []
    for start, end in ranges:
        top = max(0, start - 4)
        bottom = min(binary.shape[0], end + 4)
        crop = binary[top:bottom, :]
        cols = np.where((crop < 200).sum(axis=0) > 0)[0]
        if cols.size:
            crop = crop[:, max(0, cols[0] - 8) : min(crop.shape[1], cols[-1] + 9)]
        lines.append(crop)
    return lines


def preprocess_image(path: str | Path) -> tuple[np.ndarray, list[np.ndarray]]:
    if cv2 is None:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        gray = np.array(Image.open(path).convert("L"))
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        gray = to_grayscale(image)
    cleaned = deskew(binarize(denoise(gray)))
    return cleaned, segment_lines(cleaned)


def preprocess_pil(image: Image.Image) -> tuple[Image.Image, list[Image.Image]]:
    arr = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if cv2 is not None else arr
    gray = to_grayscale(bgr)
    cleaned = deskew(binarize(denoise(gray)))
    lines = segment_lines(cleaned)
    return Image.fromarray(cleaned), [Image.fromarray(line) for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out-dir", default="preprocess_out")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned, lines = preprocess_image(args.image)
    Image.fromarray(cleaned).save(out_dir / "cleaned.png")
    for i, line in enumerate(lines):
        Image.fromarray(line).save(out_dir / f"line_{i:03d}.png")
    print(f"saved cleaned image and {len(lines)} line(s) to {out_dir}")


if __name__ == "__main__":
    main()
