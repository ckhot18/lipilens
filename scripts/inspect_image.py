from pathlib import Path

import cv2


IMAGE_PATH = Path("data/evaluation/sample_001.png")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"OpenCV could not read: {IMAGE_PATH}")

    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    file_size_kb = IMAGE_PATH.stat().st_size / 1024

    print("=" * 50)
    print("LipiLens - Image Inspection")
    print("=" * 50)
    print(f"Path:       {IMAGE_PATH}")
    print(f"Width:      {width}px")
    print(f"Height:     {height}px")
    print(f"Channels:   {channels}")
    print(f"Data type:  {image.dtype}")
    print(f"File size:  {file_size_kb:.2f} KB")
    print(f"Min pixel:  {image.min()}")
    print(f"Max pixel:  {image.max()}")
    print("=" * 50)


if __name__ == "__main__":
    main()