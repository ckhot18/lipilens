from pathlib import Path

import cv2

from restoration.pipeline import load_image, denoise_median


INPUT_PATH = Path("data/evaluation/sample_001.png")
OUTPUT_DIR = Path("data/processed")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = load_image(INPUT_PATH)

    restored = denoise_median(
        original,
        kernel_size=3,
    )

    output_path = OUTPUT_DIR / "sample_001_median3.png"

    success = cv2.imwrite(str(output_path), restored)

    if not success:
        raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"Original shape: {original.shape}")
    print(f"Saved restored image: {output_path}")


if __name__ == "__main__":
    main()