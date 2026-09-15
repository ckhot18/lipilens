"""
degrade_image.py

Takes a clean rendered text-line image and makes it look like a genuinely
old, scanned, damaged manuscript. This is the single most important file
for making the model generalize to real Modi documents instead of only
recognizing crisp computer-rendered text.

Each function is a self-contained degradation. `random_degrade()` picks a
random subset and random strengths, so every generated image is uniquely
damaged (this is what makes the "synthetic data" actually diverse instead
of just being the same clean render 5000 times).
"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal environments
    cv2 = None


def add_gaussian_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(3, 18)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([1, 1, 3, 3, 5])  # bias toward mild blur
    if k <= 1:
        return img
    if cv2 is None:
        return np.array(
            Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=k / 2))
        )
    return cv2.GaussianBlur(img, (k, k), 0)


def add_ink_bleed(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulates ink spreading into old paper fibers (dilate dark strokes)."""
    if rng.random() < 0.5:
        return img
    if cv2 is None:
        return np.array(Image.fromarray(img).filter(ImageFilter.MinFilter(3)))
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(img, kernel, iterations=1)  # erode on white-bg = dilate ink


def add_faded_ink(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulates faded/patchy ink by randomly lightening some ink pixels."""
    if rng.random() < 0.5:
        return img
    out = img.copy().astype(np.float32)
    fade_mask = np.random.default_rng(rng.randint(0, 1_000_000)).uniform(
        0.5, 1.0, img.shape[:2]
    )
    if img.ndim == 3:
        fade_mask = fade_mask[..., None]
    out = 255 - (255 - out) * fade_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def add_stains(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random soft brownish/grey blotches, like paper staining."""
    if rng.random() < 0.6:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    n_stains = rng.randint(1, 3)
    overlay = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_stains):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        radius = rng.randint(min(h, w) // 4, max(min(h, w) // 2, 5))
        yy, xx = np.ogrid[:h, :w]
        dist = (xx - cx) ** 2 + (yy - cy) ** 2
        blob = np.exp(-dist / (2 * radius**2)) * rng.uniform(15, 45)
        overlay += blob
    if out.ndim == 3:
        overlay = overlay[..., None]
    out = np.clip(out.astype(np.float32) - overlay, 0, 255).astype(np.uint8)
    return out


def add_paper_texture(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Low-frequency background brightness variation, like uneven old paper."""
    h, w = img.shape[:2]
    gen = np.random.default_rng(rng.randint(0, 1_000_000))
    low_res = gen.uniform(-1, 1, (8, 8)).astype(np.float32)
    if cv2 is None:
        texture = np.array(
            Image.fromarray(low_res).resize((w, h), Image.Resampling.BICUBIC)
        )
    else:
        texture = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
    texture = texture * rng.uniform(5, 20)
    if img.ndim == 3:
        texture = texture[..., None]
    out = np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)
    return out


def add_rotation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    angle = rng.uniform(-2.5, 2.5)
    if cv2 is None:
        return np.array(
            Image.fromarray(img).rotate(
                angle, resample=Image.Resampling.BILINEAR, fillcolor=255
            )
        )
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


def adjust_contrast_brightness(img_pil: Image.Image, rng: random.Random) -> Image.Image:
    img_pil = ImageEnhance.Contrast(img_pil).enhance(rng.uniform(0.7, 1.3))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(rng.uniform(0.85, 1.1))
    return img_pil


def jpeg_artifact(img_pil: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.6:
        return img_pil
    import io

    quality = rng.randint(30, 70)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_degrade(img_pil: Image.Image, seed: int | None = None) -> Image.Image:
    """Apply a randomized pipeline of degradations to a clean line image."""
    rng = random.Random(seed)
    img = np.array(img_pil.convert("L"))

    img = add_paper_texture(img, rng)
    img = add_stains(img, rng)
    img = add_faded_ink(img, rng)
    img = add_ink_bleed(img, rng)
    img = add_rotation(img, rng)
    img = add_gaussian_noise(img, rng)
    img = add_blur(img, rng)

    img_pil = Image.fromarray(img).convert("RGB")
    img_pil = adjust_contrast_brightness(img_pil, rng)
    img_pil = jpeg_artifact(img_pil, rng)
    return img_pil


if __name__ == "__main__":
    from render_text import render_line

    sample = "𑘡𑘦𑘭𑘿𑘎𑘰𑘨"
    clean = render_line(sample, "../fonts/NotoSansModi-Regular.ttf", seed=1)
    for i in range(5):
        degraded = random_degrade(clean, seed=i)
        degraded.save(f"../fonts/degrade_test_{i}.png")
    print("saved 5 degraded test variants")
