"""
render_text.py

Renders a Modi Unicode text string into a clean line image using the
Noto Sans Modi font. This is the "clean" image before any manuscript-style
degradation is applied (that's degrade_image.py).

Design choices, and why:
- Variable canvas width based on text length (manuscripts don't have
  fixed-width lines).
- Random font size within a realistic range -> teaches the model
  scale-invariance.
- Random horizontal stretch/shear -> simulates natural handwriting-style
  slant variation, since we only have one font weight available.
- Random stroke width (PIL's `stroke_width`) -> crude but effective stand-in
  for bold/thin pen strokes, since no bold font file exists for Modi yet.
- Padding on all sides -> real manuscript lines are never edge-to-edge.
"""

import random

from PIL import Image, ImageDraw, ImageFont

FONT_SIZE_RANGE = (36, 56)
STROKE_WIDTH_RANGE = (0, 1)
PAD_X_RANGE = (15, 40)
PAD_Y_RANGE = (10, 20)
SHEAR_RANGE = (-0.08, 0.08)  # slight horizontal slant


def render_line(text: str, font_path: str, seed: int | None = None) -> Image.Image:
    """Render a single line of Modi text into a clean RGB PIL image."""
    rng = random.Random(seed)

    font_size = rng.randint(*FONT_SIZE_RANGE)
    stroke_width = rng.randint(*STROKE_WIDTH_RANGE)
    pad_x = rng.randint(*PAD_X_RANGE)
    pad_y = rng.randint(*PAD_Y_RANGE)

    font = ImageFont.truetype(font_path, font_size)

    # Measure text bounding box first on a throwaway image
    dummy = Image.new("L", (10, 10), 255)
    ddraw = ImageDraw.Draw(dummy)
    bbox = ddraw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    canvas_w = text_w + 2 * pad_x
    canvas_h = text_h + 2 * pad_y

    img = Image.new("L", (canvas_w, canvas_h), 255)  # white background
    draw = ImageDraw.Draw(img)
    draw.text(
        (pad_x - bbox[0], pad_y - bbox[1]),
        text,
        font=font,
        fill=0,  # black ink
        stroke_width=stroke_width,
    )

    # Apply a small random shear to simulate handwriting slant
    shear = rng.uniform(*SHEAR_RANGE)
    if abs(shear) > 1e-3:
        img = img.transform(
            (canvas_w + int(abs(shear) * canvas_h), canvas_h),
            Image.AFFINE,
            (1, shear, -shear * canvas_h if shear < 0 else 0, 0, 1, 0),
            fillcolor=255,
        )

    return img.convert("RGB")


if __name__ == "__main__":
    # quick manual test
    sample = "𑘡𑘦𑘭𑘿𑘎𑘰𑘨"  # namaskar in Modi
    img = render_line(sample, "../fonts/NotoSansModi-Regular.ttf", seed=1)
    img.save("../fonts/render_line_test.png")
    print("saved test render, size:", img.size)
