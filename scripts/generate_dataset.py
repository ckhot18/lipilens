"""
generate_dataset.py

Phase 2 main driver. Reads the Modi manifest (from Phase 1), and for each
line of text, generates multiple degraded image variants. Writes:

    data/synthetic/images/*.png
    data/synthetic/labels.csv          (image_path, marathi_text, modi_text)
    data/splits/train.csv, val.csv     (90/10 split, by SENTENCE not by
                                         image, so no near-duplicate leakage
                                         between train and val)

Usage:
    python3 generate_dataset.py --variants 40
"""

import argparse
import csv
import random
from pathlib import Path

from degrade_image import random_degrade
from render_text import render_line

ROOT = Path(__file__).parent.parent
MANIFEST_PATH = ROOT / "data" / "manifests" / "modi_manifest.csv"
LEGACY_MANIFEST_PATH = ROOT / "data" / "modi_manifest.csv"
FONT_PATH = ROOT / "fonts" / "NotoSansModi-Regular.ttf"
IMAGES_DIR = ROOT / "data" / "synthetic" / "images"
LABELS_PATH = ROOT / "data" / "synthetic" / "labels.csv"
SPLITS_DIR = ROOT / "data" / "splits"


def load_manifest():
    rows = []
    manifest_path = MANIFEST_PATH if MANIFEST_PATH.exists() else LEGACY_MANIFEST_PATH
    with open(manifest_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def generate(variants_per_line: int, val_fraction: float = 0.1, seed: int = 42):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_manifest()
    rng = random.Random(seed)

    all_records = []  # (image_path, marathi_text, modi_text, sentence_id)

    for row in rows:
        sentence_id = int(row["id"])
        marathi_text = row["marathi_text"]
        modi_text = row["modi_text"]

        if not modi_text.strip():
            continue

        for v in range(variants_per_line):
            variant_seed = seed * 100000 + sentence_id * 1000 + v
            try:
                clean_img = render_line(modi_text, str(FONT_PATH), seed=variant_seed)
                final_img = random_degrade(clean_img, seed=variant_seed)
            except Exception as e:
                print(f"Skipping sentence {sentence_id} variant {v}: {e}")
                continue

            fname = f"line_{sentence_id:04d}_v{v:03d}.png"
            fpath = IMAGES_DIR / fname
            final_img.save(fpath)

            all_records.append(
                {
                    "image_path": f"data/synthetic/images/{fname}",
                    "marathi_text": marathi_text,
                    "modi_text": modi_text,
                    "sentence_id": sentence_id,
                }
            )

    # Write full labels file
    with open(LABELS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "marathi_text", "modi_text", "sentence_id"]
        )
        writer.writeheader()
        writer.writerows(all_records)

    # Split by sentence_id (not by image) so train/val never share the same
    # underlying sentence - this gives a more honest validation signal.
    unique_sentence_ids = sorted(set(r["sentence_id"] for r in all_records))
    rng.shuffle(unique_sentence_ids)
    n_val = max(1, int(len(unique_sentence_ids) * val_fraction))
    val_ids = set(unique_sentence_ids[:n_val])
    train_ids = set(unique_sentence_ids[n_val:])

    train_records = [r for r in all_records if r["sentence_id"] in train_ids]
    val_records = [r for r in all_records if r["sentence_id"] in val_ids]

    for name, records in [("train", train_records), ("val", val_records)]:
        with open(SPLITS_DIR / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_path", "marathi_text", "modi_text", "sentence_id"]
            )
            writer.writeheader()
            writer.writerows(records)

    print(f"Generated {len(all_records)} images total.")
    print(f"  Train: {len(train_records)} images from {len(train_ids)} sentences")
    print(f"  Val:   {len(val_records)} images from {len(val_ids)} sentences")
    print(f"Images dir: {IMAGES_DIR}")
    print(f"Labels file: {LABELS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        type=int,
        default=40,
        help="Number of degraded image variants to generate per sentence",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    generate(args.variants, args.val_fraction)
