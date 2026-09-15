from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import DEFAULT_CHECKPOINT, transcribe_images

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    return edit_distance(pred, truth) / max(1, len(truth))


def wer(pred: str, truth: str) -> float:
    return edit_distance(pred.split(), truth.split()) / max(1, len(truth.split()))


def plot_losses() -> None:
    log_path = ROOT / "logs" / "training_log.csv"
    if not log_path.exists():
        return
    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("CTC loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "loss_curves.png", dpi=160)
    plt.close()


def qualitative_grid(samples: list[dict], preds: list[str], out_path: Path) -> None:
    cells = []
    for sample, pred in zip(samples[:10], preds[:10]):
        img = Image.open(ROOT / sample["image_path"]).convert("RGB")
        img.thumbnail((360, 80))
        cell = Image.new("RGB", (380, 150), "white")
        cell.paste(img, (10, 10))
        draw = ImageDraw.Draw(cell)
        draw.text((10, 95), f"GT: {sample['modi_text'][:60]}", fill="black")
        draw.text((10, 118), f"PR: {pred[:60]}", fill="black")
        cells.append(cell)
    if not cells:
        return
    grid = Image.new("RGB", (760, 150 * ((len(cells) + 1) // 2)), "white")
    for i, cell in enumerate(cells):
        grid.paste(cell, ((i % 2) * 380, (i // 2) * 150))
    grid.save(out_path)


def evaluate_split(csv_path: Path, checkpoint: Path, limit: int | None = None) -> dict:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    if limit:
        rows = rows[:limit]
    images = [Image.open(ROOT / row["image_path"]) for row in rows]
    preds = transcribe_images(images, checkpoint)
    cer_values = [cer(pred, row["modi_text"]) for pred, row in zip(preds, rows)]
    wer_values = [wer(pred, row["modi_text"]) for pred, row in zip(preds, rows)]
    with (REPORTS_DIR / "synthetic_val_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "truth", "prediction", "cer", "wer"]
        )
        writer.writeheader()
        for row, pred, c, w in zip(rows, preds, cer_values, wer_values):
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "truth": row["modi_text"],
                    "prediction": pred,
                    "cer": c,
                    "wer": w,
                }
            )
    plt.figure(figsize=(8, 4))
    plt.hist(cer_values, bins=30)
    plt.xlabel("CER")
    plt.ylabel("samples")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "cer_histogram.png", dpi=160)
    plt.close()
    picked = random.sample(rows, min(10, len(rows)))
    picked_preds = transcribe_images(
        [Image.open(ROOT / r["image_path"]) for r in picked], checkpoint
    )
    qualitative_grid(picked, picked_preds, REPORTS_DIR / "qualitative_grid.png")
    return {
        "samples": len(rows),
        "cer": sum(cer_values) / max(1, len(cer_values)),
        "wer": sum(wer_values) / max(1, len(wer_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--split", type=Path, default=ROOT / "data" / "splits" / "val.csv"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")
    REPORTS_DIR.mkdir(exist_ok=True)
    plot_losses()
    metrics = evaluate_split(args.split, args.checkpoint, args.limit)
    summary = (
        "# Evaluation Summary\n\n"
        f"- split: `{args.split}`\n"
        f"- samples: {metrics['samples']}\n"
        f"- synthetic validation CER: {metrics['cer']:.4f}\n"
        f"- synthetic validation WER: {metrics['wer']:.4f}\n"
    )
    (REPORTS_DIR / "metrics.md").write_text(summary, encoding="utf-8")
    print(summary)
    real_raw = ROOT / "data" / "real" / "raw_scans"
    real_ann = ROOT / "data" / "real" / "annotated_test"
    real_images = [
        p
        for p in real_raw.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ]
    real_labels = [p for p in real_ann.glob("*.csv")]
    if not real_images or not real_labels:
        print(
            "Real manuscript evaluation skipped: add MODI-HHDoc scans and hand-transcribed CSV labels under data/real/."
        )


if __name__ == "__main__":
    main()
