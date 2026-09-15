from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CRNN, cnn_time_steps

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
VAL_CSV = ROOT / "data" / "splits" / "val.csv"


def load_vocab(vocab_path: Path = VOCAB_PATH) -> dict:
    with vocab_path.open(encoding="utf-8") as f:
        return json.load(f)


class ModiLineDataset(Dataset):
    def __init__(self, csv_path: Path, vocab: dict, image_height: int = 32):
        self.root = ROOT
        self.image_height = image_height
        self.char_to_idx = vocab["char_to_idx"]
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for row {index}: {image_path}")
        image = Image.open(image_path).convert("L")
        width = max(8, round(image.width * (self.image_height / image.height)))
        image = image.resize((width, self.image_height), Image.Resampling.BILINEAR)
        pixels = torch.frombuffer(
            bytearray(image.tobytes()), dtype=torch.uint8
        ).reshape(self.image_height, width)
        tensor = pixels.float().div(255.0)
        tensor = (1.0 - tensor).unsqueeze(0)
        target = torch.tensor(
            [self.char_to_idx[ch] for ch in row["modi_text"]], dtype=torch.long
        )
        return {"image": tensor, "target": target, "width": width, "row": row}


def collate_batch(
    samples: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_width = max(sample["image"].shape[-1] for sample in samples)
    images = []
    targets = []
    target_lengths = []
    input_lengths = []
    for sample in samples:
        image = sample["image"]
        pad_width = max_width - image.shape[-1]
        images.append(F.pad(image, (0, pad_width, 0, 0), value=0.0))
        targets.append(sample["target"])
        target_lengths.append(len(sample["target"]))
        input_lengths.append(cnn_time_steps(sample["width"]))
    return (
        torch.stack(images),
        torch.cat(targets),
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
    )


def make_loader(
    csv_path: Path, vocab: dict, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ModiLineDataset(csv_path, vocab),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model, loader, criterion, optimizer, scaler, device, max_batches: int | None = None
) -> float:
    model.train()
    total = 0.0
    for batch_index, (images, targets, input_lengths, target_lengths) in enumerate(
        loader
    ):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            if int(input_lengths.max()) > log_probs.shape[0]:
                raise ValueError("CTC input length exceeds CRNN output length")
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        if max_batches and batch_index + 1 >= max_batches:
            break
    divisor = min(len(loader), max_batches) if max_batches else len(loader)
    return total / max(1, divisor)


@torch.no_grad()
def validate(model, loader, criterion, device, max_batches: int | None = None) -> float:
    model.eval()
    total = 0.0
    for batch_index, (images, targets, input_lengths, target_lengths) in enumerate(
        loader
    ):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)
        if int(input_lengths.max()) > log_probs.shape[0]:
            raise ValueError("CTC input length exceeds CRNN output length")
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total += loss.item()
        if max_batches and batch_index + 1 >= max_batches:
            break
    divisor = min(len(loader), max_batches) if max_batches else len(loader)
    return total / max(1, divisor)


def save_checkpoint(
    path: Path, model, optimizer, epoch: int, val_loss: float, vocab: dict
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "vocab_size": vocab["vocab_size"],
        },
        path,
    )


def train_once(args, batch_size: int, learning_rate: float) -> tuple[float, int]:
    vocab = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Refusing to train on CPU by default."
        )
    train_loader = make_loader(TRAIN_CSV, vocab, batch_size, True)
    val_loader = make_loader(VAL_CSV, vocab, batch_size, False)
    model = CRNN(vocab["vocab_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=vocab["blank_index"], zero_infinity=True)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    log_path = LOG_DIR / "training_log.csv"
    if not args.smoke_test:
        LOG_DIR.mkdir(exist_ok=True)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        if not log_path.exists():
            log_path.write_text(
                "epoch,train_loss,val_loss,timestamp\n", encoding="utf-8"
            )
    best_val = float("inf")
    stale_epochs = 0
    peak_mb = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.max_train_batches,
        )
        val_loss = validate(model, val_loader, criterion, device, args.max_val_batches)
        if device.type == "cuda":
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 1024**2)
        if not args.smoke_test:
            with log_path.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(
                    [
                        epoch,
                        train_loss,
                        val_loss,
                        datetime.now().isoformat(timespec="seconds"),
                    ]
                )
            save_checkpoint(
                CHECKPOINT_DIR / "last.pt", model, optimizer, epoch, val_loss, vocab
            )
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            if not args.smoke_test:
                shutil.copy2(CHECKPOINT_DIR / "last.pt", CHECKPOINT_DIR / "best.pt")
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_val={best_val:.4f} peak_vram_mb={peak_mb:.1f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} stale epochs.")
            break
    return best_val, batch_size


def dry_run(args) -> None:
    vocab = load_vocab()
    loader = make_loader(TRAIN_CSV, vocab, args.batch_size, False)
    images, targets, input_lengths, target_lengths = next(iter(loader))
    model = CRNN(vocab["vocab_size"])
    with torch.no_grad():
        logits = model(images)
    print("dry_run images:", tuple(images.shape))
    print("dry_run logits:", tuple(logits.shape))
    print("dry_run targets:", tuple(targets.shape))
    print(
        "dry_run input_lengths:", input_lengths[: min(5, len(input_lengths))].tolist()
    )
    print(
        "dry_run target_lengths:",
        target_lengths[: min(5, len(target_lengths))].tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train and validation batch without writing checkpoints or logs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 1
        args.max_val_batches = 1
    batch_size = args.batch_size
    lr = args.lr
    for retry in range(4):
        try:
            best_val, final_batch = train_once(args, batch_size, lr)
            print(
                f"Training complete. best_val_loss={best_val:.4f} batch_size={final_batch}"
            )
            return
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and retry < 3 and batch_size > 1:
                print(
                    f"CUDA OOM at batch_size={batch_size}; retrying with {batch_size // 2}"
                )
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            raise
        except FloatingPointError:
            if retry < 3:
                lr /= 10
                print(f"Non-finite loss; retrying from scratch with lr={lr}")
                continue
            raise


if __name__ == "__main__":
    main()
