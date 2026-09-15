from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CRNN

ROOT = Path(__file__).resolve().parent.parent

VOCAB_PATH = ROOT / "data" / "vocab.json"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"


def load_vocab() -> dict:
    with VOCAB_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def image_to_tensor(image: Image.Image, height: int = 32) -> torch.Tensor:
    image = image.convert("L")
    width = max(8, round(image.width * (height / image.height)))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
        height, width
    )
    return (1.0 - pixels.float().div(255.0)).unsqueeze(0).unsqueeze(0)


def greedy_decode(
    logits: torch.Tensor, idx_to_char: list[str], blank_index: int = 0
) -> list[str]:
    paths = logits.argmax(dim=2).detach().cpu().tolist()
    decoded = []
    for path in paths:
        chars = []
        previous = None
        for idx in path:
            if idx != previous and idx != blank_index:
                chars.append(idx_to_char[idx])
            previous = idx
        decoded.append("".join(chars))
    return decoded


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT, device: torch.device | None = None
) -> tuple[CRNN, dict, torch.device]:
    vocab = load_vocab()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(vocab["vocab_size"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, vocab, device


@torch.no_grad()
def transcribe_images(
    images: list[Image.Image], checkpoint_path: Path = DEFAULT_CHECKPOINT
) -> list[str]:
    model, vocab, device = load_model(checkpoint_path)
    outputs = []
    for image in images:
        tensor = image_to_tensor(image).to(device)
        logits = model(tensor)
        outputs.extend(
            greedy_decode(logits, vocab["idx_to_char"], vocab["blank_index"])
        )
    return outputs


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python infer.py path/to/line.png [checkpoint.pt]")
    checkpoint = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHECKPOINT
    text = transcribe_images([Image.open(sys.argv[1])], checkpoint)[0]
    print(text)


if __name__ == "__main__":
    main()
