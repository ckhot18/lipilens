from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"


def cnn_time_steps(width: int) -> int:
    """Return the CTC time dimension after the two width-halving pools."""
    return max(1, width // 4)


def load_vocab_size(vocab_path: Path = VOCAB_PATH) -> int:
    with vocab_path.open(encoding="utf-8") as f:
        vocab = json.load(f)
    return int(vocab["vocab_size"])


class CRNN(nn.Module):
    """Compact CRNN for line-level Modi OCR with CTC loss."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.classifier = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = features.squeeze(2).permute(0, 2, 1)
        sequence, _ = self.lstm(features)
        return self.classifier(sequence)


def build_model(vocab_path: Path = VOCAB_PATH) -> CRNN:
    return CRNN(load_vocab_size(vocab_path))
