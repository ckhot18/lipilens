"""
build_vocab.py

Scans the training labels and builds the fixed character vocabulary the
model will learn to output. This MUST be built before writing the model,
because the CRNN's final layer size = len(vocab) + 1 (the +1 is the CTC
"blank" token, a required special symbol, not a real character).

Why this matters: if you add more sentences to the corpus later and
regenerate the dataset, you must re-run this script and RETRAIN from
scratch if the vocabulary changes size - the model architecture is tied
to this exact vocab file. So finalize your corpus before this step.

Usage:
    python3 build_vocab.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
LABELS_PATH = ROOT / "data" / "synthetic" / "labels.csv"
VOCAB_PATH = ROOT / "data" / "vocab.json"

BLANK_TOKEN = "<blank>"  # required by CTC loss, must be index 0


def main():
    chars = set()
    with open(LABELS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chars.update(row["modi_text"])

    sorted_chars = sorted(chars)  # deterministic order = reproducible vocab
    idx_to_char = [BLANK_TOKEN] + sorted_chars
    char_to_idx = {ch: i for i, ch in enumerate(idx_to_char)}

    vocab = {
        "blank_index": 0,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(idx_to_char),
    }

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Vocabulary size (incl. blank token): {len(idx_to_char)}")
    characters = "".join(sorted_chars)
    print(f"Characters found: {characters.encode('unicode_escape').decode('ascii')}")
    print(f"Saved to: {VOCAB_PATH}")


if __name__ == "__main__":
    main()
