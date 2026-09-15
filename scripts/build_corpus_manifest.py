"""
build_corpus_manifest.py

Phase 1 output builder.

Reads the Marathi (Devanagari) corpus line by line, transliterates each
line into Modi Unicode, and writes a manifest CSV that Phase 2 (synthetic
image rendering) will consume directly:

    id, marathi_text, modi_text, char_len

Also prints a coverage report: how many lines transliterated cleanly vs.
had at least one unmapped character, and the actual unmapped characters
found (so you can extend devanagari_to_modi.py if your real corpus later
uses uncommon Marathi characters, e.g. nukta consonants in loanwords).

Usage:
    python3 build_corpus_manifest.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from devanagari_to_modi import transliterate

ROOT = Path(__file__).parent.parent
CORPUS_PATH = ROOT / "data" / "raw_corpus" / "marathi_corpus.txt"
LEGACY_CORPUS_PATH = ROOT / "data" / "marathi_corpus.txt"
MANIFEST_PATH = ROOT / "data" / "manifests" / "modi_manifest.csv"
LEGACY_MANIFEST_PATH = ROOT / "data" / "modi_manifest.csv"


def main():
    corpus_path = CORPUS_PATH if CORPUS_PATH.exists() else LEGACY_CORPUS_PATH
    lines = [
        l.strip()
        for l in corpus_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    rows = []
    clean_count = 0
    warning_char_counter = Counter()

    for i, line in enumerate(lines):
        modi_text, warnings = transliterate(line)
        if warnings:
            for w in warnings:
                # extract the character from the warning message
                ch = w.split("(")[-1].split("!r")[0].strip("')")
                warning_char_counter[ch] += 1
        else:
            clean_count += 1
        rows.append(
            {
                "id": i,
                "marathi_text": line,
                "modi_text": modi_text,
                "char_len": len(modi_text),
            }
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    for manifest_path in (MANIFEST_PATH, LEGACY_MANIFEST_PATH):
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "marathi_text", "modi_text", "char_len"]
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Read {len(lines)} lines from corpus.")
    print(f"Transliterated cleanly: {clean_count}/{len(lines)}")
    if warning_char_counter:
        print("Unmapped characters found (first 20):")
        for ch, count in warning_char_counter.most_common(20):
            print(f"   {ch!r}  (x{count})")
    else:
        print("No unmapped characters - full coverage on this corpus.")
    print(f"\nManifest written to: {MANIFEST_PATH}")
    print(f"Compatibility copy written to: {LEGACY_MANIFEST_PATH}")
    print(f"Total rows: {len(rows)}")

    # quick length stats, useful for planning image widths in Phase 2
    lengths = [r["char_len"] for r in rows]
    print(
        f"Modi text length (chars) - min: {min(lengths)}, max: {max(lengths)}, "
        f"avg: {sum(lengths)/len(lengths):.1f}"
    )


if __name__ == "__main__":
    main()
