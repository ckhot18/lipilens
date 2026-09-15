# LipiLens — Modi Script OCR

> A lightweight, fully local OCR pipeline for Modi script — a historical cursive writing system used to write Marathi before Devanagari became the standard. Built with a compact CRNN model, synthetic training data, and a Streamlit demo app.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Workflow](#workflow)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web App](#web-app)
- [Results](#results)
- [Data Notes](#data-notes)
- [Known Limitations](#known-limitations)

---

## Overview

Modi is a cursive, connected script with no large-scale public labeled dataset. LipiLens addresses this by:

- Generating **7,120 synthetic training images** from a 178-sentence Marathi corpus, rendered with the official Noto Sans Modi font and augmented with manuscript-style degradation
- Training a **CRNN + CTC** model — the standard architecture for cursive script OCR, requiring no character-level segmentation
- Validating against both synthetic held-out data and a small set of real historical scans (MODI-HHDoc)
- Running entirely on **consumer hardware** (RTX 2050, 4 GB VRAM)

| Metric | Value |
|---|---|
| Synthetic val CER | 2.24% |
| Synthetic val WER | 10.74% |
| Training images | 7,120 |
| Vocab size | 58 tokens |
| Peak VRAM (batch 2) | ~183 MB |

---

## Architecture

```
Input image (grayscale, H=32)
        |
        v
+---------------------+
|   CNN Feature       |   5 conv blocks with BatchNorm + ReLU
|   Extractor         |   Channels: 1 -> 64 -> 128 -> 256 -> 512
+---------------------+
        |
        v
+---------------------+
|   2-layer BiLSTM    |   hidden_size=256, bidirectional -> 512-dim
+---------------------+
        |
        v
+---------------------+
|   Linear + CTC Loss |   output_size = vocab_size (58)
+---------------------+
        |
        v
  Greedy CTC Decode -> Transcription
```

**Why CRNN + CTC?**
Modi is a cursive/connected script — isolating individual characters for a classifier is unreliable. CTC loss trains directly on full line images paired with text labels, no character segmentation needed. This is the same approach used by Tesseract's LSTM engine and most production Indic OCR systems.

**Why not TrOCR?**
TrOCR has zero pretraining exposure to Modi glyphs, so transfer-learning benefit is minimal while VRAM cost is significantly higher. Explored and deprioritized — a documented finding, not a gap.

---

## Project Structure

```
lipilens/
├── configs/
│   └── config.yaml              # Training hyperparameters
├── checkpoints/
│   ├── best.pt                  # Best validation checkpoint
│   └── last.pt                  # Most recent checkpoint
├── data/
│   ├── raw_corpus/
│   │   └── marathi_corpus.txt   # 178 Marathi sentences
│   ├── manifests/
│   │   └── modi_manifest.csv    # id, marathi_text, modi_text, char_len
│   ├── synthetic/
│   │   ├── images/              # 7,120 generated line images (not committed)
│   │   └── labels.csv           # image_path, marathi_text, modi_text, sentence_id
│   ├── splits/
│   │   ├── train.csv            # 6,440 images / 161 sentences
│   │   ├── val.csv              # 680 images / 17 sentences
│   │   └── test.csv
│   ├── real/
│   │   ├── raw_scans/           # MODI-HHDoc scans (not redistributed)
│   │   └── annotated_test/      # Hand-transcribed test labels
│   └── vocab.json               # 58-token character vocabulary
├── fonts/
│   └── NotoSansModi-Regular.ttf # OFL-licensed, all 79 Modi codepoints
├── logs/
│   └── training_log.csv         # epoch, train_loss, val_loss, timestamp
├── reports/
│   ├── metrics.md               # CER / WER summary
│   ├── loss_curves.png
│   ├── cer_histogram.png
│   └── qualitative_grid.png
├── scripts/
│   ├── devanagari_to_modi.py    # Devanagari -> Modi Unicode transliteration
│   ├── build_corpus_manifest.py # Corpus -> manifest CSV
│   ├── render_text.py           # Modi string -> line image (Pillow)
│   ├── degrade_image.py         # Manuscript-aging augmentation pipeline
│   ├── generate_dataset.py      # Orchestrates render + degrade -> splits
│   ├── build_vocab.py           # Extracts character vocab from labels
│   ├── preprocess.py            # Grayscale -> denoise -> binarize -> deskew -> segment
│   ├── model.py                 # CRNN definition
│   ├── train.py                 # Training loop with AMP + OOM retry
│   ├── infer.py                 # Checkpoint loading + greedy CTC decode
│   └── evaluate.py              # CER/WER + report plots
├── webapp/
│   └── app.py                   # Streamlit demo
├── paper/
│   └── draft.md                 # Research paper draft
├── requirements.txt
└── SETUP_AND_RUN.md
```

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| Pillow | 10.4.0 | Image rendering and augmentation |
| opencv-python-headless | 4.10.0.84 | Preprocessing pipeline |
| numpy | 1.26.4 | Must be 1.x for PyTorch compatibility |
| fonttools | 4.53.1 | Font verification |
| torch | 2.4.0 | Install separately with CUDA wheel |
| torchvision | 0.19.0 | Install separately with CUDA wheel |
| streamlit | latest | Web demo |
| matplotlib | latest | Evaluation plots |

PyTorch is intentionally excluded from `requirements.txt` — install it with the correct CUDA build for your GPU (see Setup below).

---

## Setup

Run all commands from the repository root in PowerShell.

### 1. Install dependencies

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install streamlit matplotlib
.\venv\Scripts\python.exe -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

If `fbgemm.dll` fails to load on Windows, also run:

```powershell
.\venv\Scripts\python.exe -m pip install intel-openmp
```

### 2. Verify GPU

```powershell
.\venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

Do not begin training until `CUDA: True` is printed.

---

## Workflow

### Regenerate the dataset

Synthetic images are not committed to the repository (~304 MB). Run this sequence to rebuild from the corpus:

```powershell
.\venv\Scripts\python.exe scripts\build_corpus_manifest.py
.\venv\Scripts\python.exe scripts\generate_dataset.py --variants 40
.\venv\Scripts\python.exe scripts\build_vocab.py
```

Expected: 178 sentences, 0 unmapped characters, 7,120 images, `vocab_size = 58`.

> **Important:** If the corpus changes, re-run all three steps in this exact order. Any change to `vocab_size` invalidates existing checkpoints — you must retrain from scratch.

---

## Training

**Dry run** — shape check only, no GPU required:

```powershell
.\venv\Scripts\python.exe scripts\train.py --dry-run --batch-size 8
```

**Smoke test** — one train + one val batch, no checkpoint written:

```powershell
.\venv\Scripts\python.exe scripts\train.py --smoke-test --batch-size 2
```

**Full training:**

```powershell
.\venv\Scripts\python.exe scripts\train.py --epochs 30 --batch-size 8
```

The trainer uses mixed-precision (AMP) and automatically retries with half the batch size on CUDA OOM. Checkpoints are saved to `checkpoints/best.pt` and `checkpoints/last.pt`. Progress is logged to `logs/training_log.csv`.

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 8 |
| Learning rate | 0.001 |
| Early stopping patience | 5 epochs |
| Gradient clip norm | 5.0 |
| Mixed precision | enabled |

---

## Evaluation

```powershell
.\venv\Scripts\python.exe scripts\evaluate.py
```

Outputs saved to `reports/`:

| File | Description |
|---|---|
| `metrics.md` | CER and WER summary |
| `loss_curves.png` | Train/val loss over epochs |
| `cer_histogram.png` | CER distribution across validation samples |
| `qualitative_grid.png` | Ground truth vs prediction side by side |
| `synthetic_val_predictions.csv` | Per-sample predictions with CER/WER |

For real manuscript evaluation, add MODI-HHDoc scans under `data/real/raw_scans/` and hand-transcribed CSV labels under `data/real/annotated_test/`. See [Data Notes](#data-notes).

---

## Web App

```powershell
.\venv\Scripts\streamlit.exe run webapp\app.py
```

The app has three tabs:

| Tab | Description |
|---|---|
| Upload & Transcribe | Upload a Modi manuscript image, view the preprocessed output, get a transcription |
| Gallery | Persistent history of all processed documents with original and cleaned images |
| Model Metrics | Live view of `reports/metrics.md` and evaluation plots |

Preprocessing (grayscale, denoise, binarize, deskew, line segmentation) works without a checkpoint. OCR output requires `checkpoints/best.pt`.

---

## Results

Trained on 6,440 synthetic images, evaluated on 680 held-out synthetic validation samples:

| Metric | Score |
|---|---|
| Character Error Rate (CER) | **2.24%** |
| Word Error Rate (WER) | **10.74%** |

Published baselines on Modi script use character-level CNN classifiers on isolated glyphs (~4,140 images), reporting 91–94% character accuracy. LipiLens trains on full line images with no character segmentation — a more realistic and general formulation of the OCR problem that also matches how modern production OCR systems work.

For reference, MoScNet (IIT Roorkee, 2025) — a CRNN/transformer model trained on 2,043 real Modi manuscript images — reports BLEU ~51. LipiLens is positioned as a lightweight, reproducible baseline achievable on modest consumer hardware, not a direct competitor.

---

## Data Notes

**Synthetic data** — generated locally from `data/raw_corpus/marathi_corpus.txt` using `NotoSansModi-Regular.ttf`. Images are not committed to this repository; regenerate with `generate_dataset.py`. ~2.5 minutes on CPU for 7,120 images.

**Noto Sans Modi font** — sourced from `github.com/google/fonts`, licensed under the SIL Open Font License (OFL). Verified to contain all 79 assigned Modi Unicode codepoints (U+11600–U+1165F). Only one weight exists (no bold/italic variant) — visual diversity is manufactured via font-size variation, stroke-width simulation, random shear, and degradation.

**MODI-HHDoc dataset** — 3,350 real scanned historical Modi documents. Licensed for research use only; do not redistribute.
Download: https://data.mendeley.com/datasets/sg337vf6wn/1
Place under `data/real/raw_scans/` for real-data evaluation.

---

## Known Limitations

- **Transliteration table is explicit, not formulaic.** Modi's consonant ळ (Lla) falls out of alphabetical order in the Unicode block (after ह/Ha, not ल/La). The `devanagari_to_modi.py` lookup table handles this correctly — do not simplify it to a codepoint-offset formula.
- **Two characters have no Modi Unicode equivalent.** Eyelash-Ra (ऱ) and candra-O (ॉ) are a known historical limitation of the script. Corpus sentences using these are reworded; the model does not handle them.
- **Synthetic training only.** Generalization to real manuscripts is evaluated on a small hand-transcribed subset. Broader real-world accuracy has not been systematically benchmarked.
- **Line-level model.** Full-page OCR requires `scripts/preprocess.py` to segment lines first.
- **NumPy version pin.** Keep `numpy==1.26.4` — PyTorch 2.4.0 expects NumPy 1.x; NumPy 2.x causes compatibility issues.
- **4 GB VRAM ceiling.** Do not increase model complexity or batch size without checking VRAM headroom first. Start with `--batch-size 8` and let the OOM retry logic handle downsizing if needed.
