# LipiLens Verified Status

This status should be refreshed after each environment or dataset change.

## Verified In This Build

- Devanagari to Modi mapping keeps the special Lla ordering: `ळ` maps to
  `U+1162F`, after Ha.
- `data/manifests/modi_manifest.csv` has columns:
  `id, marathi_text, modi_text, char_len`.
- `data/synthetic/labels.csv`, `data/splits/train.csv`, and
  `data/splits/val.csv` have columns:
  `image_path, marathi_text, modi_text, sentence_id`.
- `data/vocab.json` has `blank_index: 0` and `vocab_size: 58`.
- `data/synthetic/images/` contains 7,120 generated line images.
- A CUDA smoke test completed one training batch and one validation batch on
  the RTX 2050, with 183.4 MB peak allocated VRAM at batch size 2. The smoke
  test intentionally writes no checkpoint.

## Implemented Product Pieces

- `scripts/preprocess.py`: grayscale, denoise, adaptive threshold, deskew,
  and horizontal-projection line segmentation.
- `scripts/model.py`: compact CRNN with CNN feature extractor, 2-layer BiLSTM,
  and CTC-ready logits.
- `scripts/train.py`: lazy image dataset, CTC collate function, AMP training,
  OOM batch-size retry, best/last checkpoints, and CSV logging.
- `scripts/infer.py`: checkpoint loading and greedy CTC decoding.
- `scripts/evaluate.py`: synthetic validation CER/WER plus report plots.
- `webapp/app.py`: Streamlit upload, preprocessing, checkpoint-backed OCR,
  persistent gallery, and metrics display.

## Remaining Runtime Work

- Full training has not been run, so `checkpoints/best.pt` does not exist yet.
  It is required for live OCR output and evaluation reports.
- The current environment has NumPy 2.5.2 while the installed PyTorch build
  expects NumPy 1.x. The tensor input path is NumPy-free and the GPU smoke test
  passes, but install the pinned `numpy==1.26.4` before a full training run.
- Streamlit, Matplotlib, OpenCV, and FontTools still need the one-time install
  described in `SETUP_AND_RUN.md`; the Python sources compile successfully.
- Real manuscript evaluation needs MODI-HHDoc scans and hand-transcribed test
  labels under `data/real/`; do not redistribute that dataset.
