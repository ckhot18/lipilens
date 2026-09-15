# CLAUDE.md — Modi Script OCR Digitization Project

> Context handoff document for Claude Code / Claude Desktop. This project was scoped and partially built in a prior chat session; this file is the single source of truth going forward. Read this fully before touching any code.

---

## 0. Project Brief (non-negotiable constraints)

- **Goal:** Digitize Modi script (historical script used to write Marathi before Devanagari became standard) — build an OCR pipeline + web app for a college competition.
- **Hard constraints:**
  - Hardware: 8GB system RAM, RTX 2050 laptop GPU with **4GB VRAM**
  - Time: **8–9 days total, 3–4 hours/day** (~28–34 active hours budget)
  - **No LLM APIs allowed.** Must build/fine-tune/use a local, non-LLM model.
  - Authenticity matters for judging — prefer real historical data validation over pure synthetic claims.
  - A research paper is a secondary deliverable (not primary, but expected).
- **Why this is a good competition project:** No public, clean, labeled Modi OCR dataset exists. Published academic work trains on datasets as small as ~4,140 isolated character images. This is a genuinely under-solved, low-resource-script problem — good for novelty scoring.

---

## 1. System & Architecture Overview

### Tech stack
- **Language:** Python 3
- **Image handling:** Pillow (PIL), OpenCV (`opencv-python-headless`), NumPy
- **Font:** Google **Noto Sans Modi** (`NotoSansModi-Regular.ttf`) — pulled from `github.com/google/fonts/ofl/notosansmodi/`. Only one weight/style exists (no bold/italic variant available) — visual diversity is manufactured in code (size, stroke width, shear) instead.
- **Model framework (planned):** PyTorch (`torch`, `torchvision`) — install separately with CUDA 12.1 wheel to match RTX 2050:
  ```
  pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
  ```
- **Web app (planned):** Flask or Streamlit (not yet decided/built) — kept deliberately simple, not a priority for grading.

### Core architectural decisions (and why)

| Decision | Reasoning |
|---|---|
| **Synthetic training data**, not scraped/collected real manuscripts | No public Modi dataset exists at usable scale; real manuscript collection is too slow for the timeline. Standard practice in low-resource-script OCR research. |
| Render text via **Devanagari→Modi Unicode transliteration + Noto Sans Modi font**, not hand-drawing/collecting glyphs | Modi Unicode block (U+11600–U+1165F) structurally mirrors the Devanagari block in the same phonetic order (per official Unicode chart), so a verified character-substitution table + a real font gives correct, authentic Modi text at scale. |
| **Line-level CRNN + CTC** architecture, NOT character-level classification | Modi is a cursive/connected script — segmenting individual characters (needed for char-classifiers) is unreliable. CTC loss allows training on full line images with text labels, no manual character segmentation required. This is the standard architecture for real-world OCR (Tesseract LSTM engine, most modern Indic OCR). |
| Rejected (for now): fine-tuning **TrOCR** (transformer OCR) | TrOCR has zero pretraining exposure to Modi glyphs, so transfer-learning benefit is minimal, while VRAM/compute cost is much higher than CRNN on 4GB VRAM. Documented as "explored but deprioritized" — a legitimate finding for the paper, not a cop-out. |
| Real data used **only for validation/testing**, not training | **MODI-HHDoc dataset** (Mendeley Data / IEEE DataPort, 3,350 real scanned historical Modi documents, research-use license, no redistribution) — https://data.mendeley.com/datasets/sg337vf6wn/1. Plan: hand-transcribe a small subset (~20–30 lines) as a true test set to prove the model generalizes beyond synthetic data — this is the "authenticity" story for judges. |
| Train/val split is **by sentence ID, not by image** | Prevents near-duplicate leakage (same sentence, different degradation) between train and val, which would inflate validation accuracy artificially. |
| Data pipeline validated with visual spot-checks at every stage | Caught and fixed a real bug: 2/178 corpus sentences used loanwords (कॉलेज, किनाऱ्यावर) with sounds absent from the historical Modi character set — font silently failed to render those glyphs while labels still claimed the character was there (corrupted training pairs). Fixed by rewording; now 178/178 sentences transliterate cleanly with zero unmapped characters. |

### Reference literature (for paper citations)
- Baseline comparison: character-level CNN classifiers on isolated Modi glyphs report ~91–94% accuracy on small datasets (~4,140 images).
- **MoScNet** (IIT Roorkee, 2025) — CRNN/transformer-based model trained on 2,043 real Modi manuscript images, BLEU ~51. Cite as related work showing this is active research; position this project as a lightweight/practical version achievable on modest consumer hardware, not a competitor to beat.

---

## 2. Current Code State (Status Check)

### Phase 1 — Data foundation
- [x] **Completed** — `scripts/devanagari_to_modi.py`
  Character mapping table (Devanagari→Modi Unicode), built directly from the official Unicode nameslist chart (`unicode.org/charts/nameslist/n_11600.html`), not guessed. Includes `transliterate()` function with warning system for unmapped characters. Self-tested against 5 real Marathi sentences — zero warnings.
- [x] **Completed** — `data/marathi_corpus.txt`
  178 hand-written Marathi sentences, themed toward historical/administrative Modi-document vocabulary (Maratha-era records, manuscripts, land/revenue records) rather than generic modern text. **Fixed once**: 2 sentences originally contained loanwords with unmappable characters — reworded, now 178/178 clean.
- [x] **Completed** — `scripts/build_corpus_manifest.py`
  Reads corpus → transliterates every line → writes `data/modi_manifest.csv` (id, marathi_text, modi_text, char_len) → prints coverage report. Currently: 178/178 lines clean, 0 unmapped characters, avg 31.8 Modi chars/line.
- [x] **Completed** — `fonts/NotoSansModi-Regular.ttf`
  Verified via `fontTools` to contain all 79 assigned Modi Unicode codepoints. Sourced from `google/fonts` GitHub repo (official, licensed OFL).

### Phase 2 — Synthetic dataset generation
- [x] **Completed** — `scripts/render_text.py`
  Renders a Modi Unicode string to a clean line image: random font size (36–56px), random stroke width (crude bold simulation), random padding, random shear (slant simulation). Self-tested visually.
- [x] **Completed** — `scripts/degrade_image.py`
  Manuscript-aging pipeline: Gaussian noise, Gaussian blur, ink bleed (erosion), faded/patchy ink, random stain blobs, low-frequency paper texture, small rotation, contrast/brightness jitter, JPEG compression artifacts. `random_degrade()` applies a randomized subset/strength per image. Visually verified via montage — realistic variety confirmed.
- [x] **Completed** — `scripts/generate_dataset.py`
  Orchestrates render + degrade across the full corpus. **Currently configured for 40 variants/sentence.** Outputs:
  - `data/synthetic/images/*.png` (7,120 images generated in last run, ~304MB total — regenerate locally, not handed over as a bulk file)
  - `data/synthetic/labels.csv` (image_path, marathi_text, modi_text, sentence_id)
  - `data/splits/train.csv` (6,440 images / 161 sentences)
  - `data/splits/val.csv` (680 images / 17 sentences)
  Verified visually via random 8-image sample montage — labels match rendered content, degradation variety confirmed good.
- [x] **Completed** — `scripts/build_vocab.py`
  Scans `labels.csv`, extracts unique character set, builds `data/vocab.json`:
  ```json
  { "blank_index": 0, "char_to_idx": {...}, "idx_to_char": [...], "vocab_size": 58 }
  ```
  Current vocab size: **58** (57 real Modi characters/punctuation + 1 CTC blank token). Confirmed zero contamination from non-Modi characters after the corpus fix.
- [x] **Completed** — `requirements.txt`
  Pillow, opencv-python-headless, numpy, fonttools pinned. Torch intentionally excluded (installed separately per-machine for correct CUDA build).

### Phase 3 — Preprocessing pipeline (for real scanned images)
- [ ] **Not Started**
  Planned file: `scripts/preprocess.py`. Needed steps (all discussed, none implemented yet):
  - Grayscale conversion
  - Denoising (`cv2.fastNlMeansDenoising`)
  - Binarization (`cv2.adaptiveThreshold` — chosen over global Otsu threshold because it handles uneven old-paper lighting better)
  - Deskewing (detect dominant line angle, rotate to correct)
  - Line segmentation (horizontal projection profile to split a full page into individual line-strip images)
  **This must run on real MODI-HHDoc scans as its test input** (not yet downloaded to local environment — see Section 4).

### Phase 4 — Model architecture
- [ ] **Not Started**
  Planned file: `scripts/model.py`. Architecture decided (CRNN: CNN feature extractor → BiLSTM → CTC loss), but **no code has been written yet.** Output layer size must equal `vocab_size` (58, from `data/vocab.json`) — do not hardcode this, load from vocab file.

### Phase 5 — Training
- [ ] **Not Started**
  Planned file: `scripts/train.py`. Needs: PyTorch `Dataset`/`DataLoader` reading `data/splits/train.csv` + `val.csv`, image transforms (resize to fixed height, normalize), CTC loss training loop, checkpointing to `checkpoints/`.
  **GPU verification not yet confirmed on local machine** — user must run and confirm before training begins:
  ```
  python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

### Phase 6 — Evaluation
- [ ] **Not Started**
  Plan: evaluate on synthetic val set (`data/splits/val.csv`) for tracking, and separately hand-transcribe ~20–30 real lines from MODI-HHDoc as a genuine test set. **MODI-HHDoc has not yet been downloaded/incorporated into the local repo.**

### Phase 7 — Web app
- [ ] **Not Started**
  Planned structure: `webapp/app.py` + `templates/` + `static/`. Flow: upload → preprocess (Phase 3) → CRNN inference (Phase 4/5 model) → display cleaned image + transcription side by side → gallery view of all processed documents. Framework not yet chosen (Flask vs Streamlit) — low priority, keep simple.

### Phase 8 — Integration & polish
- [ ] **Not Started** (depends on all prior phases)

### Phase 9 — Paper
- [ ] **Not Started**
  Planned file: `paper/draft.md`. Angle already decided (see Section 1 "Reference literature" + architectural decisions table) — frame as: data-scarcity is the central challenge, synthetic-data-driven CRNN is the practical answer, report synthetic-val + real-test results, discuss why CTC/CRNN beats character classification for cursive Modi, note TrOCR exploration and why it was deprioritized.

---

## 3. Known Constraints & Requirements

### Data model / file contracts (must not break these — later scripts depend on exact schemas)

**`data/modi_manifest.csv`**
```
id, marathi_text, modi_text, char_len
```

**`data/synthetic/labels.csv`** and **`data/splits/{train,val}.csv`**
```
image_path, marathi_text, modi_text, sentence_id
```
- `image_path` is relative (e.g. `data/synthetic/images/line_0000_v000.png`) — scripts assume they're run from inside `scripts/` with `ROOT = Path(__file__).parent.parent`.

**`data/vocab.json`**
```json
{
  "blank_index": 0,
  "char_to_idx": { "<blank>": 0, "...": 1, ... },
  "idx_to_char": ["<blank>", "...", ...],
  "vocab_size": 58
}
```
- **CRITICAL RULE:** If the corpus changes (more sentences added, wording changed), you MUST re-run in this exact order: `build_corpus_manifest.py` → `generate_dataset.py` → `build_vocab.py`. If `vocab_size` changes, any previously trained model checkpoint becomes invalid (output layer size is tied to vocab size) — must retrain from scratch.

### Folder structure (must be exact — relative paths in scripts depend on it)

```
modi-ocr/
├── data/
│   ├── raw_corpus/           (conceptual grouping — currently corpus lives at data/marathi_corpus.txt directly)
│   ├── marathi_corpus.txt
│   ├── modi_manifest.csv
│   ├── vocab.json
│   ├── synthetic/
│   │   ├── images/
│   │   └── labels.csv
│   ├── splits/
│   │   ├── train.csv
│   │   └── val.csv
│   └── real/                 (NOT YET CREATED — for MODI-HHDoc real scans + hand-annotated test set)
│       ├── raw_scans/
│       └── annotated_test/
├── fonts/
│   └── NotoSansModi-Regular.ttf
├── scripts/
│   ├── devanagari_to_modi.py
│   ├── build_corpus_manifest.py
│   ├── render_text.py
│   ├── degrade_image.py
│   ├── generate_dataset.py
│   ├── build_vocab.py
│   ├── preprocess.py         (NOT YET CREATED)
│   ├── model.py               (NOT YET CREATED)
│   ├── train.py                (NOT YET CREATED)
│   └── infer.py                 (NOT YET CREATED)
├── checkpoints/                   (NOT YET CREATED — model weights go here)
├── webapp/                         (NOT YET CREATED)
├── paper/                           (NOT YET CREATED)
└── requirements.txt
```

### Environment / hardware constraints affecting implementation choices
- **4GB VRAM ceiling** — training batch sizes, image resolution, and model size must stay modest. CRNN on small line-height images (32–64px height) at batch size 16–32 is the target; do not scale up model complexity without checking VRAM headroom.
- **8GB system RAM** — avoid loading the entire 7,120-image dataset into memory at once; use lazy-loading `Dataset` with on-the-fly image reads.
- Dataset generation (Phase 2) is **pure CPU work** (PIL/OpenCV), runs in ~2.5 minutes for 7,120 images — no GPU needed for this step, only for Phase 5 training.

### Known edge cases already identified
- 2 characters (ऱ eyelash-Ra, ॉ candra-O) have **no Modi Unicode equivalent** — legitimate historical limitation, not a bug. Any real manuscript OCR output or corpus expansion that encounters these should route around them (reword) rather than expect the model to handle them.
- Modi's consonant **ळ (Lla)** is out of alphabetical order in the Unicode block (comes after ह/Ha, not after ल/La) — this is correctly handled in `devanagari_to_modi.py`'s explicit lookup table, but **do not "simplify" this mapping to a formulaic codepoint offset** — it will break on this character.
- Real MODI-HHDoc dataset license is **research-use only, no redistribution** — fine for a college project with citation, but do not bundle/upload the raw dataset files anywhere public (e.g. don't commit to a public GitHub repo, don't include in the final submitted zip if redistribution isn't permitted — verify license terms before submission).

---

## 4. Immediate Next Steps for Claude Code

Run these in strict order. Do not skip ahead — each phase's output is a hard dependency for the next.

1. **[ ] Environment verification**
   - `pip install -r requirements.txt`
   - Install PyTorch with correct CUDA build: `pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121`
   - Verify GPU visibility: `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — **must print `True`** before proceeding to training later. If `False`, stop and fix driver/CUDA setup first.

2. **[ ] Reproduce the dataset locally** (from `scripts/`, in order):
   ```
   python3 build_corpus_manifest.py
   python3 generate_dataset.py --variants 40
   python3 build_vocab.py
   ```
   Confirm: 178 sentences, 0 unmapped characters, 7,120 images generated, vocab_size = 58. If any number differs from this, something in the local environment/font/corpus differs from what's documented here — investigate before proceeding.

3. **[ ] Build `scripts/preprocess.py`** (Phase 3 — currently not started)
   Implement: grayscale → denoise → adaptive-threshold binarize → deskew → horizontal-projection line segmentation. Test against real MODI-HHDoc sample images (download from https://data.mendeley.com/datasets/sg337vf6wn/1 first, place under `data/real/raw_scans/`).

4. **[ ] Build `scripts/model.py`** (Phase 4 — currently not started)
   CRNN architecture: CNN backbone → BiLSTM → linear layer sized to `vocab_size` (load dynamically from `data/vocab.json`, do not hardcode 58).

5. **[ ] Build `scripts/train.py`** (Phase 5 — currently not started)
   PyTorch `Dataset` reading `data/splits/train.csv`/`val.csv`, CTC loss, checkpointing to `checkpoints/`. Keep batch size conservative given 4GB VRAM; add gradient accumulation if needed rather than increasing batch size blindly.

6. **[ ] Evaluate** (Phase 6 — currently not started)
   Run on synthetic val set first for a sanity signal, then on a small hand-transcribed real MODI-HHDoc subset (~20–30 lines) for the genuine generalization test — this is the key "authenticity" result for the paper/judges.

7. **[ ] Build `webapp/`** (Phase 7 — currently not started)
   Only after a trained model exists. Keep minimal: upload → preprocess → infer → display → gallery.

8. **[ ] Integration pass** (Phase 8) — wire model into web app, test end-to-end on both synthetic and real images, fix failure cases.

9. **[ ] Write `paper/draft.md`** (Phase 9) — can start earlier in parallel once results from step 6 exist; methodology section can be drafted even earlier since the approach is already fully decided (see Section 1).

**Time-boxing reminder for whoever is running this:** original plan budgeted ~28–34 active hours total across all 9 phases, with steps 1–2 above already representing "Phase 1 and 2, done" from the original schedule (~day 1 of ~8-9 day timeline). Do not over-invest in Phase 3/7 polish at the expense of Phase 4/5 (the model) — a working-but-imperfect end-to-end pipeline by the midpoint of the timeline was the stated priority over a polished but incomplete one at the deadline.
