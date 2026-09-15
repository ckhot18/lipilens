# LipiLens Modi OCR

LipiLens is a local, non-LLM OCR prototype for Modi script line recognition.
It uses synthetic Modi line images rendered with Noto Sans Modi, a compact
CRNN model, CTC loss, and a simple Streamlit demo.

## Current Workflow

```powershell
.\venv\Scripts\python.exe scripts\build_corpus_manifest.py
.\venv\Scripts\python.exe scripts\generate_dataset.py --variants 40
.\venv\Scripts\python.exe scripts\build_vocab.py
.\venv\Scripts\python.exe scripts\train.py --dry-run --batch-size 8
.\venv\Scripts\python.exe scripts\train.py --smoke-test --batch-size 2
.\venv\Scripts\python.exe scripts\train.py --epochs 30 --batch-size 8
.\venv\Scripts\python.exe scripts\evaluate.py
.\venv\Scripts\streamlit.exe run webapp\app.py
```

Training requires CUDA-visible PyTorch. The project is tuned for the documented
RTX 2050 4GB VRAM constraint, so start with batch size 8 and reduce it if an
OOM occurs.

## Data Notes

The synthetic labels and splits expect images under `data/synthetic/images/`.
If that directory is missing, regenerate the local synthetic dataset before
training. Real MODI-HHDoc scans are research-use data and should not be
redistributed in this repo.
