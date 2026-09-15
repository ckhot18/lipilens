# LipiLens Setup And Run

This project is designed for the RTX 2050 with 4 GB VRAM. Use the repository
virtual environment and begin with batch size 8.

## One-time setup

Run these commands from the repository root in PowerShell:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install streamlit matplotlib
.\venv\Scripts\python.exe -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

If PyTorch reports that `fbgemm.dll` cannot load on Windows, install the
OpenMP runtime and restart PowerShell:

```powershell
.\venv\Scripts\python.exe -m pip install intel-openmp
```

## Verify the GPU

```powershell
.\venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0); x=torch.ones(1, device='cuda'); print('CUDA tensor:', x.device)"
nvidia-smi
```

Do not begin training until `CUDA available: True` and `CUDA tensor: cuda:0`
are both printed.

## Build and train

```powershell
.\venv\Scripts\python.exe scripts\build_corpus_manifest.py
.\venv\Scripts\python.exe scripts\generate_dataset.py --variants 40
.\venv\Scripts\python.exe scripts\build_vocab.py
.\venv\Scripts\python.exe scripts\train.py --dry-run --batch-size 8
.\venv\Scripts\python.exe scripts\train.py --epochs 5 --batch-size 8
.\venv\Scripts\python.exe scripts\train.py --epochs 30 --batch-size 8
.\venv\Scripts\python.exe scripts\evaluate.py
```

The trainer automatically retries CUDA out-of-memory failures with smaller
batches. It writes `checkpoints/best.pt`, `checkpoints/last.pt`, and
`logs/training_log.csv`.

## Run the demo

```powershell
.\venv\Scripts\streamlit.exe run webapp\app.py
```

The demo can preprocess uploads without a checkpoint. OCR output requires
`checkpoints/best.pt` created by training.
