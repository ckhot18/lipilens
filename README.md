# LipiLens

LipiLens is a computer vision project for recognition of Modi/Lipi characters.

## Requirements

- Windows 10/11
- Python 3.11
- NVIDIA GPU recommended
- Git

## Links for datasets :

- https://www.kaggle.com/datasets/msanshikajain/modi-lipi-matra-dastaset

## 1. Clone the repository

git clone https://github.com/ckhot18/lipilens/

cd LipiLens

## 2. Create virtual environment

py -m venv .venv

## 3. Activate environment

.venv\Scripts\activate

## 4. Upgrade pip

python -m pip install --upgrade pip

## 5. Install dependencies

pip install -r requirements.txt

## 6. Verify PyTorch and CUDA

python

Then run:

import torch

print("PyTorch:", torch.**version**)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
print("GPU:", torch.cuda.get_device_name(0))

## 7. Start Jupyter

jupyter notebook
