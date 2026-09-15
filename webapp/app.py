from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess import preprocess_pil

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"


RESULTS_DIR = ROOT / "results" / "webapp"
STORE_PATH = RESULTS_DIR / "results.json"


def load_store() -> list[dict]:
    if STORE_PATH.exists():
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return []


def save_store(records: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_result(
    original: Image.Image, cleaned: Image.Image, transcription: str
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = RESULTS_DIR / f"{stamp}_original.png"
    cleaned_path = RESULTS_DIR / f"{stamp}_cleaned.png"
    original.save(original_path)
    cleaned.save(cleaned_path)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "original": str(original_path),
        "cleaned": str(cleaned_path),
        "transcription": transcription,
    }
    records = load_store()
    records.insert(0, record)
    save_store(records)
    return record


st.set_page_config(page_title="LipiLens Modi OCR", layout="wide")
st.title("LipiLens Modi OCR")

upload_tab, gallery_tab, metrics_tab = st.tabs(
    ["Upload & Transcribe", "Gallery", "Model Metrics"]
)

with upload_tab:
    uploaded = st.file_uploader(
        "Upload a Modi manuscript image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if uploaded:
        original = Image.open(uploaded).convert("RGB")
        cleaned, lines = preprocess_pil(original)
        left, right = st.columns(2)
        left.image(original, caption="Original", use_container_width=True)
        right.image(
            cleaned, caption=f"Cleaned / {len(lines)} line(s)", use_container_width=True
        )
        if DEFAULT_CHECKPOINT.exists():
            with st.spinner("Running OCR..."):
                from infer import transcribe_images

                predictions = transcribe_images(lines, DEFAULT_CHECKPOINT)
            transcription = "\n".join(predictions)
            st.text_area("Transcription", transcription, height=160)
            if st.button("Save result", type="primary"):
                save_result(original, cleaned, transcription)
                st.success("Saved to gallery.")
        else:
            st.warning(
                "No trained checkpoint found at checkpoints/best.pt. "
                "Preprocessing works; train the model before live OCR output is available."
            )

with gallery_tab:
    records = load_store()
    if not records:
        st.info("No processed documents saved yet.")
    for record in records:
        st.caption(record["timestamp"])
        left, mid = st.columns([1, 1])
        if Path(record["original"]).exists():
            left.image(record["original"], caption="Original", use_container_width=True)
        if Path(record["cleaned"]).exists():
            mid.image(record["cleaned"], caption="Cleaned", use_container_width=True)
        st.text(record["transcription"])
        st.divider()

with metrics_tab:
    metrics_path = ROOT / "reports" / "metrics.md"
    if metrics_path.exists():
        st.markdown(metrics_path.read_text(encoding="utf-8"))
    else:
        st.info("Run scripts/evaluate.py after training to generate metrics.")
    for name in ["loss_curves.png", "cer_histogram.png", "qualitative_grid.png"]:
        path = ROOT / "reports" / name
        if path.exists():
            st.image(str(path), caption=name, use_container_width=True)
