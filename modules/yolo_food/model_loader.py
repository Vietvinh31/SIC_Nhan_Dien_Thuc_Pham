from pathlib import Path

import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model():
    # Thu muc goc FoodVision-AI-V2
    ROOT = Path(__file__).resolve().parents[2]

    # Duong dan checkpoint YOLO Food
    MODEL_PATH = ROOT / "models" / "yolo_food" / "best.pt"

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Khong tim thay model YOLO Food: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))