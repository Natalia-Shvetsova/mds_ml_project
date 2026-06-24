from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


@lru_cache
def load_model():
    """Load the trained model once and reuse it between requests."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file was not found. Run: python utils/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_order(features: dict) -> int:
    model = load_model()
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    return int(prediction)
