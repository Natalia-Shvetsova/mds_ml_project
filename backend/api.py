import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.model import predict_order


BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

app = FastAPI(title="Order Success Prediction API")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/")
def root():
    return {"message": "API работает. Используйте POST /predict для прогноза."}


@app.get("/metadata")
def metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Метаданные модели не найдены. Сначала обучите модель.",
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        prediction = predict_order(request.features)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if prediction == 1:
        result_text = "Заказ с высокой вероятностью дойдет до подписания спецификации"
    else:
        result_text = "Заказ с высокой вероятностью не дойдет до подписания спецификации"

    return {
        "prediction": prediction,
        "result_text": result_text,
    }
