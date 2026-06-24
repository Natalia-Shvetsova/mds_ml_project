import json
from pathlib import Path

import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(page_title="Прогноз заказа")
st.title("Прогноз успешного завершения заказа")

if not METADATA_PATH.exists():
    st.error(
        "Файл models/model_metadata.json не найден. "
        "Сначала обучите модель."
    )
    st.code("python utils/train_model.py")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

st.write(
    "Введите параметры заказа и нажмите "
    "кнопку прогноза."
)

features = {}

with st.form("prediction_form"):
    for column in metadata["categorical_features"]:
        options = metadata["categorical_values"].get(column, [])
        features[column] = st.selectbox(column, options)

    for column in metadata["numeric_features"]:
        defaults = metadata["numeric_defaults"]
        min_value = float(defaults[column]["min"])
        max_value = float(defaults[column]["max"])
        median_value = float(defaults[column]["median"])
        features[column] = st.number_input(
            column,
            min_value=min_value,
            max_value=max_value,
            value=median_value,
        )

    submitted = st.form_submit_button("Сделать прогноз")

if submitted:
    try:
        response = requests.post(API_URL, json={"features": features}, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        st.error(
            "Не удалось получить ответ от FastAPI. "
            "Проверьте, что backend запущен."
        )
        st.exception(error)
    else:
        if result["prediction"] == 1:
            st.success(result["result_text"])
        else:
            st.warning(result["result_text"])
