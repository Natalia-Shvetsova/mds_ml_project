import json
from pathlib import Path
import sys
import unicodedata

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned_data.xlsx"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

POSITIVE_STATUS = "Спецификация подписана / заказ закрыт"

COLUMN_SOURCE = "Источник заявки"
COLUMN_SEGMENT = "Сегмент"
COLUMN_CUSTOMER_PRICE = "Стоимость для заказчика"
COLUMN_CONTRACTOR_PRICE = "Стоимость исполнителя"
EMPTY_CATEGORY = "Не указано"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return " ".join(value.split())


def find_status_column(df: pd.DataFrame) -> str:
    """Find the target status column by name or by the positive status value."""
    for column in df.columns:
        normalized = normalize_text(column)
        if "статус" in normalized and "укрупнен" in normalized:
            return column

    for column in df.columns:
        values = df[column].dropna().astype(str)
        if values.eq(POSITIVE_STATUS).any():
            return column

    raise ValueError(
        "В Excel-файле не найден столбец со статусом. "
        "Ожидается название вроде 'Укрупнённый статус' или 'Статус укрупненно'."
    )


def choose_existing_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = [
        column for column in [COLUMN_SOURCE, COLUMN_SEGMENT] if column in df.columns
    ]
    numeric_features = [
        column
        for column in [COLUMN_CUSTOMER_PRICE, COLUMN_CONTRACTOR_PRICE]
        if column in df.columns
    ]

    if not categorical_features and not numeric_features:
        raise ValueError("Не найдены подходящие признаки для обучения модели.")

    return categorical_features, numeric_features


def save_metadata(
    source_df: pd.DataFrame,
    status_column: str,
    categorical_features: list[str],
    numeric_features: list[str],
) -> None:
    categorical_values = {}
    for column in categorical_features:
        values = (
            source_df[column]
            .fillna(EMPTY_CATEGORY)
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        categorical_values[column] = values

    numeric_defaults = {}
    for column in numeric_features:
        series = pd.to_numeric(source_df[column], errors="coerce")
        numeric_defaults[column] = {
            "min": float(series.min()) if pd.notna(series.min()) else 0.0,
            "max": float(series.max()) if pd.notna(series.max()) else 1.0,
            "median": float(series.median()) if pd.notna(series.median()) else 0.0,
        }

    metadata = {
        "target_column": status_column,
        "positive_status": POSITIVE_STATUS,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "categorical_values": categorical_values,
        "numeric_defaults": numeric_defaults,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)
    status_column = find_status_column(df)

    # Target: 1 for signed specification, 0 for all other statuses.
    df["target"] = (df[status_column] == POSITIVE_STATUS).astype(int)

    categorical_features, numeric_features = choose_existing_columns(df)
    feature_columns = categorical_features + numeric_features

    X = df[feature_columns].copy()
    y = df["target"]

    for column in categorical_features:
        X[column] = X[column].fillna(EMPTY_CATEGORY).astype(str)

    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_metadata(df, status_column, categorical_features, numeric_features)

    print("Модель обучена и сохранена.")
    print(f"Файл модели: {MODEL_PATH}")
    print(f"Файл метаданных: {METADATA_PATH}")
    print(f"Используемые признаки: {', '.join(feature_columns)}")
    print(f"Accuracy на тестовой выборке: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Ошибка обучения модели: {error}", file=sys.stderr)
        sys.exit(1)
