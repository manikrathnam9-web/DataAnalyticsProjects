import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    DEFAULT_SAMPLE_FRACTION,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    PREPARED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.ensemble_model import build_ensemble_model


def load_prepared_data(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    except ValueError:
        return pd.read_csv(csv_path)


def chronological_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    sorted_df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_size))
    train_idx = list(range(split_idx))
    test_idx = list(range(split_idx, len(sorted_df)))
    return train_idx, test_idx


def get_feature_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def train_ensemble(
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    data_path: Path = PREPARED_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> dict:
    df = load_prepared_data(data_path)

    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=RANDOM_STATE)

    X, y = get_feature_matrix(df)
    train_idx, test_idx = chronological_split(df)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = build_ensemble_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, None)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "sample_fraction": sample_fraction,
        "models": ["random_forest", "gradient_boosting", "ridge"],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "metrics": metrics,
        },
        model_path,
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python main.py --train` first."
        )
    return joblib.load(model_path)


def predict_demand(features: dict, model_path: Path = MODEL_PATH) -> float:
    bundle = load_model(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    row = {column: float(features[column]) for column in feature_columns}
    frame = pd.DataFrame([row])
    prediction = float(model.predict(frame)[0])
    return max(0.0, prediction)


def main() -> None:
    metrics = train_ensemble()
    print("Ensemble model trained and saved.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
