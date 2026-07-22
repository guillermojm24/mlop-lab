"""Entrenamiento local y escritura del modelo/metricas reproducibles."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from .config import load_params
from .data import FEATURES
from .paths import DATA_PATH, METRICS_PATH, MODEL_PATH, ensure_parent


def entrenar(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    params = load_params()["entrenar"]
    dataframe = pd.read_csv(data_path)
    X = dataframe[FEATURES]
    y = dataframe["se_fue"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    modelo = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
    )
    modelo.fit(X_train, y_train)
    scores = cross_val_score(modelo, X_train, y_train, cv=5)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, modelo.predict(X_test))), 4),
        "cv_mean": round(float(scores.mean()), 4),
        "cv_std": round(float(scores.std()), 4),
    }

    ensure_parent(model_path)
    ensure_parent(metrics_path)
    joblib.dump(modelo, model_path)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    return metrics


def main() -> None:
    metrics = entrenar()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
