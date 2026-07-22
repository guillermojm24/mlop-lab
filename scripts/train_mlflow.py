"""Lanza un entrenamiento y registra parámetros, métricas y modelo en MLflow."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from prediccion_fuga.config import load_params
from prediccion_fuga.data import FEATURES
from prediccion_fuga.paths import DATA_PATH


def main() -> None:
    params = load_params()["entrenar"]
    n_estimators = int(os.getenv("N_ESTIMATORS", params["n_estimators"]))
    dataframe = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe[FEATURES],
        dataframe["se_fue"],
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=dataframe["se_fue"],
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "prediccion-fuga"))
    with mlflow.start_run():
        mlflow.log_params({**params, "n_estimators": n_estimators})
        modelo = RandomForestClassifier(n_estimators=n_estimators, random_state=params["random_state"])
        modelo.fit(X_train, y_train)
        scores = cross_val_score(modelo, X_train, y_train, cv=5)
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, modelo.predict(X_test)),
                "cv_mean": scores.mean(),
                "cv_std": scores.std(),
            }
        )
        mlflow.sklearn.log_model(modelo, name="modelo")
        print(f"Run registrado: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
