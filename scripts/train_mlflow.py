"""Train a model and log parameters, metrics and artifacts to MLflow."""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from prediccion_fuga.config import load_params
from prediccion_fuga.data import FEATURES
from prediccion_fuga.paths import DATA_PATH


def train_with_mlflow() -> str:
    """Run one tracked training job and return its MLflow run ID."""
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
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "employee-attrition"))

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=params["random_state"])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores = cross_val_score(model, X_train, y_train, cv=5)

        mlflow.log_params({**params, "n_estimators": n_estimators})
        mlflow.log_metrics(
            {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "cv_mean": float(scores.mean()),
                "cv_std": float(scores.std()),
            }
        )
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_test, predictions),
            input_example=X_test.head(5),
        )
        mlflow.set_tag("candidate_model_uri", model_info.model_uri)
        return run.info.run_id


def main() -> None:
    run_id = train_with_mlflow()
    print(f"MLflow run registered: {run_id}")


if __name__ == "__main__":
    main()
