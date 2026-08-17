"""Register the latest approved MLflow run and promote it to the champion alias."""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from evaluate_latest_run import latest_run

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "employee-attrition")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    run = latest_run()
    model_uri = f"runs:/{run.info.run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered.version,
    )
    print(
        f"Promoted {MODEL_NAME} version {registered.version} "
        f"from run {run.info.run_id} to @champion"
    )


if __name__ == "__main__":
    main()
