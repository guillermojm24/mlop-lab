"""Fail the workflow when the latest MLflow run does not meet the quality gate."""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from prediccion_fuga.config import load_params


def latest_run():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "employee-attrition")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow runs found in experiment: {experiment_name}")
    return runs[0]


def main() -> None:
    run = latest_run()
    threshold = float(load_params()["evaluacion"]["min_accuracy"])
    accuracy = run.data.metrics.get("accuracy")
    if accuracy is None:
        raise RuntimeError(f"Run {run.info.run_id} has no accuracy metric")

    print(f"run_id={run.info.run_id} accuracy={accuracy:.4f} threshold={threshold:.4f}")
    if accuracy < threshold:
        raise SystemExit(
            f"Model quality gate failed: accuracy {accuracy:.4f} < {threshold:.4f}"
        )


if __name__ == "__main__":
    main()
