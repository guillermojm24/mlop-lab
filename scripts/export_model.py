"""Export the MLflow champion model before building the inference image."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.artifacts import download_artifacts


def main() -> None:
    destination = Path(os.getenv("SERVED_MODEL_PATH", "models/modelo_servido"))
    if destination.exists():
        raise SystemExit(
            f"{destination} already exists; remove or rename it before exporting again."
        )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model_uri = os.getenv("MODEL_URI", "models:/employee-attrition@champion")
    download_artifacts(artifact_uri=model_uri, dst_path=str(destination))
    print(f"Exported {model_uri} to {destination}")


if __name__ == "__main__":
    main()
