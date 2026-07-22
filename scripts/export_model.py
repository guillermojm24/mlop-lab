"""Exporta el modelo champion de MLflow para construir la imagen del API."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.artifacts import download_artifacts


def main() -> None:
    destination = Path(os.getenv("SERVED_MODEL_PATH", "models/modelo_servido"))
    if destination.exists():
        raise SystemExit(f"{destination} ya existe; muévelo o renómbralo antes de exportar de nuevo.")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    download_artifacts(
        artifact_uri=os.getenv("MODEL_URI", "models:/prediccion-fuga@champion"),
        dst_path=str(destination),
    )
    print(f"Modelo exportado a {destination}")


if __name__ == "__main__":
    main()
