"""Rutas del proyecto centralizadas en un único lugar."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "empleados.csv"
MODEL_PATH = ROOT / "models" / "modelo.pkl"
METRICS_PATH = ROOT / "metrics" / "metrics.json"
SERVED_MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "modelo_servido"))
PARAMS_PATH = ROOT / "params.yaml"


def ensure_parent(path: Path) -> Path:
    """Crea la carpeta padre de un artefacto y devuelve su ruta."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path
