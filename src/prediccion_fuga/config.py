"""Carga de parámetros compartidos por los stages de DVC."""

from __future__ import annotations

from pathlib import Path

import yaml

from .paths import PARAMS_PATH


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Lee params.yaml sin duplicar su estructura en cada script."""

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)
