"""Generación del dataset sintético del caso de uso."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_params
from .paths import DATA_PATH, ensure_parent

FEATURES = ["antiguedad_anios", "salario_k", "horas_extra_mes", "satisfaccion"]


def regla_quema_laboral(dataframe: pd.DataFrame) -> pd.Series:
    """Regla sintética con la que se genera la primera versión del dataset."""

    return (dataframe.horas_extra_mes / 40) * 0.5 + (1 - dataframe.satisfaccion) * 0.5


def regla_inflacion(dataframe: pd.DataFrame) -> pd.Series:
    """Regla alternativa para demostrar concept drift."""

    return (1 - (dataframe.salario_k - 20) / 70) * 0.8 + (1 - dataframe.satisfaccion) * 0.2


def generar_dataset(
    n: int,
    regla: Callable[[pd.DataFrame], pd.Series] = regla_quema_laboral,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera entradas sintéticas y su etiqueta de fuga."""

    rng = np.random.default_rng(seed)
    dataframe = pd.DataFrame(
        {
            "antiguedad_anios": rng.integers(0, 15, n),
            "salario_k": rng.integers(20, 90, n),
            "horas_extra_mes": rng.integers(0, 40, n),
            "satisfaccion": rng.uniform(0, 1, n).round(2),
        }
    )
    riesgo = regla(dataframe)
    dataframe["se_fue"] = (riesgo + rng.normal(0, 0.1, n) > 0.55).astype(int)
    return dataframe


def main(output_path: Path = DATA_PATH) -> None:
    params = load_params()["datos"]
    dataframe = generar_dataset(params["n"], seed=params["seed"])
    ensure_parent(output_path)
    dataframe.to_csv(output_path, index=False)
    print(dataframe.head())
    print(f"\nSe fueron: {dataframe.se_fue.sum()} de {len(dataframe)}")


if __name__ == "__main__":
    main()
