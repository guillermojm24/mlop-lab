"""Monitor sintético de feature drift mediante la prueba KS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .data import FEATURES
from .paths import DATA_PATH


def generar_produccion(n: int, shift: int = 0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "antiguedad_anios": rng.integers(0, 15, n),
            "salario_k": rng.integers(20 + shift, 90 + shift, n),
            "horas_extra_mes": rng.integers(0, 40, n),
            "satisfaccion": rng.uniform(0, 1, n).round(2),
        }
    )


def detectar_drift(referencia: pd.DataFrame, produccion: pd.DataFrame, umbral: float = 0.05) -> dict[str, bool]:
    resultado = {}
    for column in FEATURES:
        _, p_value = ks_2samp(referencia[column], produccion[column])
        resultado[column] = bool(p_value < umbral)
        print(f"{column:<20} p-value={p_value:.4f} drift={'sí' if resultado[column] else 'no'}")
    return resultado


def main() -> None:
    referencia = pd.read_csv(DATA_PATH)
    print("=== Producción normal ===")
    detectar_drift(referencia, generar_produccion(500, seed=7))
    print("\n=== Producción con drift (subieron los salarios) ===")
    detectar_drift(referencia, generar_produccion(500, shift=60, seed=7))


if __name__ == "__main__":
    main()
