"""Compara feature drift observable con concept drift real."""

from __future__ import annotations

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

from .data import FEATURES, generar_dataset, regla_inflacion, regla_quema_laboral
from .paths import DATA_PATH, MODEL_PATH


def analizar(nombre: str, referencia: pd.DataFrame, produccion: pd.DataFrame, modelo) -> None:
    print(f"\n=== {nombre} ===")
    drift = [column for column in FEATURES if ks_2samp(referencia[column], produccion[column]).pvalue < 0.05]
    print("  Drift en entradas:", ", ".join(drift) if drift else "ninguno")
    accuracy = accuracy_score(produccion["se_fue"], modelo.predict(produccion[FEATURES]))
    print(f"  Acierto real del modelo: {accuracy:.3f}")


def main() -> None:
    referencia = pd.read_csv(DATA_PATH)
    modelo = joblib.load(MODEL_PATH)
    analizar("MISMO MUNDO", referencia, generar_dataset(500, regla_quema_laboral, 7), modelo)
    analizar("CONCEPT DRIFT", referencia, generar_dataset(500, regla_inflacion, 7), modelo)


if __name__ == "__main__":
    main()
