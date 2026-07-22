"""Inferencia local contra el artefacto producido por el pipeline."""

from __future__ import annotations

import joblib
import pandas as pd

from .paths import MODEL_PATH


def predecir(empleado: dict, model_path=MODEL_PATH) -> dict:
    modelo = joblib.load(model_path)
    dataframe = pd.DataFrame([empleado])
    probability = float(modelo.predict_proba(dataframe)[0][1])
    return {"se_va": probability > 0.5, "probabilidad": round(probability, 3)}


def main() -> None:
    empleado = {
        "antiguedad_anios": 3,
        "salario_k": 45,
        "horas_extra_mes": 35,
        "satisfaccion": 0.2,
    }
    resultado = predecir(empleado)
    print("¿Se va?:", int(resultado["se_va"]))
    print("Probabilidad:", resultado["probabilidad"])


if __name__ == "__main__":
    main()
