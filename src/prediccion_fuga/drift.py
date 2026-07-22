"""Demostración de performance drift y concept drift."""

from __future__ import annotations

import joblib
from sklearn.metrics import accuracy_score

from .data import FEATURES, generar_dataset, regla_inflacion, regla_quema_laboral
from .paths import MODEL_PATH


def main() -> None:
    modelo = joblib.load(MODEL_PATH)
    for name, rule in (("mismo mundo", regla_quema_laboral), ("mundo nuevo", regla_inflacion)):
        dataframe = generar_dataset(500, regla=rule, seed=7)
        accuracy = accuracy_score(dataframe["se_fue"], modelo.predict(dataframe[FEATURES]))
        print(f"{name}: precisión = {accuracy:.3f}")


if __name__ == "__main__":
    main()
