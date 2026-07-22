"""API de inferencia local, preparada para ejecutarse en contenedor."""

from __future__ import annotations

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .paths import SERVED_MODEL_PATH

modelo = mlflow.sklearn.load_model(str(SERVED_MODEL_PATH))
app = FastAPI(title="Predicción de fuga", version="1.0.0")


class Empleado(BaseModel):
    antiguedad_anios: int = Field(ge=0)
    salario_k: int = Field(gt=0)
    horas_extra_mes: int = Field(ge=0)
    satisfaccion: float = Field(ge=0, le=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predecir")
def predecir(empleado: Empleado) -> dict[str, float | bool]:
    dataframe = pd.DataFrame([empleado.model_dump()])
    probability = float(modelo.predict_proba(dataframe)[0][1])
    return {"se_va": probability > 0.5, "probabilidad": round(probability, 3)}
