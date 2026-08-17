"""FastAPI inference service with health checks and Prometheus metrics."""

from __future__ import annotations

import time

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .paths import SERVED_MODEL_PATH

model = mlflow.sklearn.load_model(str(SERVED_MODEL_PATH))
app = FastAPI(title="Employee Attrition Inference API", version="1.1.0")

PREDICTIONS = Counter(
    "mlop_lab_predictions_total",
    "Total number of inference requests.",
    ["prediction"],
)
PREDICTION_LATENCY = Histogram(
    "mlop_lab_prediction_latency_seconds",
    "Inference latency in seconds.",
)


class Employee(BaseModel):
    antiguedad_anios: int = Field(ge=0)
    salario_k: int = Field(gt=0)
    horas_extra_mes: int = Field(ge=0)
    satisfaccion: float = Field(ge=0, le=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(employee: Employee) -> dict[str, float | bool]:
    started_at = time.perf_counter()
    dataframe = pd.DataFrame([employee.model_dump()])
    probability = float(model.predict_proba(dataframe)[0][1])
    will_leave = probability > 0.5
    PREDICTIONS.labels(prediction=str(will_leave).lower()).inc()
    PREDICTION_LATENCY.observe(time.perf_counter() - started_at)
    return {"will_leave": will_leave, "probability": round(probability, 3)}
