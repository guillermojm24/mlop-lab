"""Airflow DAG for the end-to-end model training and promotion workflow."""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.getenv("MLOP_LAB_ROOT", "/opt/airflow/mlop-lab")
TASK_ENV = {
    "PYTHONPATH": f"{PROJECT_ROOT}/src",
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "employee-attrition"),
    "PATH": os.getenv("PATH", ""),
}

with DAG(
    dag_id="employee_attrition_ml_pipeline",
    description="Prepare data, train, evaluate and promote an MLflow model.",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["mlops", "mlflow", "training"],
) as dag:
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="python -m prediccion_fuga.data",
        cwd=PROJECT_ROOT,
        env=TASK_ENV,
    )

    train = BashOperator(
        task_id="train",
        bash_command="python scripts/train_mlflow.py",
        cwd=PROJECT_ROOT,
        env=TASK_ENV,
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="python scripts/evaluate_latest_run.py",
        cwd=PROJECT_ROOT,
        env=TASK_ENV,
    )

    register = BashOperator(
        task_id="register",
        bash_command="python scripts/register_latest_run.py",
        cwd=PROJECT_ROOT,
        env=TASK_ENV,
    )

    prepare_data >> train >> evaluate >> register
