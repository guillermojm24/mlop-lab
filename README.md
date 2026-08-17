# MLOps Lab: Employee Attrition

[![CI](https://github.com/guillermojm24/mlop-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/guillermojm24/mlop-lab/actions/workflows/ci.yml)

A compact end-to-end MLOps project built to learn how reproducibility, orchestration, experiment tracking, model promotion, serving and observability fit together in a production-style workflow.

The ML problem is intentionally simple: a synthetic employee-attrition classifier. The focus of the repository is **ML platform engineering**, not model sophistication.

> The dataset is synthetic and exists only to demonstrate engineering and operational practices. It must not be used for real HR decisions.

## What this project demonstrates

- Reproducible data and training pipelines with **DVC**.
- Experiment tracking and model lifecycle management with **MLflow**.
- Workflow orchestration with **Apache Airflow**.
- An explicit model quality gate before promotion to the `champion` alias.
- Model serving through **FastAPI** and Docker.
- Kubernetes deployment with replicas, probes and resource requests/limits.
- Prometheus-compatible inference metrics for request volume and latency.
- CI validation with GitHub Actions.
- Basic data/concept-drift experiments to explore production model degradation.

## Architecture

```text
                        Git / GitHub Actions
                               |
                               v
                       Source + configuration
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
        DVC reproducibility              Airflow orchestration
   data -> train -> metrics       prepare -> train -> evaluate -> register
              |                                 |
              +----------------+----------------+
                               v
                             MLflow
                    experiments + model registry
                               |
                         quality gate
                               |
                         @champion alias
                               |
                               v
                         FastAPI service
                               |
                    Docker / Kubernetes
                               |
                  health + Prometheus metrics
```

## Repository structure

```text
.github/workflows/   CI pipeline
airflow/dags/        Airflow orchestration DAG
deploy/              Docker and Kubernetes manifests
scripts/             Operational MLflow workflow commands
src/prediccion_fuga/ Reusable Python application/ML code
tests/               Automated tests
dvc.yaml             Reproducible DVC pipeline
params.yaml           Training and promotion parameters
```

## Local reproducible pipeline

Requirements: Python 3.11+ and DVC.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

dvc repro
dvc metrics show
```

The DVC pipeline generates the synthetic dataset, trains the baseline model and stores reproducible metrics while `dvc.lock` pins inputs, parameters and outputs.

## MLflow workflow

Start or point the project at an MLflow tracking server and run:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=employee-attrition

PYTHONPATH=src python scripts/train_mlflow.py
PYTHONPATH=src python scripts/evaluate_latest_run.py
PYTHONPATH=src python scripts/register_latest_run.py
```

The evaluation step reads `evaluacion.min_accuracy` from `params.yaml`. A model that does not meet the threshold fails the workflow and is not promoted. An approved model is registered in MLflow and assigned the `champion` alias.

This separates **training** from **promotion**, so automation does not blindly move every new model into the serving path.

## Airflow orchestration

`airflow/dags/ml_pipeline.py` turns the ML workflow into independently observable and retryable tasks:

```text
prepare_data -> train -> evaluate -> register
```

Airflow is used as the operational orchestration layer: dependencies, retries, scheduling and task-level visibility. DVC remains responsible for reproducibility and dependency-aware data/model pipelines.

The DAG expects the repository to be available at `/opt/airflow/mlop-lab` by default. Override this with `MLOP_LAB_ROOT`. The MLflow endpoint can be configured with `MLFLOW_TRACKING_URI`.

## Model serving

Export the MLflow champion model before building the inference image:

```bash
PYTHONPATH=src python scripts/export_model.py

docker build -f deploy/Dockerfile -t employee-attrition-api:v1 .
docker run --rm -p 8000:8000 employee-attrition-api:v1
```

Endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"antiguedad_anios":3,"salario_k":45,"horas_extra_mes":35,"satisfaccion":0.2}'
```

The API exports Prometheus metrics including:

- `mlop_lab_predictions_total`
- `mlop_lab_prediction_latency_seconds`

## Kubernetes

The manifest in `deploy/k8s/deployment.yaml` includes:

- two inference replicas;
- readiness and liveness probes;
- CPU/memory requests and limits;
- a ClusterIP service;
- Prometheus scrape annotations.

`imagePullPolicy: Never` is deliberately configured for a local Kubernetes lab where the image is loaded directly into the cluster. A production deployment would use an image registry, immutable image tags/digests and an appropriate pull policy.

## Drift experiments

The repository includes small experiments for exploring how a model behaves as production data changes:

```bash
PYTHONPATH=src python -m prediccion_fuga.drift
PYTHONPATH=src python -m prediccion_fuga.drift_ciego
PYTHONPATH=src python -m prediccion_fuga.monitor
```

These are educational demonstrations rather than a full production drift-monitoring system.

## Design choices

**Why DVC and Airflow?** They solve different problems. DVC provides reproducibility, data/model dependency tracking and cache-aware pipeline execution. Airflow provides operational orchestration: scheduling, task dependencies, retries and visibility.

**Why MLflow?** It provides a shared lifecycle for experiments, metrics, model artifacts and registry promotion instead of treating a serialized model file as the final product.

**Why a quality gate?** Training completing successfully does not mean a model is suitable for production. Promotion is conditional on explicit evaluation criteria.

**Why Kubernetes?** The goal is to treat inference as an operable service with health checks, resource controls, horizontal replicas and observable runtime behaviour.

## Production evolution

This repository is intentionally small enough to run as a learning lab. The next production-oriented steps would be:

- remote object storage for DVC and MLflow artifacts;
- managed secrets and workload identity;
- a real image registry and GitOps-based deployment promotion;
- stronger data-quality validation and model-performance monitoring;
- Prometheus/Grafana dashboards and alerting;
- integration tests for the full training-to-serving path;
- GPU-backed model serving as a separate inference workload.

## CI

GitHub Actions runs tests, Python compilation, DVC reproduction and DVC status checks on pushes and pull requests.
