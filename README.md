# MLOps Lab: Employee Attrition

End-to-end MLOps learning project built around a deliberately simple employee attrition classifier. The goal of the repository is not model complexity, but understanding how the different pieces of an ML platform fit together: reproducible training, orchestration, model evaluation, experiment tracking, model registry, champion/challenger comparison, serving and Kubernetes deployment.

> The dataset is synthetic and exists only to demonstrate ML engineering and MLOps practices. It must not be used for real HR decision-making.

## What this project demonstrates

- Reproducible data generation and model training with **DVC**.
- Experiment tracking and model lifecycle management with **MLflow**.
- Workflow orchestration with **Apache Airflow**.
- Explicit model quality gates before promotion.
- **Champion/challenger** comparison on a shared evaluation dataset.
- Automatic promotion of an approved challenger through the MLflow Model Registry `@champion` alias.
- Model serving through **FastAPI**.
- Containerization with **Docker** and deployment manifests for **Kubernetes**.
- Basic drift and monitoring experiments.
- CI validation through GitHub Actions.

## Architecture

```text
                       params.yaml
                            │
                            ▼
                  ┌──────── DVC ────────┐
                  │                     │
                  ▼                     ▼
          generated data         trained model.pkl
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                        Apache Airflow
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     │
     prepare_data                                │
          ↓                                     │
        train                                    │
          ↓                                     │
       evaluate  ── quality gate                 │
          ↓                                     │
   compare_models                               │
      ↙       ↘                                  │
 champion   challenger                           │
   MLflow    local model                         │
      ↘       ↙                                  │
   shared evaluation.csv                         │
          ↓                                     │
   comparison.json                               │
          ↓                                     │
   promote_models                                │
          ↓                                     │
 MLflow Model Registry                           │
          ↓                                     │
      @champion                                  │
          ↓                                     │
     export model                                │
          ↓                                     │
       FastAPI                                   │
          ↓                                     │
 Docker / Kubernetes  ◄──────────────────────────┘
```

## DVC vs Airflow

Both tools describe workflows, but they solve different problems in this repository.

**DVC** focuses on reproducibility and dependency tracking. It knows which code, parameters, datasets and artifacts belong to a training execution and can avoid recomputing unchanged stages.

**Airflow** is the orchestration layer. It defines task ordering, retries, execution state and the operational lifecycle of the full MLOps workflow.

The current Airflow DAG is:

```text
prepare_data
     ↓
   train
     ↓
 evaluate
     ↓
 compare
     ↓
 promote
```

## Repository structure

```text
.
├── airflow/
│   └── dags/
│       └── ml_pipeline.py       # End-to-end Airflow DAG
├── data/
│   └── evaluation.csv           # Shared dataset for champion/challenger evaluation
├── deploy/
│   ├── Dockerfile
│   └── k8s/
│       └── deployment.yaml
├── metrics/
│   ├── metrics.json             # Training metrics
│   └── comparison.json          # Champion/challenger metrics
├── scripts/
│   ├── evaluate.py              # Minimum model quality gate
│   ├── compare_models.py        # Evaluate champion and challenger on the same data
│   ├── promote_models.py        # Register and promote an approved challenger
│   ├── register_model.py        # Initial MLflow registration helper
│   ├── train_mlflow.py          # MLflow-tracked training
│   └── export_model.py          # Export the current champion for serving
├── src/prediccion_fuga/         # Reusable Python package
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── requirements-api.txt
└── requirements-airflow.txt
```

## Local setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the core project dependencies and install the local package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Install Airflow separately because it is only required by the orchestration environment:

```bash
pip install -r requirements-airflow.txt
```

## Reproduce the DVC pipeline

```bash
dvc repro
```

The reproducible training pipeline generates the training dataset, the local model artifact and its metrics.

## MLflow

Start a local MLflow tracking server with SQLite as the backend store:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db
```

The UI is then available at:

```text
http://127.0.0.1:5000
```

The repository uses MLflow for experiment tracking, logged models, model versions and the `@champion` alias.

### Initial model registration

After training a model, an initial registered model can be created with:

```bash
python scripts/register_model.py
```

This creates or updates the `employee-attrition` registered model and assigns the selected version the `@champion` alias.

## Champion/challenger workflow

A newly trained local model acts as the **challenger**. The current production candidate is loaded from MLflow through:

```text
models:/employee-attrition@champion
```

`compare_models.py` evaluates both models against the same `data/evaluation.csv` dataset and calculates:

- accuracy
- precision
- recall
- F1 score

The results are stored in `metrics/comparison.json`.

`promote_models.py` then evaluates the promotion rule. In the current learning implementation, the challenger must:

1. achieve a higher F1 score than the current champion, and
2. meet the configured minimum accuracy used by the promotion logic.

If both conditions are met, the challenger is logged to MLflow, registered as a new model version and the `@champion` alias is moved to that version. If not, the existing champion remains unchanged.

The thresholds in this repository are educational defaults rather than production business requirements.

## Airflow

The DAG is defined in `airflow/dags/ml_pipeline.py` and currently runs manually (`schedule=None`). Tasks have retries configured and execute from the repository root so that local project paths remain consistent.

Point Airflow at the repository DAG folder:

```bash
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
```

Initialize or migrate the Airflow metadata database:

```bash
airflow db migrate
```

Check that the DAG is discovered:

```bash
airflow dags list --local
```

Run the complete workflow locally:

```bash
airflow dags test employee_attrition $(date +%Y-%m-%d)
```

This executes:

```text
prepare_data → train → evaluate → compare → promote
```

A model failing the `evaluate` quality gate stops downstream execution. A challenger that is valid but does not beat the current champion is not considered a technical pipeline failure; it is simply not promoted.

## FastAPI serving

Export the current MLflow champion:

```bash
python scripts/export_model.py
```

Build and run the inference image:

```bash
docker build -f deploy/Dockerfile -t prediccion-fuga:v1 .
docker run --rm -p 8000:8000 prediccion-fuga:v1
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predecir \
  -H 'Content-Type: application/json' \
  -d '{"antiguedad_anios":3,"salario_k":45,"horas_extra_mes":35,"satisfaccion":0.2}'
```

## Kubernetes

The Kubernetes manifests live in `deploy/k8s/deployment.yaml` and include two inference replicas, health probes and CPU/memory requests and limits.

`imagePullPolicy: Never` is intentionally configured for a local Kubernetes environment where the image is loaded manually. A real deployment would use an image registry and an appropriate image update strategy.

## Drift experiments

The project also contains simple experiments around data and prediction drift under `src/prediccion_fuga/`. These are intended to explore how a model can degrade as the input distribution changes over time.

## Current limitations and possible next steps

This is a learning lab rather than a production platform. Some intentionally simplified areas are:

- MLflow and Airflow currently run locally.
- Training and orchestration use local filesystem artifacts.
- The promotion thresholds are static learning rules rather than business-driven policies.
- The evaluation dataset is stored directly in the repository for reproducibility of the demonstration.
- Model serving does not yet include authentication, production observability or autoscaling.
- Airflow currently executes Python commands locally rather than dispatching training workloads as Kubernetes Jobs.

Potential next steps include remote artifact storage, Kubernetes-based training jobs, model-serving observability, stronger data validation, configurable promotion policies and richer drift-triggered retraining workflows.

## Purpose

This repository is primarily a hands-on learning environment for understanding the complete ML model lifecycle from a Platform Engineering perspective. The focus is on being able to explain why each component exists and how the pieces interact, rather than simply combining tools in a demo stack.
