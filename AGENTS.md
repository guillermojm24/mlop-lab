# Repository Guide

## Environment
- Use the repository virtual environment: `.venv/bin/python` and `.venv/bin/dvc`.
- Install pipeline dependencies with `.venv/bin/pip install -r requirements.txt`.
- Commands below run from the repository root. The package is exposed with `PYTHONPATH=src`.

## Pipeline
- `dvc.yaml` is the executable source of truth: `preparar` runs `prediccion_fuga.data` to generate `data/empleados.csv`, then `entrenar` runs `prediccion_fuga.train` to produce `models/modelo.pkl` and `metrics/metrics.json`.
- Change pipeline parameters only in `params.yaml`; DVC tracks the listed parameter keys and output hashes in `dvc.lock`.
- Rebuild and validate the full pipeline with `.venv/bin/dvc repro`. Check whether regeneration is needed with `.venv/bin/dvc status`.
- `data/empleados.csv` and `models/modelo.pkl` are generated and Git-ignored. `metrics/metrics.json` is a tracked DVC metric (`cache: false`).

## Focused Checks
- Run an inference smoke check with `PYTHONPATH=src .venv/bin/python -m prediccion_fuga.predict`.
- Run the drift demonstrations with `PYTHONPATH=src .venv/bin/python -m prediccion_fuga.drift` and `... -m prediccion_fuga.drift_ciego`.

## Serving
- `src/prediccion_fuga/api.py` loads a local MLflow model from `models/modelo_servido/`; it does not contact the tracking server at request time.
- Before a Docker build, with MLflow available at `http://127.0.0.1:5000` and a `prediccion-fuga@champion` model alias, run `PYTHONPATH=src .venv/bin/python scripts/export_model.py`. It downloads the champion model into the Git-ignored build input.
- Build from the repository root with `docker build -f deploy/Dockerfile -t prediccion-fuga:v1 .`.
- Run the API with `docker run --rm -p 8000:8000 prediccion-fuga:v1`. Check `GET /health` and send employee JSON to `POST /predecir`.

## DVC Remote
- `.dvc/config` deliberately contains no machine-specific remote. Configure a personal or CI remote in `.dvc/config.local` or through environment-specific DVC configuration.
- Do not commit credentials, MLflow databases, local artifact stores, Docker exports or generated models.
