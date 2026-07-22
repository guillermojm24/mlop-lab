# Predicción de fuga de empleados

Proyecto de MLOps reproducible para estimar el riesgo de fuga de empleados a partir de un dataset sintético. El repositorio muestra el ciclo completo: generación de datos, entrenamiento versionado con DVC, tracking con MLflow, demostraciones de drift, API FastAPI y despliegue en Docker/Kubernetes.

> El dataset es sintético y sirve para demostrar prácticas de ingeniería y operación de modelos. No representa decisiones reales de recursos humanos.

## Arquitectura

```text
                  params.yaml
                       │
                       ▼
              ┌────── DVC ──────┐
              │                 │
              ▼                 ▼
     data/empleados.csv   models/modelo.pkl
                                  │
                                  ├── predict / drift / monitor
                                  │
                                  ▼
                         MLflow → champion
                                  │
                                  ▼
                  models/modelo_servido/ → API
                                  │
                         Docker / Kubernetes
```

La lógica reutilizable vive en `src/prediccion_fuga`. Los comandos operativos están en `scripts/`, el manifiesto de imagen en `deploy/` y los tests en `tests/`. Los datos, modelos exportados y stores locales quedan fuera del repositorio.

## Inicio rápido

Requisitos: Python 3.11+ y DVC. En este entorno se puede usar el virtualenv incluido:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/dvc repro
```

El pipeline deja sus resultados en `data/empleados.csv`, `models/modelo.pkl` y `metrics/metrics.json`.

Checks principales:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m prediccion_fuga.predict
PYTHONPATH=src .venv/bin/python -m prediccion_fuga.drift
PYTHONPATH=src .venv/bin/python -m prediccion_fuga.drift_ciego
PYTHONPATH=src .venv/bin/python -m prediccion_fuga.monitor
```

## MLflow

Con un servidor MLflow local en `http://127.0.0.1:5000`:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_mlflow.py
```

La URL, el experimento y el número de árboles se pueden cambiar sin editar código mediante `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME` y `N_ESTIMATORS`.

## API y despliegue

Para servir el modelo champion registrado en MLflow, exportarlo primero:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_model.py
docker build -f deploy/Dockerfile -t prediccion-fuga:v1 .
docker run --rm -p 8000:8000 prediccion-fuga:v1
```

El contenedor expone:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predecir \
  -H 'Content-Type: application/json' \
  -d '{"antiguedad_anios":3,"salario_k":45,"horas_extra_mes":35,"satisfaccion":0.2}'
```

El manifiesto de Kubernetes está en `deploy/k8s/deployment.yaml`. `imagePullPolicy: Never` está pensado para un clúster local con la imagen cargada manualmente; para un entorno real habría que usar un registry y una política de actualización de imágenes.

## Reproducibilidad y límites

- `dvc.yaml` es la fuente ejecutable del pipeline y `dvc.lock` fija hashes y parámetros.
- `params.yaml` es el único punto de edición de los parámetros del experimento.
- El remote DVC no está fijado en el repositorio porque una ruta local sería específica de una máquina y no es almacenamiento compartido.
- No se versionan modelos binarios, bases de datos de MLflow, artefactos Docker ni datos generados.
- En un proyecto real se añadirían validación de calidad de datos, autenticación del API, observabilidad y un registro de modelos remoto.
