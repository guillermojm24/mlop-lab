FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The model is exported from MLflow before the image is built.
COPY models/modelo_servido/ ./models/modelo_servido/
COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/modelo_servido

EXPOSE 8000
CMD ["uvicorn", "prediccion_fuga.api:app", "--host", "0.0.0.0", "--port", "8000"]
