import json
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model = joblib.load("models/modelo.pkl")

with open("metrics/metrics.json", "r") as file:
    metrics = json.load(file)

    mlflow.set_experiment("employee-attrition")
    with mlflow.start_run(run_name="chad-run"):
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="employee-attrition"
        )

        client.set_registered_model_alias(
            name="employee-attrition",
            alias="champion",
            version=registered_model.version
)
