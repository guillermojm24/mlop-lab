import joblib
import mlflow
import json
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

with open("metrics/comparison.json", "r") as file:
    metrics = json.load(file)

champion_metrics = metrics["champion"]
challenger_metrics = metrics["challenger"]

challenger = joblib.load("models/modelo.pkl")

if (challenger_metrics["f1"] > champion_metrics["f1"] 
    and challenger_metrics["accuracy"] >= 0.80):

    print("Challenger model is better than the champion model.")

    with mlflow.start_run(run_name="new-champion"):
        model_info = mlflow.sklearn.log_model(
            sk_model=challenger,
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
else:
    print("Challenger model is not better than the champion model.")