import pandas as pd
import json
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from prediccion_fuga.data import FEATURES


mlflow.set_tracking_uri("http://127.0.0.1:5000")

challenger = joblib.load("models/modelo.pkl")

evaluation = pd.read_csv("data/evaluation.csv")

client = MlflowClient()

X_eval = evaluation[FEATURES]
y_eval = evaluation["se_fue"]
champion = mlflow.sklearn.load_model(
            "models:/employee-attrition@champion"
        )

champion_predict = champion.predict(X_eval)
challenger_predict = challenger.predict(X_eval)

champion_metrics = {
    "accuracy": accuracy_score(y_eval, champion_predict),
    "precision": precision_score(y_eval, champion_predict),
    "recall": recall_score(y_eval, champion_predict),
    "f1": f1_score(y_eval, champion_predict)
}

challenger_metrics = {
    "accuracy": accuracy_score(y_eval, challenger_predict),
    "precision": precision_score(y_eval, challenger_predict),
    "recall": recall_score(y_eval, challenger_predict),
    "f1": f1_score(y_eval, challenger_predict)
}

with open("metrics/comparison.json", "w") as file:
    json.dump({
        "champion": champion_metrics,
        "challenger": challenger_metrics
    }, file)

print("Champion metrics:", champion_metrics)
print("Challenger metrics:", challenger_metrics)
        