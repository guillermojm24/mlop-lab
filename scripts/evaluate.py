import json

from prediccion_fuga.config import load_params

min_accuracy = load_params()["evaluar"]["min_accuracy"]

with open("metrics/metrics.json", "r") as file:
    metrics = json.load(file)

    accuracy = metrics["accuracy"]

    if accuracy >= min_accuracy:
        print("Model approved")
    else:
        raise SystemExit("Model accuracy is below the required threshold")
