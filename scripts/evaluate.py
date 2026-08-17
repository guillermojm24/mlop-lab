import json

with open("metrics/metrics.json", "r") as file:
    metrics = json.load(file)

    accuracy = metrics["accuracy"]

    if accuracy >= 0.85:
        print("Model approved")
    else:
        raise SystemExit("Model accuracy is below the required threshold")