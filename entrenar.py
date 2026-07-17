import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import yaml, json

p = yaml.safe_load(open("params.yaml"))["entrenar"]

df = pd.read_csv("empleados.csv")
X = df.drop("se_fue", axis=1)   # features
y = df["se_fue"]                # target

# Partimos los datos: el modelo NO verá el 20% durante el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p["test_size"], random_state=p["random_state"])

modelo = RandomForestClassifier(n_estimators=p["n_estimators"], random_state=p["random_state"])
modelo.fit(X_train, y_train)          # <-- AQUÍ ocurre el aprendizaje

aacc = accuracy_score(y_test, modelo.predict(X_test))
scores = cross_val_score(modelo, X_train, y_train, cv=5)

json.dump({
    "accuracy": round(aacc, 4),
    "cv_mean": round(scores.mean(), 4),
    "cv_std": round(scores.std(), 4),
}, open("metrics.json", "w"))

joblib.dump(modelo, "modelo.pkl")     # <-- el artefacto
