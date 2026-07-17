import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml, json

p = yaml.safe_load(open("params.yaml"))["entrenar"]

df = pd.read_csv("empleados.csv")
X = df.drop("se_fue", axis=1)   # features
y = df["se_fue"]                # target

# Partimos los datos: el modelo NO verá el 20% durante el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p["test_size"], random_state=p["random_state"])

modelo = RandomForestClassifier(n_estimators=p["n_estimators"], random_state=p["random_state"])
modelo.fit(X_train, y_train)          # <-- AQUÍ ocurre el aprendizaje

acc = accuracy_score(y_test, modelo.predict(X_test))
json.dump({"accuracy": round(acc, 4)}, open("metrics.json", "w"))

joblib.dump(modelo, "modelo.pkl")     # <-- el artefacto
