import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("empleados.csv")
X = df.drop("se_fue", axis=1)   # features
y = df["se_fue"]                # target

# Partimos los datos: el modelo NO verá el 20% durante el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)          # <-- AQUÍ ocurre el aprendizaje

pred = modelo.predict(X_test)
print("Precisión:", round(accuracy_score(y_test, pred), 3))

joblib.dump(modelo, "modelo.pkl")     # <-- el artefacto
