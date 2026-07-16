import joblib, pandas as pd

modelo = joblib.load("modelo.pkl")   # cargamos el artefacto

nuevo = pd.DataFrame([{
    "antiguedad_anios": 3,
    "salario_k": 45,
    "horas_extra_mes": 35,
    "satisfaccion": 0.2,
}])

print("¿Se va?:", modelo.predict(nuevo)[0])
print("Probabilidad:", modelo.predict_proba(nuevo)[0][1].round(3))
