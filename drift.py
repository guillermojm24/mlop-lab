import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score

modelo = joblib.load("modelo.pkl")   # el mismo modelo de siempre, intacto

def generar(n, regla, seed):
    np.random.seed(seed)
    df = pd.DataFrame({
        "antiguedad_anios": np.random.randint(0, 15, n),
        "salario_k":        np.random.randint(20, 90, n),
        "horas_extra_mes":  np.random.randint(0, 40, n),
        "satisfaccion":     np.random.uniform(0, 1, n).round(2),
    })
    riesgo = regla(df)
    df["se_fue"] = (riesgo + np.random.normal(0, 0.1, n) > 0.55).astype(int)
    return df

# La regla del mundo en que se entrenó: quema laboral
regla_vieja = lambda d: (d.horas_extra_mes / 40) * 0.5 + (1 - d.satisfaccion) * 0.5

# El mundo cambia: llega la inflación y la gente se va por dinero, no por quemarse
regla_nueva = lambda d: (1 - (d.salario_k - 20) / 70) * 0.8 + (1 - d.satisfaccion) * 0.2

for nombre, regla, seed in [("2025 (mismo mundo)", regla_vieja, 7),
                            ("2026 (mundo nuevo)", regla_nueva, 7)]:
    df = generar(500, regla, seed)
    X, y = df.drop("se_fue", axis=1), df["se_fue"]
    pred = modelo.predict(X)
    print(f"{nombre}: precisión = {round(accuracy_score(y, pred), 3)}")
