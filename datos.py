import numpy as np
import pandas as pd
import yaml

p = yaml.safe_load(open("params.yaml"))["datos"]
np.random.seed(p["seed"])  # reproducibilidad: mismos datos siempre
n = p["n"]

df = pd.DataFrame({
    "antiguedad_anios": np.random.randint(0, 15, n),
    "salario_k": np.random.randint(20, 90, n),
    "horas_extra_mes": np.random.randint(0, 40, n),
    "satisfaccion": np.random.uniform(0, 1, n).round(2),
})

# La "verdad oculta" que el modelo deberá descubrir solo:
riesgo = (df.horas_extra_mes / 40) * 0.5 + (1 - df.satisfaccion) * 0.5
df["se_fue"] = (riesgo + np.random.normal(0, 0.1, n) > 0.55).astype(int)

df.to_csv("empleados.csv", index=False)
print(df.head())
print("\nSe fueron:", df.se_fue.sum(), "de", n)
