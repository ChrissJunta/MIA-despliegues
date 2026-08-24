"""
train.py
--------
Genera datos simulados de sensores industriales (temperatura, vibracion,
retraso) y entrena un modelo de clasificacion (RandomForest) que predice
si un lote de produccion tendra retraso (1) o no (0), en funcion de la
temperatura y la vibracion registradas por los sensores.

Ejecutar:
    python train.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------------------------------------------------
# 1) Generar datos simulados (temperatura, vibracion, retraso)
# ---------------------------------------------------------------------
# temperatura: temperatura del sensor en grados C durante el proceso.
# vibracion:   nivel de vibracion de la maquina (escala normalizada).
# retraso:     etiqueta binaria (0 = proceso normal, 1 = retraso/falla).
#
# Se simulan dos poblaciones: una de operacion "normal" (temperatura y
# vibracion bajas, sin retraso) y otra de "falla" (temperatura y
# vibracion mas altas, con retraso), para que el modelo pueda aprender
# un patron de clasificacion realista.
np.random.seed(42)

normal = pd.DataFrame({
    "temperatura": np.random.normal(loc=70, scale=5, size=1000),
    "vibracion": np.random.normal(loc=0.5, scale=0.1, size=1000),
    "retraso": 0
})

falla = pd.DataFrame({
    "temperatura": np.random.normal(loc=85, scale=5, size=200),
    "vibracion": np.random.normal(loc=0.9, scale=0.1, size=200),
    "retraso": 1
})

df = pd.concat([normal, falla], ignore_index=True).sample(frac=1, random_state=42)

# Guardar CSV en data/
data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)
csv_path = data_path / "datos_sensor.csv"
df.to_csv(csv_path, index=False)
print(f"OK -> {csv_path} ({len(df)} filas)")

# ---------------------------------------------------------------------
# 2) Entrenar modelo
# ---------------------------------------------------------------------
X = df[["temperatura", "vibracion"]]
y = df["retraso"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 3) Precision
# ---------------------------------------------------------------------
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Precision holdout: {acc:.3f}")
print("Reporte de clasificacion:")
print(classification_report(y_test, y_pred, target_names=["sin_retraso", "retraso"]))

# ---------------------------------------------------------------------
# 4) Guardar modelo
# ---------------------------------------------------------------------
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, models_path / "modelo_sensores.pkl")
print("Modelo guardado -> models/modelo_sensores.pkl")
