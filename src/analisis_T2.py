"""
T2 - Análisis exploratorio y plan algorítmico
Proyecto: Predicción de la Vida a Fatiga de Elementos de Acero

Dataset: subconjunto de 41 observaciones de falla por carga de amplitud constante,
reproducidas de las Tablas A1 y A2 de Hectors et al. (2023), Metals 13(3), 621.
No se imputan run-outs como fallas: las observaciones censuradas se excluyen
del dataset supervisado inicial para no tratar 10^7 ciclos como una falla exacta.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fatigue_A37JC_constant_amplitude_subset.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

print("\n--- DIMENSIONES ---")
print(df.shape)

print("\n--- TIPOS Y NULOS ---")
print(df.info())
print(df.isna().sum())

print("\n--- DUPLICADOS ---")
print(df.duplicated().sum())

print("\n--- ESTADÍSTICA DESCRIPTIVA ---")
print(df.describe(include="all"))

# La variable objetivo se transforma a log10 por su amplio rango.
df["log10_N"] = np.log10(df["cycles_to_failure"])

print("\n--- CORRELACIÓN NUMÉRICA ---")
print(df[["stress_amplitude_MPa","frequency_RPM","cycles_to_failure","log10_N"]].corr())

# Visualización principal
plt.figure(figsize=(8,5))
for direction in sorted(df["direction"].unique()):
    d = df[df["direction"] == direction]
    plt.scatter(d["stress_amplitude_MPa"], d["log10_N"], label=direction)
plt.xlabel("Amplitud de esfuerzo σa (MPa)")
plt.ylabel("log10(Nf)")
plt.title("Amplitud de esfuerzo vs vida a fatiga")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "eda_stress_loglife.png", dpi=180)
plt.show()

# Plan de modelos: regresión supervisada.
X = df[["stress_amplitude_MPa", "frequency_RPM", "direction"]]
y = df["log10_N"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["stress_amplitude_MPa", "frequency_RPM"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["direction"])
])

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, random_state=42, max_depth=4, min_samples_leaf=2
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=42, n_estimators=150, max_depth=2, learning_rate=0.05, loss="huber"
    )
}

print("\n--- PLAN DE VALIDACIÓN ---")
print("Train/Test = 80/20; random_state=42")
print("Objetivo = log10(Nf)")
print("Métricas = MAE, RMSE y R²")
print("La comparación final se realizará con validación cruzada en la etapa de entrenamiento.")
