import pandas as pd

data = pd.read_csv("Data.csv", sep=";")

print("\n--- DIMENSIONES DE LA DATA ---")
print(data.shape)

print("\n--- NOMBRES DE LAS COLUMNAS ---")
print(data.columns.tolist())

print("\n--- VALORES NULOS POR COLUMNA ---")
print(data.isnull().sum())

print("\n--- FILAS DUPLICADAS ---")
print(data.duplicated().sum())

print("\n--- TIPOS DE DATOS ---")
print(data.dtypes)


import pandas as pd

# 1. Leer la base de datos
data = pd.read_csv("Data.csv", sep=";")

# 2. Eliminar columnas vacías tipo "Unnamed"
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# 3. Eliminar filas duplicadas
data = data.drop_duplicates()

# 4. Mostrar resultado
print("\n--- DATA DESPUÉS DE LA LIMPIEZA ---")
print(data)

print("\n--- DIMENSIONES DESPUÉS DE LIMPIAR ---")
print(data.shape)

# 5. Guardar una nueva base limpia
data.to_csv("Data_limpia.csv", sep=";", index=False)

print("\nArchivo Data_limpia.csv creado correctamente.")


print("\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
print(data.describe())


print("\n--- MATRIZ DE CORRELACIÓN ---")

columnas_numericas = data.select_dtypes(include="number")

correlacion = columnas_numericas.corr()

print(correlacion)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlacion,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matriz de correlación de variables numéricas")
plt.tight_layout()
# plt.show()

print("\n--- COLUMNAS QUE CONTIENEN LA PALABRA FATIGA ---")

for columna in data.columns:
    if "fatiga" in columna.lower():
        print(repr(columna))
        # Buscar automáticamente las columnas correctas

col_traccion = next(
    col for col in data.columns
    if "resistencia última a la tracción" in col.lower()
)

col_fatiga = next(
    col for col in data.columns
    if "resistencia a fatiga" in col.lower()
)

print("\nColumna de tracción encontrada:")
print(col_traccion)

print("\nColumna de fatiga encontrada:")
print(col_fatiga)

# Crear gráfico

plt.figure(figsize=(7,5))

sns.scatterplot(
    data=data,
    x=col_traccion,
    y=col_fatiga,
    s=100
)

plt.title("Resistencia a tracción vs resistencia a fatiga")
plt.xlabel("Resistencia última a la tracción (MPa)")
plt.ylabel("Resistencia a fatiga a 10^8 ciclos (MPa)")

plt.tight_layout()
# plt.show()


col_fluencia = next(
    col for col in data.columns
    if "límite convencional de fluencia" in col.lower()
)

plt.figure(figsize=(7,5))

sns.scatterplot(
    data=data,
    x=col_fluencia,
    y=col_fatiga,
    s=100
)

plt.title("Límite de fluencia vs resistencia a fatiga")
plt.xlabel("Límite convencional de fluencia al 0.2% (MPa)")
plt.ylabel("Resistencia a fatiga a 10^8 ciclos (MPa)")

col_fluencia_ciclica = next(
    col for col in data.columns
    if "fluencia cíclica" in col.lower()
)

print("\nColumna de fluencia cíclica encontrada:")
print(col_fluencia_ciclica)

plt.figure(figsize=(7,5))

sns.scatterplot(
    data=data,
    x=col_fluencia_ciclica,
    y=col_fatiga,
    s=100
)

plt.title("Fluencia cíclica vs resistencia a fatiga")
plt.xlabel("Esfuerzo de fluencia cíclica (MPa)")
plt.ylabel("Resistencia a fatiga a 10^8 ciclos (MPa)")

plt.tight_layout()
# plt.show()
print("\nCOLUMNAS CON LA PALABRA CICLICA:")

for col in data.columns:
    if "cíclica" in col.lower() or "ciclica" in col.lower():
        print(repr(col))


from scipy.stats import pearsonr, spearmanr

x = data[col_traccion]
y = data[col_fatiga]

pearson_r, pearson_p = pearsonr(x, y)
spearman_r, spearman_p = spearmanr(x, y)

print("\n--- CORRELACIÓN PEARSON ---")
print("Coeficiente r =", round(pearson_r, 3))
print("p-valor =", round(pearson_p, 4))

print("\n--- CORRELACIÓN SPEARMAN ---")
print("Coeficiente rho =", round(spearman_r, 3))
print("p-valor =", round(spearman_p, 4))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Variables
X = data[[col_traccion]]
y = data[col_fatiga]

# Crear modelo
modelo = LinearRegression()

# Ajustar modelo
modelo.fit(X, y)

# Predicción
y_pred = modelo.predict(X)

# Parámetros
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_
r2 = r2_score(y, y_pred)

print("\n--- REGRESIÓN LINEAL ---")
print("Pendiente =", round(pendiente, 4))
print("Intercepto =", round(intercepto, 4))
print("R² =", round(r2, 4))

print("\nEcuación:")
print(f"Fatiga = {intercepto:.4f} + {pendiente:.4f} * Resistencia_traccion")

# Gráfico
plt.figure(figsize=(7,5))

plt.scatter(X, y, label="Datos reales")
plt.plot(X, y_pred, label="Regresión lineal")

plt.xlabel("Resistencia última a la tracción (MPa)")
plt.ylabel("Resistencia a fatiga a 10^8 ciclos (MPa)")
plt.title("Regresión lineal: tracción vs fatiga")

plt.legend()
plt.tight_layout()
#plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("\n--- COMPARACIÓN DE MODELOS ---")

# Variables de entrada
X_modelo = data[[
    col_traccion,
    col_fluencia,
    col_fluencia_ciclica
]]

# Variable objetivo
y_modelo = data[col_fatiga]

# Validación Leave-One-Out
loo = LeaveOneOut()

# -------------------------
# 1. REGRESIÓN LINEAL
# -------------------------

modelo_lineal = LinearRegression()

pred_lineal = cross_val_predict(
    modelo_lineal,
    X_modelo,
    y_modelo,
    cv=loo
)

mae_lineal = mean_absolute_error(y_modelo, pred_lineal)
rmse_lineal = np.sqrt(mean_squared_error(y_modelo, pred_lineal))
r2_lineal = r2_score(y_modelo, pred_lineal)

print("\nREGRESIÓN LINEAL")
print("R² =", round(r2_lineal, 4))
print("MAE =", round(mae_lineal, 4), "MPa")
print("RMSE =", round(rmse_lineal, 4), "MPa")


# -------------------------
# 2. RANDOM FOREST
# -------------------------

modelo_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pred_rf = cross_val_predict(
    modelo_rf,
    X_modelo,
    y_modelo,
    cv=loo
)

mae_rf = mean_absolute_error(y_modelo, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_modelo, pred_rf))
r2_rf = r2_score(y_modelo, pred_rf)

print("\nRANDOM FOREST")
print("R² =", round(r2_rf, 4))
print("MAE =", round(mae_rf, 4), "MPa")
print("RMSE =", round(rmse_rf, 4), "MPa")


resultados = pd.DataFrame({
    "Modelo": [
        "Regresión Lineal",
        "Random Forest"
    ],
    "R2": [
        r2_lineal,
        r2_rf
    ],
    "MAE_MPa": [
        mae_lineal,
        mae_rf
    ],
    "RMSE_MPa": [
        rmse_lineal,
        rmse_rf
    ]
})

print("\n--- RESUMEN DE MODELOS ---")
print(resultados)



plt.figure(figsize=(7,5))

sns.barplot(
    data=resultados,
    x="Modelo",
    y="RMSE_MPa"
)

plt.title("Comparación de modelos")
plt.ylabel("RMSE (MPa)")
plt.xlabel("Modelo")

plt.tight_layout()
plt.show()

print("\n--- COMPARACIÓN FINAL DE MODELOS ---")
print(resultados.round(4))

# ============================================================
# IMPORTANCIA DE VARIABLES - RANDOM FOREST
# ============================================================

# Entrenar Random Forest con todos los datos
modelo_rf.fit(X_modelo, y_modelo)

# Obtener importancia de variables
importancias = pd.DataFrame({
    "Variable": X_modelo.columns,
    "Importancia": modelo_rf.feature_importances_
})

# Ordenar de mayor a menor
importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

# Mostrar tabla en terminal
print("\n--- IMPORTANCIA DE VARIABLES ---")
print(importancias)

# Crear gráfico
plt.figure(figsize=(9, 5))

sns.barplot(
    data=importancias,
    x="Importancia",
    y="Variable"
)

plt.title("Importancia de variables - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Variable")

plt.tight_layout()
plt.show()

# ============================================================
# VALORES REALES VS PREDICHOS - REGRESIÓN LINEAL
# ============================================================

comparacion_prediccion = pd.DataFrame({
    "Valor_real_MPa": y_modelo.values,
    "Valor_predicho_MPa": pred_lineal
})

comparacion_prediccion["Error_MPa"] = (
    comparacion_prediccion["Valor_real_MPa"]
    - comparacion_prediccion["Valor_predicho_MPa"]
)

print("\n--- VALORES REALES VS PREDICHOS ---")
print(comparacion_prediccion.round(2))

plt.figure(figsize=(6, 6))

plt.scatter(
    y_modelo,
    pred_lineal,
    s=100
)

minimo = min(y_modelo.min(), pred_lineal.min())
maximo = max(y_modelo.max(), pred_lineal.max())

plt.plot(
    [minimo, maximo],
    [minimo, maximo],
    linestyle="--",
    label="Predicción perfecta"
)

plt.xlabel("Valor real de resistencia a fatiga (MPa)")
plt.ylabel("Valor predicho de resistencia a fatiga (MPa)")
plt.title("Valores reales vs predichos")

plt.legend()
plt.tight_layout()
plt.show()

# Crear tabla de datos
comparacion_prediccion = pd.DataFrame({
    "Valor_real_MPa": y_modelo.values,
    "Valor_predicho_MPa": pred_lineal
})

comparacion_prediccion["Error_MPa"] = (
    comparacion_prediccion["Valor_real_MPa"]
    - comparacion_prediccion["Valor_predicho_MPa"]
)

print("\n--- VALORES REALES VS PREDICHOS ---")
print(comparacion_prediccion.round(2))


# TABLA GRÁFICA
fig, ax = plt.subplots(figsize=(8, 3))

ax.axis("off")

tabla = ax.table(
    cellText=comparacion_prediccion.round(2).values,
    colLabels=comparacion_prediccion.columns,
    loc="center",
    cellLoc="center"
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.5)

plt.title("Comparación de valores reales y predichos")
plt.tight_layout()
plt.show()