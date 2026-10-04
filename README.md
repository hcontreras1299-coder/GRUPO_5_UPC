# GRUPO_5_UPC
Predicción de la vida a fatiga de elementos de acero mediante Machine Learning 
# INTEGRANTES
- Diego Jair De La Rosa Toro Gutierrez 
- Erick Alexander Japay Robles 
- Hans Sebastián Contreras Rodriguez 
- Fabrizio Valentino Almandoz Castañeda


## Objetivo de esta entrega
Realizar el análisis exploratorio de datos y definir el plan algorítmico para un problema de regresión de vida a fatiga.

## Fuente de datos
Se utiliza un subconjunto de 41 observaciones experimentales de falla por carga de amplitud constante de acero A37JC, equivalente al S235, reproducidas de las Tablas A1 y A2 del artículo:

Hectors, K., Vanspeybrouck, D., Plets, J., Bouckaert, Q., & De Waele, W. (2023).
Open-Access Experiment Dataset for Fatigue Damage Accumulation and Life Prediction Models.
Metals, 13(3), 621. DOI: 10.3390/met13030621.

El artículo indica que el material A37JC es equivalente al S235 actual y que los ensayos se realizaron mediante flexión rotatoria. El conjunto completo de datos se encuentra en el repositorio abierto indicado por los autores:
https://doi.org/10.17605/OSF.IO/6Y5SD

## Delimitación
Se estudia fatiga de alto ciclo bajo carga de amplitud constante. La variable objetivo es el número de ciclos hasta la falla, Nf, modelado inicialmente como log10(Nf).

## Importante sobre los run-outs
Los ensayos que alcanzaron 10^7 ciclos sin falla son observaciones censuradas (run-out). No se convierten artificialmente en una falla a 10^7 ciclos en el conjunto supervisado inicial. Para la T2 se trabaja con las 41 observaciones de falla reproducidas de las tablas publicadas.

## Variables
- stress_amplitude_MPa: amplitud de esfuerzo.
- frequency_RPM: frecuencia de ensayo.
- direction: dirección de extracción del material (TD/RD).
- cycles_to_failure: número de ciclos hasta la falla.
- log10_N: transformación de la variable objetivo.

## Plan algorítmico
1. Modelo base: Ridge Regression.
2. Modelo no lineal: Random Forest Regressor.
3. Modelo no lineal: Gradient Boosting Regressor.
4. Red neuronal: NO se recomienda como modelo principal con solo 41 fallas en este subconjunto; se reconsiderará si se incorpora el dataset completo.

## Métricas
- MAE
- RMSE
- R²

## Ejecución
```bash
pip install -r requirements.txt
python src/analisis_T2.py
```

