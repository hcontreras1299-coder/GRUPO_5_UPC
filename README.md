# GRUPO_5_UPC
Predicción de la vida a fatiga de elementos de acero mediante Machine Learning 
# INTEGRANTES
- Diego Jair De La Rosa Toro Gutierrez 
- Erick Alexander Japay Robles 
- Hans Sebastián Contreras Rodriguez 
- Fabrizio Valentino Almandoz Castañeda


# Predicción de vida a fatiga de aceros mediante Machine Learning

Proyecto orientado a la predicción de la vida a fatiga de materiales de acero mediante modelos físico-empíricos y algoritmos de Machine Learning.

## Objetivo

Comparar el modelo tradicional de **Basquin** con diferentes modelos de Machine Learning para predecir el número de ciclos hasta la falla `N` a partir de datos experimentales de fatiga.

La variable objetivo utilizada es:

`log10(N)`

donde `N` representa el número de ciclos observados hasta la falla.

## Materiales analizados

El estudio considera datos experimentales correspondientes a dos tipos de acero:

- **A37JC/S235**
- **42CrMo4+QT**

La base de datos fue sometida a un proceso de auditoría y depuración para diferenciar ensayos de falla, grietas, interrupciones y puntos auxiliares.

## Modelos evaluados

Se compararon los siguientes modelos:

- Ley de Basquin
- Regresión Ridge
- Random Forest
- Regresión lineal considerando frecuencia
- Random Forest considerando frecuencia

La relación de Basquin utilizada se expresa como:

`log10(N) = a + b · log10(σa)`

donde `σa` corresponde a la amplitud de esfuerzo en MPa.

## Metodología

El análisis comprende:

1. Auditoría y depuración de los datos experimentales.
2. Análisis estadístico descriptivo.
3. Análisis de correlación entre esfuerzo, frecuencia y vida a fatiga.
4. Construcción de curvas S-N.
5. Ajuste del modelo de Basquin para cada material.
6. Entrenamiento de modelos de Machine Learning.
7. Validación cruzada mediante KFold y GroupKFold.
8. Comparación mediante las métricas MAE, RMSE y R².

## Principales resultados

Para **A37JC/S235**, la regresión lineal incorporando la frecuencia presentó el menor MAE:

`MAE = 0.165 log10(N)`

mientras que Basquin obtuvo:

`MAE = 0.180 log10(N)`

Para **42CrMo4+QT**, Basquin presentó el mejor desempeño:

`MAE = 0.232 log10(N)`

superando a Random Forest, Ridge y los modelos que incorporaron la frecuencia.

## Conclusión

Los resultados muestran que los modelos de Machine Learning no superan de manera consistente al modelo tradicional de Basquin.

Basquin constituye un **baseline físico sólido** para la predicción de vida a fatiga, mientras que los modelos de Machine Learning pueden utilizarse como herramientas complementarias. Su desempeño depende de la calidad de los datos, la cantidad de observaciones, la selección de variables y la estrategia de validación.


