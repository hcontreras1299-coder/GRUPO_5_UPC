
from pathlib import Path
import pandas as pd

# 1. UBICACION DE LOS ARCHIVOS
BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "Results"

ARCHIVO = RESULTADOS / "Resultados_T3.xlsx"
SALIDA = RESULTADOS / "12_Revision_Poblacion.xlsx"

if not ARCHIVO.exists():
    raise FileNotFoundError(
        f"No se encontro el archivo: {ARCHIVO}"
    )

# 2. LEER DATOS DEL ANALISIS PRINCIPAL
filtrados = pd.read_excel(ARCHIVO, sheet_name="FILTRADOS")
fallas = pd.read_excel(ARCHIVO, sheet_name="FALLAS")
censurados = pd.read_excel(ARCHIVO, sheet_name="CENSURADOS")
excluidos = pd.read_excel(ARCHIVO, sheet_name="EXCLUIDOS")

# 3. RESUMEN GENERAL
resumen = pd.DataFrame({
    "Clasificacion": [
        "Ensayos seleccionados",
        "Ensayos con falla",
        "Ensayos censurados",
        "Ensayos excluidos"
    ],
    "Cantidad": [
        len(filtrados),
        len(fallas),
        len(censurados),
        len(excluidos)
    ]
})

# 4. CLASIFICACION POR MATERIAL
materiales = []

for material, grupo in filtrados.groupby("material"):
    materiales.append({
        "Material": material,
        "Total": len(grupo),
        "Fallas": (
            fallas["material"] == material
        ).sum(),
        "Censurados": (
            censurados["material"] == material
        ).sum()
    })

por_material = pd.DataFrame(materiales)

# 5. CLASIFICACION POR FRECUENCIA
por_frecuencia = (
    filtrados.groupby(
        ["material", "frecuencia_Hz"],
        dropna=False
    )
    .size()
    .reset_index(name="Cantidad")
)

# 6. CLASIFICACION POR GRUPO EXPERIMENTAL
por_grupo = (
    filtrados.groupby(
        ["material", "grupo_o_tabla"],
        dropna=False
    )
    .size()
    .reset_index(name="Cantidad")
)

# 7. CONTROL DE CALIDAD
control = pd.DataFrame({
    "Verificacion": [
        "Fallas + censurados",
        "Total seleccionados",
        "Coinciden los conteos"
    ],
    "Resultado": [
        len(fallas) + len(censurados),
        len(filtrados),
        len(fallas) + len(censurados) == len(filtrados)
    ]
})

# 8. EXPORTAR TODO A EXCEL
with pd.ExcelWriter(SALIDA, engine="openpyxl") as writer:
    resumen.to_excel(
        writer, sheet_name="RESUMEN", index=False
    )
    por_material.to_excel(
        writer, sheet_name="POR_MATERIAL", index=False
    )
    por_frecuencia.to_excel(
        writer, sheet_name="POR_FRECUENCIA", index=False
    )
    por_grupo.to_excel(
        writer, sheet_name="POR_GRUPO", index=False
    )
    control.to_excel(
        writer, sheet_name="CONTROL", index=False
    )

print("\nREVISION GENERAL DE LA POBLACION")
print(resumen.to_string(index=False))

print("\nCLASIFICACION POR MATERIAL")
print(por_material.to_string(index=False))

print("\nCONTROL DE CALIDAD")
print(control.to_string(index=False))

print("\nPROCESAMIENTO TERMINADO")
print("Archivo generado:", SALIDA)
