"""T3: depuración, EDA, Basquin, regresión/Ridge/RF, validación y PCS preliminar.
Instalar: python -m pip install pandas numpy matplotlib scikit-learn openpyxl
Ejecutar desde cualquier directorio: python "ruta/al/main_fatiga_T3.py"
"""
from pathlib import Path
import py_compile
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
ENTRADA = BASE / 'DATA' / 'Fatiga_Poblacion_Depurada_T3.xlsx'
SALIDA = BASE / 'Results'
SALIDA.mkdir(exist_ok=True, parents=True)
if not ENTRADA.exists():
    raise FileNotFoundError(f'No se encontro: {ENTRADA}')

# 1. DATOS BRUTOS Y AUDITORIA
fab = pd.read_excel(ENTRADA, sheet_name='FABEST_COMPLETO')
pre = pd.read_excel(ENTRADA, sheet_name='BASE_SELECCIONADA')
hectors = pre[pre['fuente'].astype(str).str.contains('Hectors', case=False, na=False)].copy()
base = pd.concat([fab, hectors], ignore_index=True)
base['id_ensayo'] = base['id_ensayo'].astype(str)
base = base.drop_duplicates().reset_index(drop=True).copy()
base['uid_fila'] = np.arange(1, len(base)+1)  # el ID original puede repetirse entre tablas
for c in ['R','amplitud_MPa','media_MPa','frecuencia_Hz','ciclos_observados','falla_1_censura_0']:
    base[c] = pd.to_numeric(base[c], errors='coerce')
base['es_fabest'] = base['fuente'].astype(str).str.upper().eq('FABEST')

# 2. FILTRADO REPRODUCIBLE: se conserva todo lo excluido con motivo.
def evaluar(x):
    if pd.isna(x['amplitud_MPa']) or x['amplitud_MPa'] <= 0:
        return 'Esfuerzo no valido'
    if pd.isna(x['ciclos_observados']) or x['ciclos_observados'] <= 0:
        return 'Ciclos no validos'
    if x['falla_1_censura_0'] not in (0, 1):
        return 'Estado de falla desconocido'
    if x['es_fabest']:
        if str(x['seleccion']).strip().upper() != 'INCLUIR':
            return 'FABEST: clasificacion previa no validada para inclusion'
        if pd.isna(x['R']) or not np.isclose(x['R'], -1, atol=.001):
            return 'R distinto de -1'
        if 'axial' not in str(x['tipo_carga']).lower():
            return 'Carga no axial'
        if 'unnotched' not in str(x['geometria']).lower():
            return 'Probeta con entalla u otra geometria'
        nota = str(x['nota_o_direccion']).strip().lower()
        estado = str(x['estado']).strip().lower()
        if estado == 'finished' and nota in ('no crack found', 'nan'):
            return 'Evento ambiguo; revisar fuente FABEST'
    return 'SELECCION_PRELIMINAR'

base['motivo'] = base.apply(evaluar, axis=1)
base['log10_N'] = np.log10(base['ciclos_observados'].where(base['ciclos_observados'] > 0))
base['log10_sigma'] = np.log10(base['amplitud_MPa'].where(base['amplitud_MPa'] > 0))
sel = base.loc[base['motivo'].eq('SELECCION_PRELIMINAR')].copy()
exc = base.loc[~base['motivo'].eq('SELECCION_PRELIMINAR')].copy()
fallas = sel.loc[sel['falla_1_censura_0'].eq(1)].copy()
cens = sel.loc[sel['falla_1_censura_0'].eq(0)].copy()
conteos = sel.groupby(['fuente','material','falla_1_censura_0']).size().rename('cantidad').reset_index()
exclusiones = exc['motivo'].value_counts().rename_axis('motivo').reset_index(name='cantidad')
print(f'BASE: {len(base)} | SELECCION: {len(sel)} | FALLAS: {len(fallas)} | CENSURADOS: {len(cens)}')

# 3. ESTADISTICA, CORRELACION Y DISPERSION POR NIVEL DE ESFUERZO
vars_ = ['amplitud_MPa','frecuencia_Hz','ciclos_observados','log10_N']
descriptiva = sel[vars_].describe(percentiles=[.05,.25,.5,.75,.95]).T
por_material = sel.groupby('material')[vars_].agg(['count','mean','std','min','median','max'])
correlaciones = {m:g[['amplitud_MPa','frecuencia_Hz','log10_sigma','log10_N']].corr()
                 for m,g in fallas.groupby('material')}
# intervalos de 10 MPa como exploracion; no equivalen necesariamente a esfuerzos identicos
fallas['nivel_10MPa'] = (fallas['amplitud_MPa']/10).round()*10
dispersion = fallas.groupby(['material','nivel_10MPa']).agg(
    n=('log10_N','size'), logN_media=('log10_N','mean'),
    logN_std=('log10_N','std'), logN_min=('log10_N','min'),
    logN_max=('log10_N','max')).reset_index()
dispersion = dispersion.loc[dispersion['n'] >= 2]

# 4. GRAFICOS DESCRIPTIVOS (TODOS EN RESULTS)
def guardar(nombre):
    plt.tight_layout(); plt.savefig(SALIDA / nombre, dpi=170); plt.close()

for mat,g in sel.groupby('material'):
    f = g[g['falla_1_censura_0'].eq(1)]
    c = g[g['falla_1_censura_0'].eq(0)]
    plt.figure(figsize=(8,5))
    plt.scatter(f['ciclos_observados'], f['amplitud_MPa'],alpha=.65,label='Fallas')
    if len(c):
        plt.scatter(c['ciclos_observados'],c['amplitud_MPa'],marker='>',facecolors='none',edgecolors='black',label='Censurados')
    plt.xscale('log');plt.xlabel('Ciclos N (log)');plt.ylabel('Amplitud (MPa)')
    plt.title(f'S-N experimental - {mat}');plt.legend()
    guardar('01_SN_'+('FABEST' if '42Cr' in mat else 'A37JC')+'.png')

for c,nombre in [('amplitud_MPa','02_Histograma_esfuerzo.png'),('log10_N','03_Histograma_logN.png')]:
    plt.figure(figsize=(8,5))
    for mat,g in fallas.groupby('material'):
        plt.hist(g[c].dropna(),bins=18,alpha=.55,label=mat)
    plt.xlabel(c);plt.ylabel('Numero de ensayos');plt.title('Distribucion - ensayos con falla');plt.legend()
    guardar(nombre)

plt.figure(figsize=(8,5))
for mat,g in fallas.groupby('material'):
    plt.scatter(g['amplitud_MPa'],g['frecuencia_Hz'],alpha=.5,label=mat)
plt.xlabel('Esfuerzo (MPa)');plt.ylabel('Frecuencia (Hz)');plt.legend()
plt.title('Colinealidad esfuerzo-frecuencia (por material)')
guardar('04_Esfuerzo_frecuencia.png')

for mat,corr in correlaciones.items():
    plt.figure(figsize=(7,6))
    plt.imshow(corr, vmin=-1,vmax=1,cmap='coolwarm')
    plt.xticks(range(len(corr)),corr.columns,rotation=40,ha='right')
    plt.yticks(range(len(corr)),corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if pd.notna(corr.iat[i,j]):
                plt.text(j,i,f'{corr.iat[i,j]:.2f}',ha='center',va='center',fontsize=9)
    plt.colorbar(label='Pearson r');plt.title('Matriz de correlacion - '+mat)
    guardar('05_Correlacion_'+('FABEST' if '42Cr' in mat else 'A37JC')+'.png')

plt.figure(figsize=(9,5))
for mat,g in dispersion.groupby('material'):
    plt.scatter(g['nivel_10MPa'],g['logN_std'],s=20+12*g['n'],label=mat,alpha=.75)
plt.xlabel('Nivel aproximado de esfuerzo, intervalos 10 MPa')
plt.ylabel('Desviacion estandar log10(N)');plt.legend()
plt.title('Dispersion de vida en niveles de esfuerzo con 2+ ensayos')
guardar('06_Dispersion_por_esfuerzo.png')

# 5. BASQUIN CLASICO, SOLO FALLAS EXACTAS; AJUSTE ≠ VALIDACION
basquin_rows=[]
for mat,g in fallas.groupby('material'):
    if len(g)<3 or g['log10_sigma'].nunique()<2: continue
    modelo=LinearRegression().fit(g[['log10_sigma']],g['log10_N'])
    yhat=modelo.predict(g[['log10_sigma']])
    basquin_rows.append(dict(material=mat,n_fallas=len(g),a=modelo.intercept_,b=modelo.coef_[0],
                             R2_ajuste=r2_score(g['log10_N'],yhat),
                             MAE_ajuste=mean_absolute_error(g['log10_N'],yhat)))
    xx=np.linspace(g['log10_sigma'].min(),g['log10_sigma'].max(),100)
    plt.figure(figsize=(8,5))
    plt.scatter(g['log10_sigma'],g['log10_N'],alpha=.55,label='Ensayos con falla')
    plt.plot(xx,modelo.predict(pd.DataFrame({'log10_sigma':xx})),label='Basquin (ajuste)')
    plt.xlabel('log10(esfuerzo MPa)');plt.ylabel('log10(ciclos hasta falla)')
    plt.title('Basquin - '+mat);plt.legend()
    guardar('07_Basquin_'+('FABEST' if '42Cr' in mat else 'A37JC')+'.png')
basquin_tab=pd.DataFrame(basquin_rows)

# 6. VALIDACION PREDICTIVA. Mismo esquema para Basquin y ML dentro de cada material.
metricas=[]; predicciones=[]; importancias=[]; estabilidad=[]
for mat,g in fallas.groupby('material'):
    g=g.reset_index(drop=True)
    if len(g)<8 or g['log10_sigma'].nunique()<3: continue
    grupo=g['grupo_o_tabla'].fillna(g['fuente']).astype(str)
    if grupo.nunique()>=3:
        cv=list(GroupKFold(n_splits=min(5,grupo.nunique())).split(g,groups=grupo))
        metodo='GroupKFold (grupos experimentales)'
    else:
        cv=list(KFold(n_splits=5,shuffle=True,random_state=42).split(g))
        metodo='KFold 5 (independencia por verificar)'
    # Comparar modelos con esfuerzo solamente vs esfuerzo+frecuencia (si es util)
    configuraciones=[('Basquin', ['log10_sigma'], LinearRegression()),
      ('Ridge', ['log10_sigma'],Pipeline([('scale',StandardScaler()),('reg',Ridge(alpha=1.))])),
      ('Random Forest',['log10_sigma'],RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=42,n_jobs=1))]
    if g['frecuencia_Hz'].notna().all() and g['frecuencia_Hz'].nunique()>1:
        configuraciones += [
          ('Lineal + frecuencia',['log10_sigma','frecuencia_Hz'],LinearRegression()),
          ('RF + frecuencia',['log10_sigma','frecuencia_Hz'],RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=42,n_jobs=1))]
    for nombre,cols,est in configuraciones:
        yh=np.full(len(g),np.nan)
        for tr,te in cv:
            est.fit(g.loc[tr,cols],g.loc[tr,'log10_N'])
            yh[te]=est.predict(g.loc[te,cols])
        yt=g['log10_N'].to_numpy()
        metricas.append(dict(material=mat,modelo=nombre,validacion=metodo,n=len(g),
                             variables=', '.join(cols),MAE=mean_absolute_error(yt,yh),
                             RMSE=np.sqrt(mean_squared_error(yt,yh)),R2=r2_score(yt,yh)))
        predicciones.append(pd.DataFrame(dict(material=mat,modelo=nombre,id_ensayo=g['id_ensayo'],uid_fila=g['uid_fila'],
                                                  logN_real=yt,logN_predicho=yh, error=yh-yt)))
        plt.figure(figsize=(6,6));plt.scatter(yt,yh,alpha=.65)
        lo=min(yt.min(),yh.min());hi=max(yt.max(),yh.max())
        plt.plot([lo,hi],[lo,hi],'--',color='black');plt.xlabel('logN real');plt.ylabel('logN predicho (CV)')
        plt.title(f'{mat} - {nombre}');guardar('08_Pred_'+('FABEST' if '42Cr' in mat else 'A37JC')+'_'+nombre.replace(' ','_').replace('+','mas')+'.png')
    # PCS estabilidad exploratoria: remuestreo de coeficientes Basquin, no CI formal.
    rng=np.random.default_rng(42)
    for _ in range(100):
        ind=rng.integers(0,len(g),len(g)); b=g.iloc[ind]
        if b['log10_sigma'].nunique()<2: continue
        z=LinearRegression().fit(b[['log10_sigma']],b['log10_N'])
        estabilidad.append(dict(material=mat,a=z.intercept_,b=z.coef_[0]))

cv_tab=pd.DataFrame(metricas)
pred_tab=pd.concat(predicciones,ignore_index=True) if predicciones else pd.DataFrame()
estab_tab=pd.DataFrame(estabilidad)
if len(cv_tab):
    for mat,g in cv_tab.groupby('material'):
        plt.figure(figsize=(9,5));plt.bar(g['modelo'],g['MAE']);plt.xticks(rotation=30,ha='right')
        plt.ylabel('MAE validado en log10(N)');plt.title('Comparacion de modelos - '+mat)
        guardar('09_MAE_'+('FABEST' if '42Cr' in mat else 'A37JC')+'.png')

# 7. REPORTE DE LIMITACIONES Y PCS
notas=pd.DataFrame({'tema':['Fuente','Poblacion','R=-1','Run-outs','Basquin','Evaluacion','PCS-P','PCS-C','PCS-S','Limitacion'],
'detalle':['FABEST y Hectors, sin mezclar coeficientes de materiales',
'Seleccion preliminar, pendiente contrastar eventos ambiguos con la fuente',
'Filtro aplicado a FABEST; verificar compatibilidad R de Hectors',
'Conservados y graficados, no se entrenan como fallas exactas',
'Basquin por material con regresion de logN respecto a log esfuerzo',
'Modelos comparados con la misma CV por material; grupo ≠ necesariamente lote independiente',
'Predictibilidad: MAE, RMSE y R2 CV',
'Computabilidad: filtros y parametros registrados en codigo',
'Estabilidad: bootstrap exploratorio de a y b (100 repeticiones)',
'Revisar ascenso anomalo de rama S-N, eventos ambiguos y comparabilidad de ensayos']})
resumen=pd.DataFrame([{'concepto':'Total inicial','n':len(base)},
                      {'concepto':'Incluidos preliminarmente','n':len(sel)},
                      {'concepto':'Fallas','n':len(fallas)},
                      {'concepto':'Censurados','n':len(cens)},
                      {'concepto':'Excluidos','n':len(exc)}])
with pd.ExcelWriter(SALIDA / 'Resultados_T3.xlsx',engine='openpyxl') as wr:
    for nombre,d in [('RESUMEN',resumen),('CONTEOS',conteos),('BASE_ORIGEN',base),
                     ('FILTRADOS',sel),('FALLAS',fallas),('CENSURADOS',cens),
                     ('EXCLUIDOS',exc),('MOTIVOS_EXCLUSION',exclusiones),
                     ('DESCRIPTIVA',descriptiva),('POR_MATERIAL',por_material),
                     ('DISPERSION',dispersion),('BASQUIN',basquin_tab),
                     ('MODELOS_CV',cv_tab),('PREDICCIONES',pred_tab),
                     ('ESTABILIDAD',estab_tab),('NOTAS_PCS',notas)]:
        d.to_excel(wr,sheet_name=nombre,index=nombre in ['DESCRIPTIVA','POR_MATERIAL'])
    for mat,c in correlaciones.items():
        c.to_excel(wr,sheet_name=('CORR_FABEST' if '42Cr' in mat else 'CORR_A37JC'))
print('\nLISTO: Excel y graficos guardados en:',SALIDA)
print('AVISO: resultados exploratorios; la clasificacion y la censura requieren verificacion.')


# 8. TABLAS DESCRIPTIVAS EN IMAGEN PNG

def guardar_tabla_png(datos, titulo, nombre_archivo):
    tabla = datos.copy()

    # Convertir encabezados multinivel
    if isinstance(tabla.columns, pd.MultiIndex):
        tabla.columns = [
            " - ".join(map(str, col))
            for col in tabla.columns
        ]

    # Convertir indices a columnas
    tabla = tabla.reset_index()

    # Redondear valores numericos
    for col in tabla.select_dtypes(include="number").columns:
        tabla[col] = tabla[col].round(3)

    # Convertir encabezados a texto
    tabla.columns = [str(col) for col in tabla.columns]

    ancho = max(12, len(tabla.columns) * 1.6)
    alto = max(3, len(tabla) * 0.4 + 1.5)

    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")

    visual = ax.table(
        cellText=tabla.fillna("—").astype(str).values,
        colLabels=tabla.columns.tolist(),
        cellLoc="center",
        loc="center"
    )

    visual.auto_set_font_size(False)
    visual.set_fontsize(9)
    visual.scale(1, 1.4)

    for (fila, columna), celda in visual.get_celld().items():
        if fila == 0:
            celda.set_facecolor("#17365D")
            celda.set_text_props(color="white", weight="bold")
        elif fila % 2 == 0:
            celda.set_facecolor("#EAF1F8")

    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=20)

    fig.savefig(
        SALIDA / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# Generar tabla general
guardar_tabla_png(
    descriptiva,
    "Estadistica descriptiva general",
    "10_Tabla_Descriptiva.png"
)

# Generar tabla por material
guardar_tabla_png(
    por_material,
    "Estadistica descriptiva por material",
    "11_Tabla_Por_Material.png"
)

print("Tablas descriptivas PNG generadas correctamente.")


from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "Results" / "Resultados_T3.xlsx"
SALIDA = BASE / "Results" / "12_Revision_Poblacion.xlsx"

# Leer los resultados existentes
filtrados = pd.read_excel(ARCHIVO, sheet_name="FILTRADOS")
fallas = pd.read_excel(ARCHIVO, sheet_name="FALLAS")
censurados = pd.read_excel(ARCHIVO, sheet_name="CENSURADOS")
excluidos = pd.read_excel(ARCHIVO, sheet_name="EXCLUIDOS")

# Conteos generales
resumen = pd.DataFrame({
    "Clasificacion": [
        "Seleccionados",
        "Fallas",
        "Censurados",
        "Excluidos"
    ],
    "Cantidad": [
        len(filtrados),
        len(fallas),
        len(censurados),
        len(excluidos)
    ]
})

# Conteos por material
conteos = []

for material, grupo in filtrados.groupby("material"):
    conteos.append({
        "Material": material,
        "Total": len(grupo),
        "Fallas": len(fallas[fallas["material"] == material]),
        "Censurados": len(
            censurados[censurados["material"] == material]
        )
    })

por_material = pd.DataFrame(conteos)

# Conteos por frecuencia
por_frecuencia = (
    filtrados.groupby(["material", "frecuencia_Hz"])
    .size()
    .reset_index(name="Cantidad")
)

# Conteos por grupo experimental
por_grupo = (
    filtrados.groupby(["material", "grupo_o_tabla"], dropna=False)
    .size()
    .reset_index(name="Cantidad")
)

# Exportar resultados
with pd.ExcelWriter(SALIDA, engine="openpyxl") as writer:
    resumen.to_excel(writer, sheet_name="RESUMEN", index=False)
    por_material.to_excel(writer, sheet_name="POR_MATERIAL", index=False)
    por_frecuencia.to_excel(writer, sheet_name="POR_FRECUENCIA", index=False)
    por_grupo.to_excel(writer, sheet_name="POR_GRUPO", index=False)

print("\nREVISION DE POBLACION")
print(resumen.to_string(index=False))

print("\nCLASIFICACION POR MATERIAL")
print(por_material.to_string(index=False))

print("\nArchivo generado:", SALIDA)
