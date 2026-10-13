"""T3: auditoría y análisis reproducible de datos de fatiga.
Entrada: DATA/Auditoria_Reconstruccion_Fatiga_T3.xlsx (o con (1)).
No reclasifica eventos experimentales ambiguos como fallas.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
CARPETAS = [BASE/'DATA', BASE/'Data', BASE]
ARCHIVOS = ['Auditoria_Reconstruccion_Fatiga_T3.xlsx', 'Auditoria_Reconstruccion_Fatiga_T3(1).xlsx']
rutas = [carpeta/nombre for carpeta in CARPETAS for nombre in ARCHIVOS]
ruta = next((r for r in rutas if r.is_file()), None)
if ruta is None:
    sys.exit('NO SE ENCONTRO EL EXCEL. Coloca Auditoria_Reconstruccion_Fatiga_T3.xlsx en DATA, junto a este script en la carpeta principal.')
SALIDA = BASE / 'Results_Auditados'
SALIDA.mkdir(exist_ok=True)
print('Leyendo:', ruta)
traza = pd.read_excel(ruta, sheet_name='TRAZABILIDAD')
cand = pd.read_excel(ruta, sheet_name='CANDIDATOS')
pend = pd.read_excel(ruta, sheet_name='REVISION_PENDIENTE')
aux = pd.read_excel(ruta, sheet_name='CURVAS_AUXILIARES')
# Nunca se descartan filas usando ID: algunos identificadores son compartidos por tablas de cálculo.
for d in (traza, cand, pend, aux):
    for k in ['Esfuerzo_MPa','Frecuencia_Hz','Ciclos','R']:
        d[k] = pd.to_numeric(d[k], errors='coerce')

# Sólo fallas explícitas y con condiciones elegibles. Las grietas centrales se
# separan para permitir análisis de sensibilidad si el criterio de falla difiere.
elegible = cand.loc[(cand['Condicion']=='ELEGIBLE_CONDICIONES') & (cand['Esfuerzo_MPa']>0) & (cand['Ciclos']>0)].copy()
notas = elegible['Nota_origen'].fillna('').astype(str).str.lower().str.strip()
clases = elegible['Clasificacion_auditada'].fillna('').astype(str)
elegible['evento_analisis'] = np.select([
    (elegible['Fuente'].astype(str).str.contains('Hectors',case=False,na=False)) & clases.eq('FALLA_ORIGINAL_41'),
    clases.eq('CENSURA_CONFIRMABLE'),
    notas.str.contains(r'\bbroken\b',regex=True),
    clases.eq('FALLA_O_GRIETA') & notas.str.contains('crack',case=False) & ~notas.str.contains('no crack',case=False),
],['FALLA_HECTORS','CENSURA_PROVISIONAL','ROTURA','GRIETA_REVISAR_CRITERIO'],default='PENDIENTE')


fallas = elegible[elegible.evento_analisis.isin(['FALLA_HECTORS','ROTURA','GRIETA_REVISAR_CRITERIO'])].copy()
censurados = elegible[elegible.evento_analisis=='CENSURA_PROVISIONAL'].copy()
fallas['logN'] = np.log10(fallas.Ciclos)
fallas['logEsfuerzo'] = np.log10(fallas.Esfuerzo_MPa)
elegible['logCiclosObservados'] = np.log10(elegible.Ciclos)

recuento = (elegible.groupby(['Material','evento_analisis'], dropna=False).size()
            .reset_index(name='N'))
resumen = pd.DataFrame([
    ('Filas auditoría',len(traza)), ('Candidatos auditados',len(cand)),
    ('Elegibles con valores positivos',len(elegible)),
    ('Eventos tratados como fallas (provisional)',len(fallas)),
    ('Censuras provisionales',len(censurados)),
    ('Revisión pendiente',len(pend)),('Puntos auxiliares separados',len(aux))
],columns=['Concepto','Cantidad'])

# Estadística de ciclos observados, no confundir con vida exacta hasta la falla.
variables=['Esfuerzo_MPa','Frecuencia_Hz','Ciclos','logCiclosObservados']
desc_all=elegible.groupby('Material')[variables].describe().round(3)
desc_falla=fallas.groupby('Material')[['Esfuerzo_MPa','Frecuencia_Hz','Ciclos','logN']].describe().round(3)

metrics=[]; predictions=[]; coeff=[]
for material, sub in fallas.groupby('Material'):
    if len(sub)<12: continue
    sub=sub.reset_index(drop=True)
    y=sub.logN.to_numpy()
    X0=sub[['logEsfuerzo']].to_numpy()
    freq_ok=sub.Frecuencia_Hz.notna().all() and sub.Frecuencia_Hz.nunique()>1
    configs=[('Basquin',X0,LinearRegression()),('Ridge',X0,Ridge(alpha=1.0)),
             ('RandomForest',X0,RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=1))]
    if freq_ok:
        Xf=np.column_stack([X0, sub.Frecuencia_Hz.to_numpy()])
        configs.extend([('Lineal_frecuencia',Xf,LinearRegression()),
                        ('RF_frecuencia',Xf,RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=1))])
    # Mantener grupos completos fuera del entrenamiento cuando existen suficientes.
    grupos=sub['Grupo'].fillna('SIN_GRUPO').astype(str).to_numpy()
    ng=len(set(grupos))
    if ng>=4:
        folds=list(GroupKFold(n_splits=min(5,ng)).split(X0,y,grupos))
        validacion='GroupKFold'
    else:
        folds=list(KFold(n_splits=min(5,len(y)),shuffle=True,random_state=42).split(X0,y))
        validacion='KFold (limitación: grupos insuficientes)'
    for modelo,X,est in configs:
        yh=np.full(len(y),np.nan)
        for train,test in folds:
            est.fit(X[train],y[train]); yh[test]=est.predict(X[test])
        metrics.append([material,modelo,len(y),validacion,mean_absolute_error(y,yh),
                        np.sqrt(mean_squared_error(y,yh)),r2_score(y,yh)])
        predictions.extend([[material,modelo,sub.loc[i,'ID'],float(y[i]),float(yh[i]),float(yh[i]-y[i])]
                            for i in range(len(y))])
    ajuste=LinearRegression().fit(X0,y)
    coeff.append([material,len(sub),float(ajuste.intercept_),float(ajuste.coef_[0]),float(ajuste.score(X0,y))])
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(sub.Ciclos,sub.Esfuerzo_MPa,alpha=.75,label='Fallas/grietas (provisionales)')
    cens=censurados[censurados.Material==material]
    if len(cens): ax.scatter(cens.Ciclos,cens.Esfuerzo_MPa,marker='>',label='Censura provisional')
    xx=np.linspace(X0.min(),X0.max(),120)
    ax.plot(10**ajuste.predict(xx.reshape(-1,1)),10**xx,label='Basquin (ajuste)')
    ax.set_xscale('log'); ax.set_xlabel('Ciclos observados N');ax.set_ylabel('Amplitud de esfuerzo (MPa)')
    ax.set_title('S-N: '+str(material));ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(SALIDA/('SN_'+str(material).replace('/','_')+'.png'),dpi=180);plt.close(fig)

metricas=pd.DataFrame(metrics,columns=['Material','Modelo','N_fallas','Validacion','MAE_log10N','RMSE_log10N','R2_CV'])
pred=pd.DataFrame(predictions,columns=['Material','Modelo','ID','logN_real','logN_predicho','Error_log10N'])
coef=pd.DataFrame(coeff,columns=['Material','N','a','b','R2_ajuste_NO_CV'])

for material, m in metricas.groupby('Material'):
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(m.Modelo,m.MAE_log10N);ax.set_title('MAE validado: '+str(material));ax.set_ylabel('MAE log10(N)');ax.tick_params(axis='x',rotation=25);fig.tight_layout()
    fig.savefig(SALIDA/('MAE_'+str(material).replace('/','_')+'.png'),dpi=180);plt.close(fig)

metodologia=pd.DataFrame([
('Fallas','Se usan roturas explícitas, grietas centrales y fallas Hectors identificadas; grietas requieren confirmar equivalencia con rotura.'),
('Censurados','Se identifican para evaluación posterior, no se entrenan como fallas exactas ni se imputan ciclos de rotura.'),
('Validación','GroupKFold por Grupo si hay >=4 grupos; si no, KFold y limitación documentada.'),
('Comparación','Todos los modelos comparten las mismas particiones dentro de cada material.'),
('Basquin','log10(N)=a+b*log10(sigma_a), esfuerzo en MPa; R2 de ajuste no equivale a R2 validado.'),
('Pendiente crucial','Contrastar descripción de crack, interrupted, finished y no crack found con el libro original FABEST.'),
('Limitación','Los puntos auxiliares no se interpretan como observaciones experimentales.'),
('PCS','Se calculan medidas iniciales de predictibilidad y trazabilidad; estabilidad completa y supervivencia censurada pendientes.')
],columns=['Aspecto','Criterio'])
archivo=SALIDA/'Resultados_Auditados.xlsx'
with pd.ExcelWriter(archivo,engine='openpyxl') as writer:
    for name,df in [('RESUMEN',resumen),('RECUENTO_EVENTOS',recuento),('ELEGIBLES',elegible),('FALLAS',fallas),('CENSURADOS',censurados),('PENDIENTES',pend),('AUXILIARES',aux),('DESCRIPTIVA_TODOS',desc_all),('DESCRIPTIVA_FALLAS',desc_falla),('COEF_BASQUIN',coef),('CV_METRICAS',metricas),('CV_PREDICCIONES',pred),('METODOLOGIA_PCS',metodologia)]:
        df.to_excel(writer,sheet_name=name,index=(name.startswith('DESCRIPTIVA')))
print('\n',resumen.to_string(index=False))
print('\nResultados:',archivo)
print('NOTA: clasificación provisional; no convierte censuras en fallas.')
