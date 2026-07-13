import json
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

def analizar_conflictividad_sin_mesa(ruta_json: str | Path):
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    df = pd.DataFrame(datos)
    
    # 1. Filtramos partidos definidos y descartamos explícitamente "Mesa" y "SIN PARTIDO"
    df_limpio = df[
        df['partido'].notna() & 
        (df['partido'] != "SIN PARTIDO") & 
        (df['partido'] != "Mesa")
    ].copy()
    
    # 2. Ordenamos temporalmente
    df_limpio = df_limpio.sort_values('inicio')

    # 3. Creamos una lista de sucesiones ignorando a la Mesa
    # Al haber eliminado los registros de "Mesa" antes, el shift(-1) 
    # conectará automáticamente el ponente anterior con el siguiente ponente político[cite: 1]
    df_limpio['siguiente_partido'] = df_limpio['partido'].shift(-1)
    df_limpio = df_limpio.dropna(subset=['siguiente_partido'])

    # 4. Contamos las interacciones[cite: 1]
    matriz = df_limpio.groupby(['partido', 'siguiente_partido']).size().reset_index(name='frecuencia')
    matriz_pivot = matriz.pivot(index='partido', columns='siguiente_partido', values='frecuencia').fillna(0)

    # 5. Ponemos a 0 la diagonal (interacciones del mismo partido)[cite: 1]
    for partido in matriz_pivot.index:
        if partido in matriz_pivot.columns:
            matriz_pivot.loc[partido, partido] = 0

    # 6. Creamos el mapa de calor[cite: 1]
    fig = px.imshow(
        matriz_pivot,
        labels=dict(x="Siguiente Partido", y="Partido Actual", color="Frecuencia"),
        title="Interacciones Parlamentarias (Ignorando a la Mesa)",
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(plot_bgcolor='white')
    fig.show()

if __name__ == "__main__":
    ruta = Path(__file__).parent / "resultados_finales" / "datos_rag_video_hoy_1_identificado.json"
    if ruta.exists():
        analizar_conflictividad_sin_mesa(ruta)