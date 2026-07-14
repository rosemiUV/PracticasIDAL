"""
analisis_tiempos.py
Módulo de Análisis Estadístico y Visualización de Tiempos de Intervención.

Este script se encarga de calcular matemáticamente cuánto tiempo ha hablado 
cada político y cada partido político durante la sesión, basándose en las 
marcas de tiempo extraídas por el sistema de diarización.

Sus objetivos principales son:
1. Leer los fragmentos de texto y agregar (sumar) los tiempos de intervención.
2. Calcular porcentajes absolutos (respecto a todo el pleno) y relativos 
   (respecto al tiempo total del propio partido).
3. Preparar las estructuras de datos (DataFrames) precisas que el frontend 
   necesita para dibujar los gráficos de barras y de tarta.
4. Proveer un visualizador local interactivo (Plotly).
"""

import json
import pandas as pd
from pathlib import Path

# --- DICCIONARIO DE COLORES POLÍTICOS ---
MAPA_COLORES = {
    "Sumar": "#E51C55",       
    "PSOE": "red",            
    "Vox": "green",           
    "PP": "#1E90FF",          
    "Mesa": "saddlebrown",    
    "ERC": "#FFD700",         
    "EH Bildu": "turquoise"   
}

def calcular_tiempos_dashboard(ruta_json: str | Path):
    """
    Lee el archivo JSON procesado y realiza las agregaciones de tiempo.
    
    Agrupa los datos matemáticamente para generar dos tablas (DataFrames):
    1. df_tarta: Suma total del tiempo de habla agrupado únicamente por partido político.
    2. df_final (barras): Desglose detallado por ponente, calculando su porcentaje 
       de tiempo tanto respecto al total del vídeo (porcentaje_global) como 
       respecto al tiempo total de su propio partido (porcentaje_relativo).
       
    Retorna ambas estructuras formateadas y ordenadas listas para la API.
    """
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None, None

    registros = []
    for chunk in datos:
        nombre = chunk.get("nombre") or "DESCONOCIDO"
        partido = chunk.get("partido") or "SIN PARTIDO"
        duracion = chunk.get("duracion", chunk.get("fin", 0) - chunk.get("inicio", 0))
        registros.append({"nombre": nombre, "partido": partido, "duracion": duracion})

    if not registros:
        print("El archivo JSON está vacío o no tiene el formato esperado.")
        return None, None

    df = pd.DataFrame(registros)
    
    # 1. Tabla agregada para el gráfico de TARTA (Resumen por partido)
    df_partido = df.groupby('partido')['duracion'].sum().reset_index()
    
    # 2. Tabla desglosada para el gráfico de BARRAS (Ponentes)
    df_persona = df.groupby(['partido', 'nombre'])['duracion'].sum().reset_index()
    
    duracion_total_global = df_persona['duracion'].sum()
    
    # Preparamos las matemáticas para las barras
    df_partido_merge = df_partido.rename(columns={'duracion': 'duracion_total_partido'})
    df_final = pd.merge(df_persona, df_partido_merge, on='partido')
    
    df_final['porcentaje_relativo'] = (df_final['duracion'] / df_final['duracion_total_partido']) * 100
    df_final['porcentaje_global'] = (df_final['duracion'] / duracion_total_global) * 100
    df_final = df_final.sort_values(by=['partido', 'porcentaje_relativo'], ascending=[True, False])

    return df_final, df_partido

def mostrar_dashboard_interactivo(df_barras: pd.DataFrame, df_tarta: pd.DataFrame):
    """
    Construye y renderiza un panel de gráficos interactivo en el navegador local 
    utilizando la librería Plotly. 
    
    Diseña un menú desplegable (dropdown) que permite alternar entre:
    - Vista Global: Gráfico de barras con todos los oradores.
    - Vista Resumen: Gráfico de tarta con los tiempos totales por partido.
    - Vistas Internas: Gráficos de barras filtrados por cada partido para ver el 
      reparto interno de tiempos entre sus diputados.
      
    Nota: Esta función es de uso interno/depuración, el frontend usará los 
    datos puros para dibujar sus propios gráficos en React.
    """
    if df_barras is None or df_barras.empty:
        return

    partidos = list(df_barras['partido'].unique())
    import plotly.graph_objects as go
    fig = go.Figure()
    
    y_globales = []
    textos_globales = []
    y_relativos = []
    textos_relativos = []
    
    # ==============================================================
    # 1. CREAR LAS CAPAS DE BARRAS (Una por partido)
    # ==============================================================
    for partido in partidos:
        df_filtrado = df_barras[df_barras['partido'] == partido]
        
        y_glob = df_filtrado['porcentaje_global'].tolist()
        txt_glob = df_filtrado['porcentaje_global'].apply(lambda x: f'{x:.1f}%').tolist()
        
        y_rel = df_filtrado['porcentaje_relativo'].tolist()
        txt_rel = df_filtrado['porcentaje_relativo'].apply(lambda x: f'{x:.1f}%').tolist()
        
        y_globales.append(y_glob)
        textos_globales.append(txt_glob)
        y_relativos.append(y_rel)
        textos_relativos.append(txt_rel)
        
        if partido == "PNV":
            config_marcador = dict(
                color="green", 
                pattern=dict(shape="/", fgcolor="red", bgcolor="green", fillmode="overlay")
            )
        else:
            color_elegido = MAPA_COLORES.get(partido, "gray")
            config_marcador = dict(color=color_elegido)
            
        fig.add_trace(
            go.Bar(
                x=df_filtrado['nombre'],
                y=y_glob, 
                name=partido,
                text=txt_glob, 
                textposition='auto',
                visible=True,
                marker=config_marcador
            )
        )

    # ==============================================================
    # 2. CREAR LA CAPA DE TARTA (Resumen Total de Partidos)
    # ==============================================================
    colores_tarta = []
    for p in df_tarta['partido']:
        if p == "PNV":
            colores_tarta.append("#228B22") # Verde Bosque sólido para evitar fallos de renderizado en curvas
        else:
            colores_tarta.append(MAPA_COLORES.get(p, "gray"))

    fig.add_trace(
        go.Pie(
            labels=df_tarta['partido'],
            values=df_tarta['duracion'],
            name="Resumen",
            marker=dict(colors=colores_tarta),
            textinfo='label+percent',
            hoverinfo='label+percent+value',
            visible=False # Apagada al iniciar
        )
    )
        
    # ==============================================================
    # 3. CONSTRUIR EL MENÚ DESPLEGABLE (DASHBOARD)
    # ==============================================================
    botones = []
    total_capas = len(partidos) + 1 # Sumamos 1 por la capa de la Tarta
    
    # ---> BOTÓN 1: Barras Globales (Todos los ponentes)
    visibilidad_todos = [True] * len(partidos) + [False] # Enciende barras, apaga tarta
    botones.append(
        dict(
            label="Todos los ponentes (Barras)",
            method="update",
            args=[
                {
                    "visible": visibilidad_todos,
                    "y": y_globales + [None], 
                    "text": textos_globales + [None]
                },
                {
                    "title": "Porcentaje de Intervención Global por Ponente",
                    "yaxis.title.text": "Porcentaje respecto al TOTAL (%)",
                    "xaxis.title.text": "Intervinientes",
                    "xaxis.tickfont.size": 10
                }
            ]
        )
    )

    # ---> BOTÓN 2: Gráfico de Tarta (Resumen Partidos)
    visibilidad_tarta = [False] * len(partidos) + [True] # Apaga barras, enciende tarta
    botones.append(
        dict(
            label="Total Partidos (Tarta)",
            method="update",
            args=[
                {
                    "visible": visibilidad_tarta
                },
                {
                    "title": "Distribución del Tiempo Total de Habla por Partidos",
                    "yaxis.title.text": "", # Ocultamos los títulos de ejes (no aplican en tarta)
                    "xaxis.title.text": ""
                }
            ]
        )
    )
    
    # ---> BOTONES 3...N: Partidos Individuales (Vistas Relativas)
    for i, partido in enumerate(partidos):
        visibilidad_partido = [False] * len(partidos) + [False]
        visibilidad_partido[i] = True
        
        num_ponentes = len(df_barras[df_barras['partido'] == partido]['nombre'].unique())
        tamano_fuente = 20 if num_ponentes == 1 else 14
            
        botones.append(
            dict(
                label=f"Partido: {partido}",
                method="update",
                args=[
                    {
                        "visible": visibilidad_partido,
                        "y": y_relativos + [None],
                        "text": textos_relativos + [None]
                    },
                    {
                        "title": f"Intervención Interna: {partido}",
                        "yaxis.title.text": f"Porcentaje respecto a {partido} (%)",
                        "xaxis.title.text": "Intervinientes",
                        "xaxis.tickfont.size": tamano_fuente
                    }
                ]
            )
        )
        
    # ==============================================================
    # 4. APLICAR DISEÑO Y RENDERIZAR
    # ==============================================================
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=botones,
                direction="down",
                showactive=True,
                x=1.0,             
                xanchor="right",   
                y=1.15,
                yanchor="top",
                font=dict(size=14)
            )
        ],
        title="Porcentaje de Intervención Global por Ponente",
        xaxis_title="Intervinientes",
        yaxis_title="Porcentaje respecto al TOTAL (%)",
        yaxis=dict(range=[0, 105]), 
        xaxis=dict(tickfont=dict(size=10)), 
        plot_bgcolor='rgba(240,240,240,0.5)',
        margin=dict(t=100) 
    )

    fig.show()

if __name__ == "__main__":
    ruta_script = Path(__file__).parent
    ruta_archivo_prueba = ruta_script / "resultados_finales" / "datos_rag_video_hoy_1_identificado.json"
    
    if not ruta_archivo_prueba.exists():
        print(f"ERROR: No encuentro el archivo JSON en:\n{ruta_archivo_prueba}")
    else:
        print(f"Procesando archivo: {ruta_archivo_prueba.name}")
        df_bars, df_pie = calcular_tiempos_dashboard(ruta_archivo_prueba)
        
        if df_bars is not None:
            print("\nAbriendo el Dashboard Interactivo...")
            mostrar_dashboard_interactivo(df_bars, df_pie)