'''
Los endpoints (URLs) de la API
'''

import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from datetime import datetime
from src.api.schemas import SearchRequest, SearchResponse
from src.api.schemas import UrlVideoRequest, ContextRequest, SummaryRequest, EntitiesRequest, StatsRequest
from src.transcriptor_diarizador.pipeline_principalV2 import ejecutar_pipeline_lote
from typing import Dict
from starlette.concurrency import run_in_threadpool

# Diccionario para guardar las conexiones activas de los clientes
active_websockets: Dict[str, WebSocket] = {}

# Instanciamos el enrutador
router = APIRouter()

@router.get('/sessions')
def get_sessions():
    """
    Devuelve la lista de sesiones únicas almacenadas en ChromaDB.
    """
    try:
        from src.transcriptor_diarizador.cargador_chroma import conectar_chromadb, obtener_coleccion
        cliente = conectar_chromadb()
        coleccion = obtener_coleccion(cliente)
        
        resultado = coleccion.get(include=["metadatas"])
        metadatos = resultado.get("metadatas", [])
        
        sesiones_vistas = set()
        sesiones_unicas = []
        
        for m in metadatos:
            vid = m.get("video_id")
            if vid and vid not in sesiones_vistas:
                sesiones_vistas.add(vid)
                # Formateamos la duración a un formato legible si es posible
                dur = "Procesada"
                
                sesiones_unicas.append({
                    "id_sesion": vid,
                    "titulo": m.get("titulo_video", "Sesión Parlamentaria"),
                    "fecha": m.get("fecha_publicacion", "Desconocida"),
                    "duracion": dur
                })
                
        return sesiones_unicas
    except Exception as e:
        print(f"Error al obtener sesiones de ChromaDB: {e}")
        # Si falla, devolvemos un array vacío para no romper el frontend
        return []

@router.post('/search', response_model=SearchResponse)
def perform_search(request: SearchRequest):
    """
    Recibe la pregunta del usuario y busca en ChromaDB y llama a Llama-3.
    """
    from src.motor_busqueda.pipeline_rag import buscar, buscar_transversal
    
    try:
        if request.is_global:
            resultados = buscar_transversal(pregunta=request.pregunta)
        else:
            # Realizamos la búsqueda en ChromaDB y llamamos a Llama-3 local con memoria
            if not request.id_sesion:
                raise HTTPException(status_code=400, detail="id_sesion requerido para búsqueda no global")
            resultados = buscar(pregunta=request.pregunta, video_id=request.id_sesion)
        return resultados
    except Exception as e:
        print(f"Error en la búsqueda RAG: {e}")
        raise HTTPException(status_code=500, detail="Error realizando la búsqueda.")

@router.post('/clear_transversal_history')
def clear_transversal_history():
    from src.motor_busqueda.pipeline_rag import limpiar_historial_transversal
    try:
        limpiar_historial_transversal()
        return {"status": "ok"}
    except Exception as e:
        print(f"Error limpiando historial: {e}")
        raise HTTPException(status_code=500, detail="Error limpiando historial.")


@router.post('/context')
def get_context(request: ContextRequest):
    """
    Recupera el contexto completo (fragmentos adyacentes) de una intervención.
    """
    from src.motor_busqueda.pipeline_rag import obtener_intervencion_completa
    
    try:
        resultado = obtener_intervencion_completa(
            request.video_id, request.ponente, request.inicio, request.fin
        )
        if resultado.get("error"):
            # Devolvemos el error limpiamente sin lanzarlo como excepción para que no lo atrape el except global
            return {"error": resultado["error"]}
            
        return resultado
    except Exception as e:
        print(f"Error al obtener el contexto completo: {e}")
        raise HTTPException(status_code=500, detail="Error obteniendo el contexto.")


@router.websocket('/ws/progress/{client_id}')
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    active_websockets[client_id] = websocket
    try:
        while True:
            # Mantenemos la conexión abierta escuchando, aunque el cliente no envíe nada
            await websocket.receive_text()
    except WebSocketDisconnect:
        if client_id in active_websockets:
            del active_websockets[client_id]


@router.post('/process')
async def process_video(request: UrlVideoRequest):
    try:
        urls_to_process = []
        if request.urls:
            urls_to_process.extend(request.urls)
        elif request.url:
            urls_to_process.append(request.url)
            
        if not urls_to_process:
            raise HTTPException(status_code=400, detail="Se requiere al menos una URL.")
            
        # Función callback que enviará el estado en tiempo real al WebSocket conectado
        loop = asyncio.get_running_loop()
        def callback_progreso(datos):
            if request.client_id and request.client_id in active_websockets:
                ws = active_websockets[request.client_id]
                # Utilizamos asyncio.run_coroutine_threadsafe porque esto se ejecutará en un thread secundario
                asyncio.run_coroutine_threadsafe(ws.send_json(datos), loop)

        # Su función debería encargarse de iterar, subir a ChromaDB y devolver 
        # una lista con el formato exacto que necesita el frontend.
        resultados_exitosos, videos_fallidos = await run_in_threadpool(
            ejecutar_pipeline_lote,
            urls_to_process,
            callback_progreso
        )

        # Mapeamos los resultados exitosos al formato exacto que espera nuestro frontend React
        sesiones_procesadas = []
        for res in resultados_exitosos:
            sesiones_procesadas.append({
                "id_sesion": res["video_id"],
                "titulo": res["titulo"],
                "fecha": datetime.now().strftime("%d/%m/%Y"),
                "duracion": "Procesada"
            })

        return sesiones_procesadas

    except ImportError:
        raise HTTPException(status_code=501, detail="La función de procesamiento en lote aún no está implementada (esperando pull).")
    except Exception as e:
        print(f"Error crítico en el endpoint /api/process: {e}")
        raise HTTPException(status_code=500, detail="Error procesando los vídeos en el backend.")

@router.post('/summary')
def get_summary(request: SummaryRequest):
    """
    Devuelve un resumen global y el índice de temas para la sesión especificada.
    """
    from src.motor_busqueda.pipeline_rag import generar_resumen
    from src.motor_busqueda.db_neo4j import db
    
    try:
        # 1. Intentar cargar desde Neo4j (Instantáneo)
        query = "MATCH (v:Video {id: $video_id}) RETURN v.resumen AS resumen"
        resultados = db.execute_query(query, {"video_id": request.video_id})
        
        if resultados and resultados[0]["resumen"]:
            print(f"Neo4j Hit: Resumen devuelto para {request.video_id}")
            return {"video_id": request.video_id, "resumen": resultados[0]["resumen"], "error": None}

        # 2. Fallback: Vídeo antiguo sin caché. Llamar al LLM al vuelo.
        print(f"Neo4j Miss: Generando resumen para {request.video_id} al vuelo...")
        resultado = generar_resumen(request.video_id)
        if resultado.get("error"):
            return {"error": resultado["error"]}
            
        # Guardar en Neo4j para la próxima vez
        if not resultado.get("error"):
            db.execute_write('''
                MERGE (v:Video {id: $video_id})
                ON CREATE SET v.resumen = $resumen
                ON MATCH SET v.resumen = $resumen
            ''', {"video_id": request.video_id, "resumen": resultado["resumen"]})
            
        return resultado
    except Exception as e:
        print(f"Error al generar resumen: {e}")
        raise HTTPException(status_code=500, detail="Error generando el resumen global.")

@router.post('/entities')
def get_entities(request: EntitiesRequest):
    """
    Devuelve las entidades vinculadas al vídeo (Leyes, Personas, etc.) directamente desde Neo4j.
    """
    from src.motor_busqueda.pipeline_rag import extraer_entidades
    from src.motor_busqueda.db_neo4j import db
    
    try:
        # 1. Intentar cargar entidades desde Neo4j
        if not request.pregunta:
            query = """
            MATCH (v:Video {id: $video_id})-[:MENCIONA]->(n)
            RETURN n, labels(n) AS tipo
            """
            resultados = db.execute_query(query, {"video_id": request.video_id})
            
            if resultados:
                print(f"Neo4j Hit: Entidades devueltas para {request.video_id}")
                entidades_formateadas = []
                for res in resultados:
                    nodo = res["n"]
                    tipo = res["tipo"][0]
                    entidad = {
                        "id": nodo.get("id"),
                        "tipo": tipo.lower(),
                        "nombre": nodo.get("nombre"),
                        "url": nodo.get("url"),
                        "fuente": nodo.get("fuente", "Desconocida")
                    }
                    if tipo == "Persona":
                        entidad["explicacion"] = nodo.get('rol', 'Político.')
                        entidad["foto_url"] = nodo.get("foto_url", f"https://ui-avatars.com/api/?name={nodo.get('nombre', 'X').replace(' ', '+')}&background=random")
                    elif tipo == "Ley":
                        entidad["explicacion"] = nodo.get("resumen", "")
                        entidad["foto_url"] = "https://ui-avatars.com/api/?name=⚖️&background=fff3cd&color=856404"
                    elif tipo == "Partido":
                        entidad["explicacion"] = nodo.get("descripcion", "Partido u Organización.")
                        entidad["tipo"] = "partido"
                        entidad["foto_url"] = nodo.get("logo_url", f"https://ui-avatars.com/api/?name={nodo.get('nombre', 'P').replace(' ', '+')}&background=random")
                    elif tipo == "Lugar":
                        entidad["explicacion"] = nodo.get("descripcion", "")
                        entidad["foto_url"] = "https://ui-avatars.com/api/?name=📍&background=cce5ff&color=004085"
                    elif tipo == "Evento":
                        entidad["explicacion"] = nodo.get("descripcion", "")
                        entidad["tipo"] = "evento"
                        entidad["foto_url"] = "https://ui-avatars.com/api/?name=📅&background=f8d7da&color=721c24"
                    elif tipo == "Concepto":
                        entidad["explicacion"] = nodo.get("descripcion", "")
                        entidad["tipo"] = "concepto"
                        entidad["foto_url"] = "https://ui-avatars.com/api/?name=💡&background=d1ecf1&color=0c5460"
                        
                    entidades_formateadas.append(entidad)
                    
                return {"video_id": request.video_id, "entidades": entidades_formateadas, "error": None}

        # 2. Fallback: Llamar a LLM y Wikipedia
        print(f"Neo4j Miss o Búsqueda Específica: Extrayendo entidades para {request.video_id} al vuelo...")
        return extraer_entidades(request.video_id, request.pregunta, top_k=25)
    except Exception as e:
        print(f"Error extrayendo entidades: {e}")
        raise HTTPException(status_code=500, detail="Error extrayendo las entidades.")

@router.get('/entidad/{entidad_id}')
def get_entidad_por_id(entidad_id: str):
    """
    Busca una entidad en Neo4j por su ID y devuelve sus detalles para mostrar
    en la tarjeta interactiva (HoverCard).
    """
    from src.motor_busqueda.db_neo4j import db
    try:
        # Buscamos en todas las etiquetas posibles
        query = """
        MATCH (n)
        WHERE n.id = $id AND (n:Persona OR n:Partido OR n:Ley OR n:Lugar OR n:Evento OR n:Concepto)
        RETURN n, labels(n) AS tipo
        """
        resultados = db.execute_query(query, {"id": entidad_id})
        
        if not resultados:
            raise HTTPException(status_code=404, detail="Entidad no encontrada en el grafo")
            
        nodo = resultados[0]["n"]
        tipo = resultados[0]["tipo"][0] # ej: 'Persona', 'Partido', 'Ley', 'Lugar', 'Evento', 'Concepto'
        
        respuesta = {
            "id": nodo.get("id"),
            "tipo": tipo,
            "nombre": nodo.get("nombre"),
            "url": nodo.get("url"),
            "fuente": nodo.get("fuente", "Desconocida")
        }
        
        if tipo == "Persona":
            respuesta["rol"] = nodo.get("rol", "Cargo desconocido")
            # Foto provisional
            respuesta["foto_url"] = nodo.get("foto_url", "https://ui-avatars.com/api/?name=" + nodo.get("nombre", "X").replace(" ", "+") + "&background=random")
            respuesta["descripcion"] = f"Político. {nodo.get('rol', '')}"
            
        elif tipo == "Partido":
            respuesta["siglas"] = nodo.get("siglas", "")
            respuesta["descripcion"] = nodo.get("descripcion", "Partido u Organización.")
            respuesta["foto_url"] = nodo.get("logo_url", "https://ui-avatars.com/api/?name=" + nodo.get("siglas", "P").replace(" ", "+") + "&background=random")
            
        elif tipo == "Ley":
            respuesta["descripcion"] = nodo.get("resumen", "Normativa o texto legal.")
            
        elif tipo == "Lugar":
            respuesta["descripcion"] = nodo.get("descripcion", "Lugar geográfico.")
            
        elif tipo == "Evento":
            respuesta["descripcion"] = nodo.get("descripcion", "Evento o suceso clave.")
            
        elif tipo == "Concepto":
            respuesta["descripcion"] = nodo.get("descripcion", "Programa o concepto político.")
            
        return respuesta
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error buscando entidad en Neo4j: {e}")
        raise HTTPException(status_code=500, detail="Error interno de base de datos.")

@router.post('/stats')
def get_stats(request: StatsRequest):
    """
    Devuelve los datos procesados para generar los gráficos de tiempo de habla en el frontend.
    """
    from pathlib import Path
    from src.transcriptor_diarizador.analisis_tiempos import calcular_tiempos_dashboard
    import numpy as np
    
    try:
        # Construimos la ruta al archivo JSON identificado
        ruta_script = Path(__file__).resolve().parent
        ruta_json = ruta_script.parent.parent / "data" / "resultados_finales" / f"datos_rag_{request.video_id}_identificado.json"
        
        # Si no existe el identificado, intentamos con el crudo
        if not ruta_json.exists():
            ruta_json = ruta_script.parent.parent / "data" / "resultados_finales" / f"datos_rag_{request.video_id}.json"
            
        if not ruta_json.exists():
            return {"error": "No se encontraron datos procesados para este vídeo."}
            
        df_barras, df_tarta = calcular_tiempos_dashboard(ruta_json)
        
        if df_barras is None or df_tarta is None:
            return {"error": "No se pudo procesar la información de tiempos."}
            
        # Reemplazar NaN e Infinity por None antes de convertir a dict
        df_barras = df_barras.replace([np.inf, -np.inf], np.nan).fillna(0)
        df_tarta = df_tarta.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Convertimos los DataFrames a listas de diccionarios
        datos_barras = df_barras.to_dict(orient='records')
        datos_tarta = df_tarta.to_dict(orient='records')
        
        return {
            "video_id": request.video_id,
            "barras": datos_barras,
            "tarta": datos_tarta
        }
    except Exception as e:
        print(f"Error generando estadísticas de tiempo: {e}")
        raise HTTPException(status_code=500, detail="Error generando estadísticas.")