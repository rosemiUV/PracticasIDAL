"""
pipeline_principalV2.py
Orquestador principal del sistema "Buscador Plenario Inteligente" (Pipeline de Ingesta).

Este script actúa como el director de orquesta que coordina todas las fases del 
procesamiento de un vídeo parlamentario, desde que se introduce la URL hasta que 
los datos están listos para el chatbot.
"""
import json
import warnings
import hashlib
from pathlib import Path

warnings.filterwarnings("ignore")

# Rutas absolutas corregidas para que funcione perfectamente con 'python -m'
from src.transcriptor_diarizador.procesador import configurar_ffmpeg_local, descargar_audio_youtube, transcribir_y_diarizar
from src.transcriptor_diarizador.fusionador_pruebaV2 import fusionar_datos_para_rag
from src.transcriptor_diarizador.identificador_comisiones import identificar_video_auto
from src.transcriptor_diarizador.cargador_chroma import subir_datos_a_chroma
from src.transcriptor_diarizador.extractor_votaciones import procesar_archivo, guardar_resultados

configurar_ffmpeg_local()


def obtener_metadatos_youtube(url: str):
    """
    Extrae el título y la fecha de publicación de un vídeo de YouTube utilizando yt-dlp,
    sin necesidad de descargar el archivo multimedia. Utiliza cookies locales si están 
    disponibles para evitar bloqueos por parte de YouTube.
    Devuelve el título y la fecha formateada (YYYY-MM-DD).
    """
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "skip_download": True}
        
        # Buscar cookies.txt en la raíz del proyecto
        ruta_cookies = Path(__file__).parent.parent.parent / "cookies.txt"
        if ruta_cookies.exists():
            ydl_opts["cookiefile"] = str(ruta_cookies)
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            titulo = info.get("title", "Sesión Parlamentaria")
            fecha = info.get("upload_date", "19700101")
            fecha_formateada = f"{fecha[:4]}-{fecha[4:6]}-{fecha[6:]}"
            return titulo, fecha_formateada
    except Exception as e:
        print(f"No se pudo obtener metadatos: {e}")
        return "Sesión Parlamentaria", "1970-01-01"


def ejecutar_pipeline_completo(url_video: str, callback_progreso=None):
    """
    Orquesta el ciclo de vida completo de procesamiento de un único vídeo:
    1. Descarga el audio.
    2 y 3. Transcribe y diariza con WhisperX/PyAnnote.
    4. Trocea el texto (Chunking) y asigna speakers (Fase 4b).
    4c. Extrae automáticamente las votaciones detectadas.
    5. Sube los fragmentos vectorizados a la base de datos ChromaDB.
    6. Genera resúmenes globales y entidades utilizando LLMs.

    Si el vídeo ya ha sido procesado previamente, optimiza el flujo saltando a la Fase 6.
    Emite actualizaciones de estado en tiempo real a través de callback_progreso.
    """
    video_id = "video_" + hashlib.sha1(url_video.encode("utf-8")).hexdigest()[:8]
    titulo_real, fecha_publicacion = obtener_metadatos_youtube(url_video)

    print(f"\n{'='*45}")
    print(f"INICIANDO PIPELINE PARA: {video_id}")
    print(f"TÍTULO:  {titulo_real}")
    print(f"FECHA:   {fecha_publicacion}")
    print(f"{'='*45}")

    # 1. AVISO: Inicio (10%)
    if callback_progreso:
        callback_progreso({"video_id": video_id, "progreso": 5, "estado": "Descargando audio de YouTube..."})

    ruta_script = Path(__file__).parent
    dir_data_root = ruta_script.parent.parent / "data"
    dir_audio = dir_data_root / "audios"
    dir_final = dir_data_root / "resultados_finales"
    
    ruta_json_identificado = dir_final / f"datos_rag_{video_id}_identificado.json"
    
    # === COMPROBACIÓN DE VÍDEO YA PROCESADO ===
    if ruta_json_identificado.exists():
        print(f"\n[!] El vídeo {video_id} ya fue transcrito y procesado previamente. Saltando a la Fase 6...")
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 98, "estado": "El vídeo ya existe, forzando Fase 6 (resumen y entidades)..."})
        
        # Ejecutar sólo la Fase 6 para guardar en BD
        try:
            from src.motor_busqueda.pipeline_rag import generar_resumen, extraer_entidades
            from src.api.database import guardar_metadatos_video
            
            print("Extrayendo entidades...")
            res_entidades = extraer_entidades(video_id)
            print("Generando resumen...")
            res_resumen = generar_resumen(video_id)
            
            guardar_metadatos_video(
                video_id=video_id,
                resumen=res_resumen.get("resumen", ""),
                entidades=res_entidades.get("entidades", [])
            )
            print("Metadatos globales guardados en SQLite/Neo4j.")
        except Exception as e:
            print(f"Error calculando metadatos globales en Fase 6: {e}")

        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 100, "estado": "¡Procesamiento completado con éxito!"})
            
        print("\nPIPELINE COMPLETADO AL 100%.")
        return video_id, str(ruta_json_identificado), titulo_real
    # ==========================================

    # --- FASE 1: DESCARGA ---
    audio_temporal = descargar_audio_youtube(url_video, dir_audio)
    if not audio_temporal or not audio_temporal.exists():
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": -1, "estado": "Error crítico en la descarga del audio"})
        raise Exception("Falló la descarga del audio.")

    audio_path = dir_audio / f"{video_id}.wav"
    audio_temporal.replace(audio_path)

    # 2. AVISO: Transcripción (10%) - Dejamos margen hasta el 90% para la IA
    if callback_progreso:
        callback_progreso({"video_id": video_id, "progreso": 10, "estado": "Transcribiendo y separando voces (Esto tardará un poco)..."})

    try:
        # --- FASE 2+3: TRANSCRIPCIÓN + DIARIZACIÓN INTEGRADA ---
        print("\n--- FASE 2+3: TRANSCRIPCIÓN Y DIARIZACIÓN (WhisperX + PyAnnote) ---")
        segmentos = transcribir_y_diarizar(audio_path)

        # Guardar segmentos crudos (útil para depuración)
        dir_diar = dir_data_root / "resultados_diarizacion"
        dir_diar.mkdir(parents=True, exist_ok=True)
        ruta_segmentos_crudos = dir_diar / f"diarizacion_{video_id}.json"
        with open(ruta_segmentos_crudos, "w", encoding="utf-8") as f:
            json.dump(segmentos, f, indent=2, ensure_ascii=False)
        print(f"Segmentos crudos guardados en: {ruta_segmentos_crudos}")

        # 3. AVISO: Procesamiento de datos (90%)
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 90, "estado": "Generando fragmentos inteligentes para el buscador RAG..."})

        # --- FASE 4: CHUNKING + METADATOS ---
        print("\n--- FASE 4: GENERANDO CHUNKS PARA RAG ---")
        dir_final = dir_data_root / "resultados_finales"
        ruta_json_final = dir_final / f"datos_rag_{video_id}.json"

        chunks = fusionar_datos_para_rag(
            segmentos=segmentos,
            video_id=video_id,
            url_video=url_video,
            titulo_video=titulo_real,
            fecha_publicacion=fecha_publicacion,
            ruta_guardado=ruta_json_final,
            max_palabras=170,
        )
        print(f"Chunks generados: {len(chunks)} — guardados en: {ruta_json_final}")

        # 4b. AVISO: Identificación de speakers (92%)
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 92, "estado": "Identificando a los oradores (nombre y partido)..."})

        # --- FASE 4b: IDENTIFICACIÓN DE SPEAKERS ---
        print("\n--- FASE 4b: IDENTIFICACIÓN DE SPEAKERS ---")
        ruta_json_identificado = identificar_video_auto(
            ruta_json_entrada=ruta_json_final,
            usar_llm=True,
            )
        print(f"Speakers identificados — JSON final: {ruta_json_identificado}")

        # El JSON identificado es el que se sube a ChromaDB de aquí en adelante
        ruta_json_final = ruta_json_identificado

        # =================================================================
        # --- NUEVA FASE 4c: EXTRACCIÓN DE VOTACIONES ---
        # =================================================================
        print("\n--- FASE 4c: EXTRACCIÓN DE VOTACIONES ---")
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 93, "estado": "Buscando y extrayendo votaciones del pleno..."})

        try:
            # 1. Leemos el JSON que acabamos de generar y extraer votaciones
            votaciones = procesar_archivo(ruta_json_final)
            
            if votaciones:
                # 2. Creamos la subcarpeta 'votaciones' dentro de 'data' si no existe
                dir_votaciones = dir_data_root / "votaciones"
                dir_votaciones.mkdir(parents=True, exist_ok=True)
                
                # 3. Guardamos el resultado dentro de esa nueva carpeta
                ruta_salida_votaciones = dir_votaciones / f"votaciones_extraidas_{video_id}.json"
                guardar_resultados(votaciones, ruta_salida_votaciones)
                print(f"[+] ¡Éxito! {len(votaciones)} votaciones guardadas en: {ruta_salida_votaciones}")
            else:
                print("[-] No se detectaron votaciones en este vídeo.")
        except Exception as e:
            print(f"[!] Error extrayendo votaciones: {e}")
        # =================================================================

        # 4. AVISO: Base de datos (95%)
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 95, "estado": "Guardando información procesada en ChromaDB..."})

        # --- FASE 5: SUBIDA A CHROMADB ---
        print("\n--- FASE 5: SUBIDA A CHROMA DB ---")
        if ruta_json_final.exists():
            subir_datos_a_chroma(ruta_json_final)
        else:
            print("Error: no se encontró el JSON final.")

        # --- FASE 6: CÁLCULO DE METADATOS GLOBALES CON IA ---
        print("\n--- FASE 6: CÁLCULO DE METADATOS GLOBALES CON IA ---")
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 98, "estado": "Generando resumen global y entidades con IA..."})

        try:
            from src.motor_busqueda.pipeline_rag import generar_resumen, extraer_entidades
            from src.api.database import guardar_metadatos_video
            
            print("Extrayendo entidades...")
            res_entidades = extraer_entidades(video_id)
            print("Generando resumen...")
            res_resumen = generar_resumen(video_id)
            
            guardar_metadatos_video(
                video_id=video_id,
                resumen=res_resumen.get("resumen", ""),
                entidades=res_entidades.get("entidades", [])
            )
            print("Metadatos globales guardados en SQLite.")
        except Exception as e:
            print(f"Error calculando metadatos globales en Fase 6: {e}")

        # 5. AVISO: Completado (100%)
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": 100, "estado": "¡Procesamiento completado con éxito!"})

    except Exception as e:
        if callback_progreso:
            callback_progreso({"video_id": video_id, "progreso": -1, "estado": f"Error durante el procesamiento: {e}"})
        raise Exception(f"Falló el pipeline: {e}")

    print("\nPIPELINE COMPLETADO AL 100%.")
    return video_id, str(ruta_json_final), titulo_real


def ejecutar_pipeline_lote(lista_urls: list, callback_progreso=None):
    """
    Procesa una lista de URLs de YouTube de forma secuencial, aplicando el pipeline completo
    a cada una. Actúa como un gestor tolerante a fallos: si un vídeo falla, captura el error, 
    lo registra y continúa con el siguiente sin detener la ejecución global.
    Devuelve una tupla con los vídeos procesados exitosamente y los que fallaron.
    """
    print(f"\nINICIANDO PROCESAMIENTO EN LOTE DE {len(lista_urls)} VÍDEOS")

    resultados_exitosos = []
    videos_fallidos = []

    # --- NUESTRA FUNCIÓN ESPÍA PARA VER LOS LOGS EN LA TERMINAL ---
    def mi_impresora_de_progreso(datos):
        print(f"  ➜ [ESTADO WEB] Vídeo: {datos['video_id']} | Progreso: {datos['progreso']}% | {datos['estado']}")
        if callback_progreso:
            callback_progreso(datos)
    # --------------------------------------------------------------

    for indice, url in enumerate(lista_urls, start=1):
        
        try:
            # Pasamos la función espía para que nos imprima en vivo
            video_id, ruta_json, titulo = ejecutar_pipeline_completo(url, callback_progreso=mi_impresora_de_progreso)
            
            resultados_exitosos.append({"url": url, "video_id": video_id, "titulo": titulo, "ruta_json": ruta_json})
            print(f"[VÍDEO {indice}/{len(lista_urls)}] Completado.")
            
        except Exception as e:
            print(f"[VÍDEO {indice}/{len(lista_urls)}] ERROR: {e}. Saltando al siguiente.")
            videos_fallidos.append({"url": url, "error": str(e)})

    print(f"\n{'='*45}")
    print(f"RESUMEN FINAL")
    print(f"  Completados: {len(resultados_exitosos)}")
    print(f"  Fallidos:    {len(videos_fallidos)}")
    for fallo in videos_fallidos:
        print(f"  - {fallo['url']} -> {fallo['error']}")
    print(f"{'='*45}")

    return resultados_exitosos, videos_fallidos


if __name__ == "__main__":
    mis_videos = [
        "https://www.youtube.com/watch?v=nb_elRRfpM4"
    ]
    ejecutar_pipeline_lote(mis_videos)