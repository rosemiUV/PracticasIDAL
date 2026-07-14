"""
cargador_chroma.py
Módulo de Gestión e Ingesta en la Base de Datos Vectorial.

Este script se encarga de coger el archivo JSON final (que contiene todos los 
fragmentos de texto, tiempos y nombres de oradores ya identificados) y subirlo 
a ChromaDB. 

Sus funciones principales son:
1. Gestionar la conexión con la base de datos (tanto en local como en la nube).
2. Actuar como "cortafuegos": verifica si un vídeo ya ha sido subido anteriormente 
   para evitar procesos duplicados que gasten recursos y ensucien las búsquedas.
3. Estructurar e insertar cada fragmento en la base de datos junto a todos sus 
   metadatos (nombre, partido, URL al segundo exacto, etc.), dejándolos listos 
   para las búsquedas semánticas del usuario.
"""
import json
import chromadb
from pathlib import Path

# =============================================================
# CONFIGURACIÓN DE LA BASE DE DATOS
# =============================================================
MODO_LOCAL   = True
CHROMA_HOST  = "chromadb-production-8466.up.railway.app"
CHROMA_PORT  = 443
NOMBRE_COLECCION = "plenario"
# =============================================================

def cargar_json(ruta: Path) -> list:
    """
    Abre y lee el archivo JSON especificado, cargando todos los fragmentos (chunks) 
    procesados en la memoria de Python para su posterior inserción.
    """
    print(f"Leyendo JSON desde: {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    print(f"  {len(datos)} fragmentos cargados.")
    return datos

def conectar_chromadb():
    """
    Establece la conexión con la base de datos vectorial ChromaDB.
    Dependiendo de la variable 'MODO_LOCAL', se conecta a una carpeta local o 
    a un servidor en la nube (ej. Railway) para entornos de producción.
    """
    if MODO_LOCAL:
        print("Conectando a ChromaDB local...")
        return chromadb.PersistentClient(path="./chroma_db_local")
    else:
        print(f"Conectando a ChromaDB servidor: {CHROMA_HOST}:{CHROMA_PORT}")
        return chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=True
        )

def obtener_coleccion(cliente: chromadb.Client):
    """
    Recupera la colección principal (tabla) donde se guardan los fragmentos. 
    Si la colección no existe (por ejemplo, en la primera ejecución), la crea automáticamente.
    """
    coleccion = cliente.get_or_create_collection(name=NOMBRE_COLECCION)
    print(f"Colección '{NOMBRE_COLECCION}' lista. Documentos actuales en total: {coleccion.count()}")
    return coleccion

# --- NUEVA FUNCIÓN DE FILTRADO ---
def video_ya_procesado(coleccion: chromadb.Collection, video_id: str) -> bool:
    """
    Actúa como sistema de seguridad (cortafuegos). Consulta a la base de datos si ya 
    existe al menos un fragmento asociado a ese 'video_id'. Esto evita subir el mismo 
    pleno dos veces si se vuelve a procesar por error.
    """
    try:
        resultado = coleccion.get(
            where={"video_id": video_id},
            limit=1, # Solo necesitamos encontrar uno para saber que existe
            include=["metadatas"]
        )
        return len(resultado["ids"]) > 0
    except Exception as e:
        print(f"Error al comprobar la existencia del vídeo: {e}")
        return False

def insertar_fragmentos(coleccion, fragmentos: list) -> tuple[int, int]:
    """
    Recorre la lista de fragmentos y los inserta uno a uno en ChromaDB.
    Asocia el texto principal del fragmento con una rica lista de metadatos 
    (orador, partido, minuto y segundo, título, etc.) que permitirán filtrar 
    y mostrar la información correctamente en el frontend.
    Devuelve el número de fragmentos nuevos insertados y cuántos fueron ignorados por duplicados.
    """
    insertados = 0
    saltados = 0

    ids_existentes = set(coleccion.get(include=[])["ids"])

    for fragmento in fragmentos:
        frag_id = fragmento.get("chunk_id", f"{fragmento['video_id']}_{fragmento['inicio']}")

        if frag_id in ids_existentes:
            saltados += 1
            continue

        try:
            coleccion.add(
                ids=[frag_id],
                documents=[fragmento["texto"]],
                metadatas=[{
                    "video_id": fragmento["video_id"],
                    "titulo_video": fragmento.get("titulo_video", "Desconocido"),
                    "fecha_publicacion": fragmento.get("fecha_publicacion", "Desconocida"),
                    "url_video": fragmento["url_video"],
                    "url_exacta_tiempo": fragmento.get("url_exacta_tiempo", ""),
                    "ponente":   fragmento["ponente"],
                    "inicio":    fragmento["inicio"],
                    "fin":       fragmento["fin"],
                    "duracion":  fragmento["duracion"],
                    # Campos del identificador de speakers (v2.7)
                    "nombre":       fragmento.get("nombre") or "",
                    "partido":      fragmento.get("partido") or "",
                    "estado_id":    fragmento.get("estado_id") or "",
                    "confianza_id": fragmento.get("confianza_id", 0.0),
                    "metodo_id":    fragmento.get("metodo_id") or "",
                }]
            )
            insertados += 1
        except Exception as e:
            print(f"  [ERROR] Fallo al insertar {frag_id}: {e}")

    return insertados, saltados

def subir_datos_a_chroma(ruta_json: Path):
    """
    Función orquestadora principal que se llama desde el pipeline.
    Carga el JSON, conecta a la base de datos, verifica que el vídeo no exista ya 
    para no duplicar datos, y lanza el proceso masivo de inserción de fragmentos,
    mostrando un resumen estadístico al finalizar.
    """
    """Función principal para ser llamada desde el pipeline"""
    fragmentos = cargar_json(ruta_json)
    
    if not fragmentos:
        print("Error: El JSON está vacío.")
        return

    # Extraemos el ID del vídeo del primer fragmento
    video_id = fragmentos[0].get("video_id")
    
    cliente    = conectar_chromadb()
    coleccion  = obtener_coleccion(cliente)

    # --- NUEVO CORTAFUEGOS ---
    if video_id and video_ya_procesado(coleccion, video_id):
        print(f"\nEl vídeo '{video_id}' ya se encuentra en la Base de Datos.")
        print("Abortando subida para evitar duplicados y ahorrar tiempo.")
        return
    
    print(f"\nSubiendo nuevos fragmentos del vídeo '{video_id}'...")
    insertados, saltados = insertar_fragmentos(coleccion, fragmentos)
    
    print("\n--- RESUMEN CHROMA DB ---")
    print(f"  Nuevos insertados:  {insertados}")
    print(f"  Saltados (ya existían): {saltados}")
    print(f"  Total en BD global: {coleccion.count()}")