import os
import re
import json
import urllib.request
import urllib.parse
import chromadb
from chromadb.utils import embedding_functions
from mistralai.client import Mistral
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONECTAR A CHROMADB
# ─────────────────────────────────────────────────────────────

NOMBRE_COLECCION = "plenario"
MODO_LOCAL = True

ef = embedding_functions.DefaultEmbeddingFunction()

if MODO_LOCAL:
    print("Conectando a ChromaDB local para RAG...")
    client = chromadb.PersistentClient(path="./chroma_db_local")
else:
    CHROMA_HOST = "chromadb-production-8466.up.railway.app"
    CHROMA_PORT = 443
    print(f"Conectado a ChromaDB en {CHROMA_HOST}:{CHROMA_PORT}")
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        ssl=True
    )

collection = client.get_collection(
    name=NOMBRE_COLECCION,
    embedding_function=ef
)

print(f"Coleccion: '{NOMBRE_COLECCION}' — {collection.count()} fragmentos")


# ─────────────────────────────────────────────────────────────
# MEMORIA DEL CHATBOT
# Se guarda en un diccionario: { video_id: [lista de mensajes] }
# ─────────────────────────────────────────────────────────────

# Número máximo de TURNOS (pregunta+respuesta) que se guardan por vídeo.
# Cada turno son 2 mensajes (user + assistant), así que 3 turnos = 6 mensajes.
# Esto evita que la conversación crezca sin límite y se vuelva lenta/cara
# de mandar al LLM cuando alguien hace muchas preguntas seguidas.
MAX_TURNOS_HISTORIAL = 3

_historial: dict[str, list[dict]] = {}

def obtener_historial(video_id: str) -> list[dict]:
    """Devuelve el historial de conversación de un vídeo. Lista vacía si no hay."""
    return _historial.get(video_id, [])

def limpiar_historial(video_id: str) -> None:
    """Borra el historial de un vídeo (para empezar conversación nueva)."""
    if video_id in _historial:
        del _historial[video_id]

def _recortar_historial(video_id: str) -> None:
    """
    Si el historial de un vídeo supera MAX_TURNOS_HISTORIAL turnos,
    borra los mensajes más antiguos (se queda solo con los últimos).
    """
    limite_mensajes = MAX_TURNOS_HISTORIAL * 2
    if video_id in _historial and len(_historial[video_id]) > limite_mensajes:
        _historial[video_id] = _historial[video_id][-limite_mensajes:]


# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────

def _segundos_a_mmss(segundos: float) -> str:
    """Convierte 125.4 → '02:05'"""
    s = int(segundos)
    return f"{s // 60:02d}:{s % 60:02d}"


def _nombre_mostrar(meta: dict) -> str:
    """
    Devuelve el nombre real del ponente si el sistema lo ha identificado
    (campo 'nombre' del JSON, ej: 'Francina Armengol').

    Si no hay nombre real (campo 'nombre' vacío o null, algo que pasa cuando
    'estado_id' es "DESCONOCIDO" o "AMBIGUO"), cae de vuelta al identificador
    técnico del speaker (campo 'ponente', ej: 'SPEAKER_14') para no dejarlo
    en blanco.

    Opcionalmente añade el partido entre paréntesis si se conoce.
    """
    nombre = meta.get("nombre")
    if not nombre:
        nombre = meta.get("ponente", "Desconocido")

    # Intentar resolver el nombre oficial usando CongresoAPI
    if nombre and not nombre.startswith("SPEAKER_"):
        from src.motor_busqueda.congreso_api import congreso_api
        diputado = congreso_api.buscar_diputado(nombre)
        if diputado:
            nombre = diputado["nombre"]

    partido = meta.get("partido")
    if partido:
        return f"{nombre} ({partido})"
    return nombre


def _fusionar_chunks_contiguos(documentos: list, metadatos: list) -> tuple[list, list]:
    """
    Une fragmentos adyacentes del mismo ponente para darle al LLM
    párrafos más largos y continuos, evitando la fragmentación del contexto.
    """
    if not documentos:
        return [], []

    # Es vital ordenar los fragmentos por tiempo de inicio antes de fusionarlos
    pares = sorted(zip(metadatos, documentos), key=lambda x: x[0].get("inicio", 0))
    
    docs_fusionados = []
    metas_fusionados = []
    
    meta_actual = dict(pares[0][0])
    doc_actual = pares[0][1]
    
    # Tolerancia en segundos para considerar que dos chunks están pegados
    TOLERANCIA_SEGUNDOS = 5.0 
    
    for meta_sig, doc_sig in pares[1:]:
        mismo_ponente = meta_actual.get("ponente") == meta_sig.get("ponente")
        
        # Consideramos si el vídeo origen es el mismo (importante para transversal)
        mismo_video = meta_actual.get("video_id") == meta_sig.get("video_id")
        
        tiempo_cercano = meta_sig.get("inicio", 0) - meta_actual.get("fin", 0) <= TOLERANCIA_SEGUNDOS
        
        if mismo_ponente and mismo_video and tiempo_cercano:
            doc_actual += " " + doc_sig
            meta_actual["fin"] = max(meta_actual["fin"], meta_sig["fin"])
        else:
            docs_fusionados.append(doc_actual)
            metas_fusionados.append(meta_actual)
            meta_actual = dict(meta_sig)
            doc_actual = doc_sig
            
    docs_fusionados.append(doc_actual)
    metas_fusionados.append(meta_actual)
    
    return docs_fusionados, metas_fusionados


def _construir_contexto(documentos: list, metadatos: list) -> str:
    """
    Construye el bloque de contexto que se mete al LLM.
    Incluye el NOMBRE REAL del ponente (si se conoce), tiempo y texto de cada fragmento.
    """
    lineas = []
    for doc, meta in zip(documentos, metadatos):
        t_inicio = _segundos_a_mmss(meta["inicio"])
        t_fin    = _segundos_a_mmss(meta["fin"])
        nombre   = _nombre_mostrar(meta)
        lineas.append(f"{nombre} ({t_inicio} - {t_fin}): [{doc}]")
    return "\n".join(lineas)


def _muestrear_video_completo(video_id: str, n_fragmentos: int = 40, excluir_mesa: bool = True) -> tuple[list, list]:
    """
    Coge fragmentos REPARTIDOS por todo el vídeo (principio, medio y final),
    en vez de solo los primeros n_fragmentos por tiempo.

    Esto es clave para que los "temas" generados reflejen todo el pleno,
    no solo los primeros minutos.

    Devuelve (documentos, metadatos) ya ordenados por tiempo.
    """
    if excluir_mesa:
        where_filtro = {
            "$and": [
                {"video_id": {"$eq": video_id}},
                {"partido": {"$ne": "Mesa"}}
            ]
        }
    else:
        where_filtro = {"video_id": {"$eq": video_id}}

    resultados = collection.get(
        where=where_filtro,
        include=["documents", "metadatas"]
    )

    documentos = resultados.get("documents", [])
    metadatos  = resultados.get("metadatas", [])

    if not documentos:
        return [], []

    # Ordenar todo el vídeo por tiempo de inicio
    pares = sorted(zip(metadatos, documentos), key=lambda x: x[0].get("inicio", 0))

    total = len(pares)

    # Si hay menos fragmentos que los que pedimos, los devolvemos todos
    if total <= n_fragmentos:
        seleccionados = pares
    else:
        # Elegimos índices repartidos uniformemente a lo largo de TODO el vídeo
        # (ej: si hay 400 fragmentos y queremos 40, cogemos 1 de cada 10)
        paso = total / n_fragmentos
        indices = [int(i * paso) for i in range(n_fragmentos)]
        seleccionados = [pares[i] for i in indices]

    metadatos_sel  = [m for m, _ in seleccionados]
    documentos_sel = [d for _, d in seleccionados]
    return documentos_sel, metadatos_sel


def _llamar_mistral(system: str, messages: list[dict]) -> str:
    """
    Llama a Mistral en la nube.
    Se usa en: buscar() y generar_resumen()
    Requiere MISTRAL_API_KEY en el archivo .env
    """
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró MISTRAL_API_KEY. "
            "Asegúrate de tenerla en el archivo .env"
        )
    cliente = Mistral(api_key=api_key)
    respuesta = cliente.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.3
    )
    return respuesta.choices[0].message.content




def _llamar_groq(system: str, messages: list[dict]) -> str:
    """
    Llama a Llama en la nube con Groq.
    Se usa SOLO en extraer_entidades(), porque hace muchas llamadas seguidas
    y Groq tiene un limite gratuito mas generoso que Mistral (30 rpm vs 2 rpm).
    Requiere GROQ_API_KEY en el archivo .env

    Modelo: meta-llama/llama-4-scout-17b-16e-instruct
    → 30K tokens/minuto (5x más que llama-3.1-8b-instant) con mismo RPM y 500K/día
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontro GROQ_API_KEY. "
            "Asegurate de tenerla en el archivo .env"
        )
    cliente = Groq(api_key=api_key)
    respuesta = cliente.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.3
    )
    return respuesta.choices[0].message.content

# ─────────────────────────────────────────────────────────────
# FILTRO DE RUIDO — fragmentos de presentación de turno (Mesa)
# ─────────────────────────────────────────────────────────────
# Frases como "Gracias, señor presidente" o "tiene la palabra el señor..."
# no aportan contenido real al debate. Este filtro las detecta para que
# no ocupen sitio en el contexto que le mandamos al LLM en las búsquedas
# específicas (no se aplica al muestreo global, que ya reparte por tiempo).

MIN_PALABRAS_CHUNK = 8

_PATRONES_RUIDO = [
    r"^(muchas?\s+)?gracias[,.]?\s+(se[ñn]or[a]?\s+\w+|presidente[a]?)[,.]?\s+(por el grupo|tiene la palabra|turno|el siguiente)",
    r"^(muchas?\s+)?gracias[,.]?\s*$",
    r"^(buenas?\s+d[íi]as?|buenas?\s+tardes?)[,.]?\s*(se[ñn]or[íi]as?)?[,.]?\s*$",
    r"^(en turno de|pasamos (ahora )?al punto|continuamos (ahora )?con)",
    r"^por el grupo parlamentario .{0,60}tiene la palabra",
    r"^(muchas?\s+)?gracias[,.]?\s+(se[ñn]or[a]?\s+\w+\.?\s*)?(por el grupo|tiene la palabra|te har[aá] la palabra|har[aá] uso de la palabra)",
    r"^un momento[,.]?\s*(se[ñn]or[a]?\s+\w+)?[,.]?\s*(por favor[,.]?\s*)?(silencio|espere)",
    r"^se suspende la sesi[óo]n|^se reanuda la sesi[óo]n|^se levanta la sesi[óo]n",
]
_REGEX_RUIDO = [re.compile(p, re.IGNORECASE) for p in _PATRONES_RUIDO]


def _es_ruido(texto: str) -> bool:
    """Detecta si un fragmento es 'relleno' (saludos, cesión de turno...) sin contenido útil."""
    texto = (texto or "").strip()
    if len(texto.split()) < MIN_PALABRAS_CHUNK:
        return True
    return any(regex.search(texto) for regex in _REGEX_RUIDO)


def _filtrar_ruido(documentos: list, metadatos: list) -> tuple[list, list]:
    """Quita de la lista los fragmentos detectados como ruido."""
    pares_utiles = [
        (doc, meta) for doc, meta in zip(documentos, metadatos)
        if not _es_ruido(doc)
    ]
    if not pares_utiles:
        # Si por lo que sea TODO se marcó como ruido, mejor no perder nada
        return documentos, metadatos
    docs_filtrados  = [d for d, _ in pares_utiles]
    metas_filtrados = [m for _, m in pares_utiles]
    return docs_filtrados, metas_filtrados


# ─────────────────────────────────────────────────────────────
# DETECTOR DE PREGUNTAS SOBRE VOTACIONES
# ─────────────────────────────────────────────────────────────
# Las preguntas sobre votaciones ("¿qué se votó?", "¿cuántos votos a favor?")
# suelen fallar en la búsqueda semántica normal, porque las frases donde se
# anuncia un resultado de votación son muy formulaicas y no se parecen mucho,
# en significado, a la pregunta del usuario. Para compensar, además de la
# búsqueda semántica normal, forzamos la inclusión de cualquier fragmento del
# vídeo que contenga un patrón típico de resultado de votación.

_PALABRAS_VOTACION = re.compile(
    r"\b(votaci[oó]n(es)?|resultado(s)?|votos?|aprobad|rechazad|abstenci[oó]n|"
    r"qu[eé] se vot[oó]|c[oó]mo vot[oó]|cu[aá]ntos votos|qu[eé] aprob|qu[eé] rechaz)\b",
    re.IGNORECASE
)

_PAT_RESULTADO_VOTACION = re.compile(
    r"(votos? (a favor|emitidos|presentes)\s+\d|"
    r"\d+\s*(votos?\s*)?(a favor|en contra|abstenciones?)|"
    r"en consecuencia[,]?\s+(queda|no se aprueba|se aprueba)|"
    r"votamos (ahora|en primer lugar|el punto n[uú]mero))",
    re.IGNORECASE
)


def _es_pregunta_votacion(pregunta: str) -> bool:
    """Detecta si la pregunta trata sobre el resultado de una votación."""
    return bool(_PALABRAS_VOTACION.search(pregunta))


def _chunks_con_resultados_votacion(video_id: str) -> tuple[list, list]:
    """
    Recorre TODOS los fragmentos del vídeo (no solo el top_k semántico) y
    devuelve aquellos cuyo texto contiene un resultado de votación explícito
    (ej. '180 votos a favor', 'en consecuencia, queda aprobado').
    """
    resultados = collection.get(
        where={"video_id": {"$eq": video_id}},
        include=["documents", "metadatas"]
    )
    documentos = resultados.get("documents", [])
    metadatos  = resultados.get("metadatas", [])

    pares_forzados = [
        (doc, meta) for doc, meta in zip(documentos, metadatos)
        if _PAT_RESULTADO_VOTACION.search(doc or "")
    ]
    docs_forzados  = [d for d, _ in pares_forzados]
    metas_forzados = [m for _, m in pares_forzados]
    return docs_forzados, metas_forzados


def _combinar_sin_duplicados(
    documentos_a: list, metadatos_a: list,
    documentos_b: list, metadatos_b: list
) -> tuple[list, list]:
    """
    Junta dos listas de fragmentos (documentos_a primero, luego documentos_b)
    sin repetir el mismo fragmento dos veces. Se identifica cada fragmento
    por su texto + tiempo de inicio.
    """
    vistos = set()
    docs_combinados, metas_combinados = [], []
    for doc, meta in list(zip(documentos_a, metadatos_a)) + list(zip(documentos_b, metadatos_b)):
        clave = (doc, meta.get("inicio"))
        if clave not in vistos:
            vistos.add(clave)
            docs_combinados.append(doc)
            metas_combinados.append(meta)
    return docs_combinados, metas_combinados


# ─────────────────────────────────────────────────────────────
# DETECTOR DE PREGUNTAS GLOBALES (sobre TODO el vídeo)
# ─────────────────────────────────────────────────────────────

_PALABRAS_CLAVE_GLOBALES = [
    "resum", "resúmeme", "sinopsis", "sintetiza",
    "de qué va", "de qué trata", "de qué habla", "sobre qué habla", "sobre qué trata",
    "qué se ha debatido", "qué se debatió", "qué se dijo en el video", "qué se dijo en el vídeo",
    "qué se ha dicho en el video", "qué se ha dicho en el vídeo",
    "temas principales", "principales temas", "de qué se ha hablado", "de qué se habló",
    "todo el video", "todo el vídeo", "en general", "en resumen",
    "qué ha pasado", "qué pasó en", "cuéntame el video", "cuéntame el vídeo",
    "explícame el video", "explícame el vídeo", "puntos clave", "puntos principales"
]


# ─────────────────────────────────────────────────────────────
# DETECTOR DE PREGUNTAS SOBRE UN PONENTE CONCRETO
# ─────────────────────────────────────────────────────────────
# Cuando alguien pregunta por un ponente o partido concreto
# ("¿qué dijo Feijóo?", "postura del PP"), con pocos fragmentos
# basta y son más precisos. Con más fragmentos se mezcla ruido.

_PATRONES_PONENTE = re.compile(
    r"(qu[eé] dijo|qu[eé] argument[oó]|postura (de|del)|"
    r"intervenci[oó]n de|seg[uú]n|c[oó]mo vot[oó]|"
    r"qu[eé] pens[oó]|qu[eé] propuso|qu[eé] defendi[oó])",
    re.IGNORECASE
)


def _es_pregunta_ponente(pregunta: str) -> bool:
    """Detecta si la pregunta va sobre la intervención de un ponente o partido concreto."""
    return bool(_PATRONES_PONENTE.search(pregunta))


def _es_pregunta_global(pregunta: str) -> bool:
    """
    Detecta si la pregunta es sobre TODO el vídeo en general (ej. "resúmeme el vídeo",
    "¿de qué trata?", "temas principales") en vez de sobre un tema concreto.

    Esto es importante porque la búsqueda semántica normal (collection.query) compara
    tu pregunta con fragmentos concretos del vídeo. Una pregunta como "resúmeme el vídeo"
    no se parece semánticamente a ningún fragmento real (nadie dice literalmente "esto es
    un resumen"), así que devolvería fragmentos poco relevantes y el LLM, siendo honesto,
    diría que no tiene suficiente información. Para estas preguntas usamos en su lugar una
    muestra representativa de todo el vídeo (ver _muestrear_video_completo).
    """
    texto = pregunta.lower()
    return any(palabra in texto for palabra in _PALABRAS_CLAVE_GLOBALES)


def _detectar_ponentes_en_pregunta(pregunta: str, video_id: str) -> list[str]:
    """
    Busca los ponentes del vídeo en la base de datos y comprueba si 
    alguna palabra de su nombre (de longitud > 3) aparece en la pregunta.
    """
    res = collection.get(where={"video_id": {"$eq": video_id}}, include=["metadatas"])
    todos_ponentes = {m.get("ponente") for m in res.get("metadatas", []) if m.get("ponente")}
    
    ponentes_scores = {}
    palabras_pregunta = set(re.findall(r'\b\w+\b', pregunta.lower()))
    for ponente in todos_ponentes:
        if ponente.startswith("SPEAKER_"):
            continue
        nombre_limpio = ponente.split("(")[0].strip().lower()
        palabras_nombre = [p for p in re.findall(r'\b\w+\b', nombre_limpio) if len(p) > 3]
        
        # Contamos cuántas palabras del nombre aparecen en la pregunta
        matches = sum(1 for p in palabras_nombre if p in palabras_pregunta)
        if matches > 0:
            ponentes_scores[ponente] = matches
            
    if not ponentes_scores:
        return []
        
    # Solo nos quedamos con los ponentes que tengan la máxima puntuación
    max_score = max(ponentes_scores.values())
    ponentes_encontrados = [p for p, score in ponentes_scores.items() if score == max_score]
            
    return ponentes_encontrados

# ─────────────────────────────────────────────────────────────
# FUNCIÓN 1: BÚSQUEDA CON MEMORIA
# ─────────────────────────────────────────────────────────────

# Top_k por tipo de pregunta:
#   - Ponente concreto → 5: fragmentos consecutivos del mismo speaker, muy precisos
#   - Temática amplia  → 15: más cobertura para preguntas con varios aspectos
#   - Votaciones       → 5 semánticos + todos los de resultado forzados
#   - Global           → muestreo distribuido (ignora top_k)
TOP_K_PONENTE  = 5
TOP_K_TEMATICA = 15


def buscar(pregunta: str, video_id: str) -> dict:
    """
    Busca fragmentos relevantes y responde usando Mistral en la nube.
    Recuerda las preguntas anteriores del mismo vídeo (memoria de conversación,
    limitada a los últimos MAX_TURNOS_HISTORIAL turnos).

    Detecta automáticamente el tipo de pregunta y ajusta la estrategia:
      - GLOBAL (ej. "resúmeme", "de qué trata") → muestra repartida por todo el vídeo.
      - VOTACIONES (ej. "¿qué se votó?") → semántica + resultados de votación forzados.
      - PONENTE CONCRETO (ej. "¿qué dijo Feijóo?") → top_k=5, fragmentos precisos.
      - TEMÁTICA AMPLIA (cualquier otra) → top_k=15, más cobertura.

    Devuelve un diccionario con:
      - pregunta       → lo que preguntó el usuario
      - prompt         → el contexto que se mandó al LLM
      - respuesta_llm  → la respuesta generada
      - fuentes_top_k  → lista de fragmentos usados (ponente, texto, enlace, tiempos)
    """

    # 1. Clasificar la pregunta y elegir estrategia + top_k
    es_global   = _es_pregunta_global(pregunta)
    es_votacion = (not es_global) and _es_pregunta_votacion(pregunta)
    es_ponente  = (not es_global) and (not es_votacion) and _es_pregunta_ponente(pregunta)

    ponentes_detectados = []
    if not es_global and not es_votacion:
        ponentes_detectados = _detectar_ponentes_en_pregunta(pregunta, video_id)
        if ponentes_detectados:
            es_ponente = True

    top_k = TOP_K_PONENTE if es_ponente else TOP_K_TEMATICA

    condiciones_and = [{"video_id": {"$eq": video_id}}]
    
    # Excluir la Mesa solo si no mencionan la palabra "mesa" en la pregunta.
    if "mesa" not in pregunta.lower():
        condiciones_and.append({"partido": {"$ne": "Mesa"}})
        
    # Filtrado exacto de Ponente / Partido (Mejora A)
    partidos_detectados = [p for p in _PARTIDOS_CONOCIDOS if p.lower() in pregunta.lower() and p != "Mesa"]
    
    if ponentes_detectados:
        if len(ponentes_detectados) == 1:
            condiciones_and.append({"ponente": {"$eq": ponentes_detectados[0]}})
        else:
            condiciones_and.append({"ponente": {"$in": ponentes_detectados}})
    elif partidos_detectados:
        if len(partidos_detectados) == 1:
            condiciones_and.append({"partido": {"$eq": partidos_detectados[0]}})
        else:
            condiciones_and.append({"partido": {"$in": partidos_detectados}})
            
    where_filtro = {"$and": condiciones_and} if len(condiciones_and) > 1 else condiciones_and[0]

    if es_global:
        # Pregunta sobre TODO el vídeo (ej. "resúmeme", "de qué trata"):
        # usamos una muestra repartida por todo el vídeo, no búsqueda semántica.
        documentos, metadatos = _muestrear_video_completo(
            video_id, 
            n_fragmentos=40,
            excluir_mesa=("mesa" not in pregunta.lower())
        )

    elif es_votacion:
        # Pregunta sobre votaciones: combinamos búsqueda semántica normal
        # con los fragmentos que contienen resultados de votación explícitos,
        # que si no se podrían perder porque no se parecen semánticamente
        # a la pregunta ("¿qué se votó?" vs "180 votos a favor...").
        resultados = collection.query(
            query_texts=[pregunta],
            n_results=top_k,
            where=where_filtro,
            include=["documents", "metadatas"]
        )
        docs_semanticos  = resultados["documents"][0]
        metas_semanticos = resultados["metadatas"][0]

        docs_forzados, metas_forzados = _chunks_con_resultados_votacion(video_id)

        documentos, metadatos = _combinar_sin_duplicados(
            docs_forzados, metas_forzados,
            docs_semanticos, metas_semanticos
        )

    else:
        # Pregunta específica: búsqueda semántica normal en ChromaDB.
        # Pedimos algo más de top_k para poder filtrar ruido y aun así
        # quedarnos con top_k fragmentos útiles.
        resultados = collection.query(
            query_texts=[pregunta],
            n_results=top_k * 2,
            where=where_filtro,
            include=["documents", "metadatas"]
        )
        documentos = resultados["documents"][0]
        metadatos  = resultados["metadatas"][0]

        documentos, metadatos = _filtrar_ruido(documentos, metadatos)
        documentos = documentos[:top_k]
        metadatos  = metadatos[:top_k]

    if not documentos:
        return {
            "pregunta":      pregunta,
            "prompt":        "",
            "respuesta_llm": f"No se encontraron fragmentos relevantes en el video '{video_id}'.",
            "fuentes_top_k": []
        }

    # Fusión de chunks contiguos (Mejora B)
    documentos, metadatos = _fusionar_chunks_contiguos(documentos, metadatos)

    # 2. Construir contexto
    contexto = _construir_contexto(documentos, metadatos)

    # 3. Recuperar historial previo de este vídeo
    historial_previo = obtener_historial(video_id)

    # 4. Construir el mensaje del usuario (pregunta + contexto nuevo)
    if es_global:
        mensaje_usuario = (
            f"Pregunta del usuario: {pregunta}\n\n"
            f"A continuación tienes una MUESTRA REPRESENTATIVA de fragmentos repartidos "
            f"por TODO el vídeo (principio, medio y final). No son todos los fragmentos que "
            f"existen, pero sí una muestra fiel de todo el contenido. Responde con una visión "
            f"general basada en esta muestra; no te niegues a responder solo porque no sean "
            f"literalmente todos los fragmentos del vídeo.\n\n"
            f"Fragmentos:\n{contexto}"
        )
    elif es_votacion:
        mensaje_usuario = (
            f"Pregunta del usuario: {pregunta}\n\n"
            f"A continuación tienes fragmentos relevantes del vídeo, incluyendo aquellos "
            f"donde se anuncian resultados de votaciones (número de votos, si se aprueba "
            f"o rechaza, etc.). Usa esos datos exactos si responden a la pregunta.\n\n"
            f"Fragmentos:\n{contexto}"
        )
    else:
        mensaje_usuario = (
            f"Pregunta del usuario: {pregunta}\n\n"
            f"Fragmentos relevantes del vídeo:\n{contexto}"
        )

    # 5. Armar la lista de mensajes (historial + pregunta nueva)
    mensajes = historial_previo + [{"role": "user", "content": mensaje_usuario}]

    # Inyectar entidades conocidas para auto-etiquetado XML
    from src.motor_busqueda.db_neo4j import get_all_entities_for_prompt
    texto_entidades = get_all_entities_for_prompt()

    system = (
        "Eres un asistente especializado en sesiones parlamentarias españolas. "
        "Responde ÚNICAMENTE con información que esté en los fragmentos proporcionados; "
        "si la respuesta no está en los fragmentos, dilo claramente en vez de inventarla. "
        "Dentro de ese límite, sé todo lo útil y completo posible:\n"
        "- Si varios ponentes hablan del mismo tema, SINTETIZA sus posturas en vez de listarlas "
        "de forma aislada: señala en qué coinciden, en qué difieren y por qué.\n"
        "- Si la pregunta tiene varios aspectos o hay varias posturas distintas, ESTRUCTURA la "
        "respuesta en apartados breves (por ejemplo, un apartado por postura o por grupo "
        "parlamentario), en vez de un único párrafo genérico.\n"
        "- CITA siempre por nombre y, si se conoce, por partido o grupo parlamentario "
        "(ej. 'Pérez Masó (Junts) defendió que...'), en vez de decir simplemente 'un diputado dijo...'.\n"
        "- Cada fragmento del contexto empieza con la identificación exacta de quien habla "
        "(su nombre real, o un código como 'SPEAKER_04' cuando el sistema no pudo reconocer su "
        "identidad). Copia siempre esa identificación tal cual aparece, letra por letra, al citarla "
        "en tu respuesta.\n"
        "- ATRIBUCIÓN ESTRICTA: solo atribuye una afirmación o postura al partido o persona que "
        "la dijo literalmente en su propio fragmento. No infieras ni des por hecho que un partido "
        "'apoya' o 'está de acuerdo con' algo solo porque otro ponente lo mencionó o lo dijo de él.\n"
        "- SÉ CONCISO: responde de forma directa, sin relleno ni repetir la pregunta. Para preguntas "
        "simples, 2-4 párrafos suelen bastar; reserva las respuestas más largas y estructuradas para "
        "preguntas con varios aspectos o posturas distintas.\n"
        "- Puedes usar el historial de la conversación para dar respuestas de seguimiento coherentes.\n\n"
        "INSTRUCCIÓN MUY IMPORTANTE (AUTO-ETIQUETADO XML):\n"
        "Cuando menciones a alguna de las personas, partidos, leyes, eventos o conceptos listados abajo en tu respuesta, "
        "DEBES envolver su nombre en una etiqueta XML <entidad id=\"...\">nombre</entidad> usando el ID "
        "exacto que se te proporciona. Esto es vital para el frontend. Solo hazlo con las entidades que "
        "aparezcan en esta lista. Si una entidad sale varias veces, etiquétala todas las veces.\n\n"
        f"{texto_entidades}"
    )

    # 6. Llamar a Mistral
    respuesta_llm = _llamar_mistral(system, mensajes)

    # 7. Guardar en historial (pregunta + respuesta) y recortarlo si hace falta
    if video_id not in _historial:
        _historial[video_id] = []
    _historial[video_id].append({"role": "user",      "content": mensaje_usuario})
    _historial[video_id].append({"role": "assistant", "content": respuesta_llm})
    _recortar_historial(video_id)

    # 8. Construir fuentes
    fuentes_top_k = []
    for doc, meta in zip(documentos, metadatos):
        fuente = {**meta}  # Incluimos todos los metadatos (nombre, partido, confianza_id, etc.)
        
        # Añadimos y formateamos los campos específicos que espera el frontend
        ponente_mostrado = _nombre_mostrar(meta)
        fuente["ponente"] = ponente_mostrado
        fuente["nombre"] = ponente_mostrado.split(' (')[0]
        fuente["texto"] = doc
        fuente["enlace_video"] = meta.get("url_exacta_tiempo", "")
        # Guardamos el formato en minutos:segundos, manteniendo los originales como floats en la copia de meta
        fuente["inicio_str"] = _segundos_a_mmss(meta["inicio"])
        fuente["fin_str"] = _segundos_a_mmss(meta["fin"])
        
        # Para compatibilidad con el frontend actual que lee "inicio" y "fin" formateados:
        fuente["inicio_segundos"] = meta["inicio"]
        fuente["fin_segundos"] = meta["fin"]
        fuente["inicio"] = _segundos_a_mmss(meta["inicio"])
        fuente["fin"] = _segundos_a_mmss(meta["fin"])
        
        fuentes_top_k.append(fuente)

    return {
        "pregunta":      pregunta,
        "prompt":        mensaje_usuario,
        "respuesta_llm": respuesta_llm,
        "fuentes_top_k": fuentes_top_k
    }


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 2: RESUMEN GLOBAL DEL VÍDEO
# ─────────────────────────────────────────────────────────────

def generar_resumen(video_id: str, n_fragmentos: int = 40) -> dict:
    """
    Genera un resumen de los temas principales debatidos en el vídeo.
    Se recomienda llamar esta función al cargar el vídeo en el frontend.

    IMPORTANTE: usa _muestrear_video_completo() para coger fragmentos
    repartidos por TODO el vídeo (no solo el principio), y le pide al LLM
    que use las palabras literales del texto al nombrar los temas, para que
    luego la búsqueda semántica (buscar()) encuentre fragmentos reales
    cuando el usuario pregunte sobre esos temas.

    Devuelve un diccionario con:
      - video_id  → id del vídeo
      - resumen   → texto del resumen generado por el LLM
      - error     → mensaje de error si algo falló (None si todo fue bien)
    """

    documentos, metadatos = _muestrear_video_completo(video_id, n_fragmentos)

    if not documentos:
        return {
            "video_id": video_id,
            "resumen":  "",
            "error":    f"No se encontraron fragmentos para el video '{video_id}'."
        }

    contexto = _construir_contexto(documentos, metadatos)

    from src.motor_busqueda.db_neo4j import get_all_entities_for_prompt
    texto_entidades = get_all_entities_for_prompt()

    system = (
        "Eres un asistente especializado en sesiones parlamentarias españolas. "
        "Tu tarea es hacer un resumen claro, organizado y FIEL AL TEXTO ORIGINAL. "
        "DEBES estructurar tu respuesta EXACTAMENTE con los siguientes dos encabezados:\n"
        "### 1. Índice de Temas\n"
        "### 2. Resumen Global\n\n"
        "Cada fragmento del contexto empieza con la identificación exacta de quien habla "
        "(su nombre real, o un código como 'SPEAKER_XX' cuando el sistema no pudo reconocer su "
        "identidad). Cuando cites a un ponente, copia siempre esa identificación tal cual "
        "aparece, letra por letra.\n\n"
        "INSTRUCCIÓN MUY IMPORTANTE (AUTO-ETIQUETADO XML):\n"
        "Cuando menciones a alguna de las personas, partidos, leyes, eventos o conceptos listados abajo en tu respuesta, "
        "DEBES envolver su nombre en una etiqueta XML <entidad id=\"...\">nombre</entidad> usando el ID "
        "exacto que se te proporciona. Esto es vital para el frontend. Solo hazlo con las entidades que "
        "aparezcan en esta lista. Si una entidad sale varias veces, etiquétala todas las veces.\n\n"
        f"{texto_entidades}"
    )

    mensaje = (
        f"A continuación tienes fragmentos de una sesión parlamentaria, "
        f"repartidos a lo largo de TODO el vídeo (no solo el principio).\n\n"
        f"{contexto}\n\n"
        "Por favor, genera tu respuesta siguiendo esta estructura estricta:\n\n"
        "### 1. Índice de Temas\n"
        "- Escribe una lista de viñetas con los temas principales que se debatieron.\n"
        "- Usa el formato: '**Tema**: Breve descripción'.\n"
        "- MUY IMPORTANTE: nombra cada tema usando las MISMAS PALABRAS o expresiones "
        "que aparecen literalmente en los fragmentos de arriba (por ejemplo, si en el "
        "texto se habla de 'la subida del salario mínimo', el tema debe llamarse "
        "'Subida del salario mínimo', NO uses sinónimos rebuscados ni lo generalices "
        "como 'Política económica'). Esto es porque luego un usuario va a preguntar "
        "usando ese mismo nombre de tema y necesitamos que las palabras coincidan "
        "con lo que realmente se dijo.\n"
        "- No inventes ni incluyas temas que no aparezcan en los fragmentos.\n\n"
        "### 2. Resumen Global\n"
        "Escribe un resumen en párrafos continuos que explique:\n"
        "- Las posturas más destacadas de los ponentes.\n"
        "- Cualquier acuerdo o desacuerdo relevante.\n"
        "Sé claro y directo. Máximo 300 palabras en el resumen global."
    )

    try:
        resumen = _llamar_mistral(system, [{"role": "user", "content": mensaje}])
        return {"video_id": video_id, "resumen": resumen, "error": None}
    except Exception as e:
        return {"video_id": video_id, "resumen": "", "error": str(e)}


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 3: EXTRACCIÓN DE ENTIDADES CON BÚSQUEDA WEB
# ─────────────────────────────────────────────────────────────

def _buscar_wikipedia(termino: str) -> tuple[str, str]:
    """
    Busca un término en Wikipedia en español y devuelve el primer párrafo y la URL de la foto principal.
    No necesita API key, usa la API pública gratuita de Wikipedia.
    Devuelve (extracto, foto_url).
    """
    try:
        termino_codificado = urllib.parse.quote(termino)
        url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{termino_codificado}"
        req = urllib.request.Request(url, headers={"User-Agent": "ParlamentoChatbot/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            datos = json.loads(resp.read().decode("utf-8"))
            extracto = datos.get("extract", "")
            foto = ""
            if "thumbnail" in datos and "source" in datos["thumbnail"]:
                foto = datos["thumbnail"]["source"]
            elif "originalimage" in datos and "source" in datos["originalimage"]:
                foto = datos["originalimage"]["source"]
            return extracto, foto
    except Exception:
        return "", ""


def _detectar_entidades_con_llm(texto: str) -> dict:
    """
    Usa Groq para detectar leyes, lugares e instituciones en el texto parlamentario,
    ya que es más rápido y tiene un límite de llamadas más alto que Mistral.

    NOTA: ya NO se le pide detectar personas aquí. Las personas (ponentes) ya vienen
    identificadas con nombre y partido en los metadatos de ChromaDB gracias al sistema
    de diarización, así que pedirle al LLM que las "adivine" leyendo el texto es
    redundante y menos fiable (ver _extraer_personas_de_metadatos).

    Devuelve {"leyes": [...], "lugares": [...], "instituciones": [...]}
    """
    system = (
        "Eres un extractor de entidades de textos parlamentarios españoles. "
        "Responde ÚNICAMENTE con un JSON válido, sin texto adicional ni backticks."
    )
    mensaje = (
        "El siguiente texto proviene de una sesión plenaria del Congreso de los Diputados de España.\n\n"
        f"Texto:\n{texto}\n\n"
        "Extrae todas las entidades relevantes que aparezcan:\n"
        "1. LEYES: leyes, decretos, normativas o reglamentos mencionados. "
        "Incluye el nombre completo si aparece (ej: 'Ley Orgánica 3/2007').\n"
        "2. LUGARES: países, comunidades autónomas, ciudades o pueblos mencionados "
        "(ej: 'Cataluña', 'Francia', 'Sevilla', 'Vitoria-Gasteiz').\n"
        "3. INSTITUCIONES: organismos, ministerios, tribunales u "
        "otras instituciones mencionadas "
        "(ej: 'Tribunal Constitucional', 'Ministerio de Hacienda').\n"
        "4. PARTIDOS: partidos políticos o coaliciones mencionadas explícitamente "
        "(ej: 'Partido Popular', 'PSOE', 'Sumar', 'Junts'). Para cada partido, "
        "intenta extraer su nombre completo y sus posibles alias o siglas (ej: 'PP', 'Grupo Popular').\n"
        "5. EVENTOS: sucesos históricos, crisis, escándalos o hitos temporales mencionados "
        "(ej: 'La pandemia', 'El 11-M', 'Guerra civil', 'La crisis de 2008').\n"
        "6. CONCEPTOS: programas políticos, fondos, o abstracciones clave mencionadas "
        "(ej: 'Fondos europeos', 'Agenda 2030', 'Amnistía', 'Estado de bienestar').\n\n"
        "Devuelve SOLO este JSON (sin nada más):\n"
        '{"leyes": ["nombre completo ley 1", "..."], '
        '"lugares": ["Lugar 1", "..."], '
        '"instituciones": ["Institución 1", "..."], '
        '"partidos": [{"nombre": "Partido Popular", "alias": ["PP", "Grupo Popular"]}], '
        '"eventos": ["Evento 1", "..."], '
        '"conceptos": ["Concepto 1", "..."]}'
    )
    try:
        respuesta = _llamar_groq(system, [{"role": "user", "content": mensaje}])
        respuesta_limpia = respuesta.strip().strip("```json").strip("```").strip()
        return json.loads(respuesta_limpia)
    except Exception as e:
        print(f"[!] _detectar_entidades_con_llm falló: {type(e).__name__}: {e}")
        return {"leyes": [], "lugares": [], "instituciones": [], "partidos": [], "eventos": [], "conceptos": []}


def _resolver_nombre_wikipedia(nombre: str, tipo: str) -> tuple[str, str]:
    """
    Intenta encontrar el término exacto que Wikipedia reconoce.
    Primero prueba el nombre tal cual. Si no funciona, prueba variantes.
    Devuelve (texto_wikipedia, foto_url).
    """
    # Intento 1: nombre tal cual
    extracto, foto = _buscar_wikipedia(nombre)
    if extracto and len(extracto) > 50:
        return extracto, foto

    # Intento 2: si es persona, probar "Nombre Apellido (político)"
    if tipo == "persona":
        extracto, foto = _buscar_wikipedia(f"{nombre} (político)")
        if extracto and len(extracto) > 50:
            return extracto, foto

    # Intento 3: preguntar a Mistral cuál es el título exacto del artículo de Wikipedia
    system = (
        "Eres un asistente que conoce la Wikipedia en español. "
        "Responde ÚNICAMENTE con el título exacto del artículo, sin explicaciones."
    )
    if tipo == "persona":
        mensaje = (
            f"¿Cuál es el título exacto del artículo de Wikipedia en español "
            f"sobre '{nombre}', político español? "
            "Responde solo con el título, nada más."
        )
    else:
        mensaje = (
            f"¿Cuál es el título exacto del artículo de Wikipedia en español "
            f"sobre la ley o normativa '{nombre}'? "
            "Responde solo con el título, nada más."
        )
    try:
        titulo_wikipedia = _llamar_groq(system, [{"role": "user", "content": mensaje}])
        titulo_limpio = titulo_wikipedia.strip().strip('"').strip("'")
        extracto, foto = _buscar_wikipedia(titulo_limpio)
        if extracto and len(extracto) > 50:
            return extracto, foto
    except Exception:
        pass

    return "", ""


def _extraer_personas_de_metadatos(metadatos: list) -> list[dict]:
    """
    Saca las personas (ponentes) DIRECTAMENTE de los metadatos de ChromaDB,
    usando los campos 'nombre' y 'partido' que ya rellenó el sistema de
    diarización/identificación de tu compañero.

    Esto sustituye a pedirle al LLM que "adivine" quién es alguien leyendo
    el texto (ej. deducir que 'Sánchez' es 'Pedro Sánchez'): es más preciso
    porque usa una identificación ya verificada, en vez de una suposición.

    Ignora fragmentos sin nombre identificado (estado_id "DESCONOCIDO" o "AMBIGUO",
    donde meta["nombre"] es None), y no repite a la misma persona dos veces.

    Devuelve una lista de dicts: [{"nombre": ..., "partido": ...}, ...]
    """
    vistos: dict[str, dict] = {}
    for meta in metadatos:
        nombre = meta.get("nombre")
        if not nombre:
            continue
        if nombre not in vistos:
            vistos[nombre] = {"nombre": nombre, "partido": meta.get("partido")}
    return list(vistos.values())


def _explicar_entidad(nombre: str, tipo: str, partido: str = None) -> tuple[str, str, str, str]:
    """
    Busca información en Wikipedia y usa Mistral para generar una explicación breve.
    tipo puede ser "ley", "lugar", "institucion" o "persona".

    Si ya conocemos el partido de la persona (porque viene de los metadatos
    identificados, ver _extraer_personas_de_metadatos), se lo pasamos al LLM
    directamente en vez de dejar que tenga que deducirlo o alucinarlo.
    
    Devuelve (explicacion, foto_url, fuente_origen).
    """
    if tipo == "persona":
        from src.motor_busqueda.congreso_api import congreso_api
        diputado = congreso_api.buscar_diputado(nombre)
        if diputado:
            explicacion = diputado.get("biografia", "")
            foto_url = diputado.get("foto_url", "")
            fuente_origen = "Congreso de los Diputados"
            return explicacion, foto_url, fuente_origen, diputado["nombre"]

    info_wikipedia, foto_url = _resolver_nombre_wikipedia(nombre, tipo)


    if info_wikipedia:
        fuente = f"Información de Wikipedia:\n{info_wikipedia[:1500]}"
        fuente_origen = "Wikipedia"
    else:
        fuente = "No se encontró información ni en Wikipedia ni mediante LLM."
        fuente_origen = "No encontrada"

    if tipo == "ley":
        instruccion = (
            f"Explica en 2-3 frases qué es '{nombre}', "
            "para qué sirve y cuándo entró en vigor. "
            "Sé directo y claro, como si se lo explicaras a alguien sin conocimientos jurídicos."
        )
    elif tipo == "lugar":
        instruccion = (
            f"Explica en 2-3 frases qué es o dónde está '{nombre}' "
            "y por qué es relevante en el contexto de la política española. "
            "Sé directo y claro."
        )
    elif tipo == "institucion" or tipo == "partido":
        instruccion = (
            f"Explica en 2-3 frases qué es '{nombre}': "
            "qué función o ideología tiene y por qué aparece en debates parlamentarios. "
            "Sé directo y claro."
        )
    elif tipo == "evento":
        instruccion = (
            f"Explica en 2-3 frases qué fue el evento '{nombre}', "
            "cuándo ocurrió y por qué es relevante en la política o historia de España. "
            "Sé directo y objetivo."
        )
    elif tipo == "concepto":
        instruccion = (
            f"Explica en 2-3 frases qué significa el concepto, programa o término '{nombre}' "
            "en el contexto político actual español. "
            "Sé directo y claro."
        )
    else:
        info_partido = f" Sabemos que está afiliado/a o pertenece al grupo '{partido}'." if partido else ""
        instruccion = (
            f"Explica en 2-3 frases quién es '{nombre}' en el contexto "
            "de la política española actual: qué cargo ocupa o ha ocupado "
            f"y por qué aparece en debates parlamentarios.{info_partido} "
            "Sé directo y objetivo."
        )

    system = "Eres un asistente que explica términos políticos y jurídicos de forma sencilla, neutra y enciclopédica."
    prompt = (
        f"{instruccion}\n"
        f"{fuente}\n\n"
        "REGLA MUY IMPORTANTE: Si la fuente no te da información y tú tampoco sabes con seguridad "
        "quién o qué es esta entidad, debes responder ÚNICAMENTE con la frase exacta: "
        "'Entidad mencionada en la sesión (información adicional no disponible).' "
        "Bajo NINGÚN concepto pidas disculpas, no des consejos de cómo buscar, ni expliques por qué no lo sabes. "
        "Si sí lo sabes, responde solo con la explicación directa, sin introducciones conversacionales ni saludos."
    )
    try:
        explicacion = _llamar_groq(system, [{"role": "user", "content": prompt}])
        return explicacion, foto_url, fuente_origen, nombre
    except Exception:
        explicacion_fallback = info_wikipedia[:300] if info_wikipedia else "No se pudo obtener información."
        return explicacion_fallback, foto_url, fuente_origen, nombre


def extraer_entidades(video_id: str, pregunta: str = "", top_k: int = 10) -> dict:
    """
    Detecta leyes y personas relevantes en los fragmentos del vídeo,
    busca información sobre cada una en Wikipedia y genera una explicación breve.

    Devuelve un diccionario con:
      - video_id   → id del vídeo
      - entidades  → lista de entidades, cada una con:
                       · nombre      → nombre de la entidad
                       · tipo        → "ley" o "persona"
                       · explicacion → 2-3 frases explicando qué es / quién es
      - error      → mensaje de error si algo falló (None si todo fue bien)
    """
    try:
        if pregunta:
            resultados = collection.query(
                query_texts=[pregunta],
                n_results=top_k,
                where={"video_id": {"$eq": video_id}},
                include=["documents", "metadatas"]
            )
            documentos = resultados["documents"][0]
            metadatos  = resultados["metadatas"][0]
        else:
            documentos, metadatos = _muestrear_video_completo(video_id, top_k)

        if not documentos:
            return {
                "video_id":  video_id,
                "entidades": [],
                "error":     f"No se encontraron fragmentos para el video '{video_id}'."
            }

        from src.motor_busqueda.db_neo4j import guardar_entidad_dinamica, obtener_entidad_existente

        # PERSONAS: se sacan directamente de los metadatos (nombre + partido ya
        # identificados por diarización), NO se le pide al LLM que las adivine.
        personas = _extraer_personas_de_metadatos(metadatos)
        
        # Diccionario temporal para guardar las equivalencias nombre_original -> nombre_oficial
        mapeo_nombres = {}
        entidades = []

        for persona in personas:
            # Identificar nombre oficial para corrección en frontend
            nombre_original = persona["nombre"]
            nombre_oficial = nombre_original
            
            existente = obtener_entidad_existente(nombre_original, "persona")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente["foto_url"]
                fuente_origen = existente.get("fuente", "Caché Neo4j")
                nombre_oficial = existente.get("nombre", nombre_original)
            else:
                explicacion, foto_url, fuente_origen, nombre_oficial = _explicar_entidad(
                    nombre_original, "persona", partido=persona.get("partido")
                )
                
            # Guardar en Neo4j usando el nombre OFICIAL para no ensuciar el grafo con alucinaciones
            id_entidad = guardar_entidad_dinamica(nombre_oficial, "persona", explicacion, foto_url=foto_url, partido=persona.get("partido"), video_id=video_id, fuente=fuente_origen)
            
            entidades.append({
                "id":          id_entidad,
                "nombre":      nombre_oficial,
                "tipo":        "persona",
                "explicacion": explicacion
            })
            
            if nombre_original != nombre_oficial:
                mapeo_nombres[nombre_original] = nombre_oficial

        # LEYES, LUGARES E INSTITUCIONES: esto sí sigue necesitando al LLM,
        # porque esa información no viene en los metadatos.
        texto_completo = " ".join(documentos)
        
        # Para evitar el error HTTP 413 (Payload Too Large) de Groq al pedir top_k altos,
        # dividimos el texto si es muy largo y combinamos los resultados.
        max_chars = 15000
        detectadas = {"leyes": [], "lugares": [], "instituciones": [], "partidos": [], "eventos": [], "conceptos": []}
        
        for i in range(0, len(texto_completo), max_chars):
            segmento = texto_completo[i:i+max_chars]
            try:
                res_segmento = _detectar_entidades_con_llm(segmento)
                print(f"[DEBUG LLM] Groq: leyes={res_segmento.get('leyes',[])} partidos={res_segmento.get('partidos',[])} lugares={res_segmento.get('lugares',[])} instituciones={res_segmento.get('instituciones',[])} eventos={res_segmento.get('eventos',[])} conceptos={res_segmento.get('conceptos',[])}")
                for k in detectadas.keys():
                    if k in res_segmento:
                        # Los partidos son dicts ({"nombre":..., "alias":...}), el resto son strings
                        if k == "partidos":
                            nombres_existentes = [p["nombre"] if isinstance(p, dict) else p for p in detectadas[k]]
                            for p in res_segmento[k]:
                                p_nombre = p["nombre"] if isinstance(p, dict) else p
                                if p_nombre not in nombres_existentes:
                                    detectadas[k].append(p)
                        else:
                            for item in res_segmento[k]:
                                if item not in detectadas[k]:
                                    detectadas[k].append(item)
            except Exception as e:
                print(f"Error procesando segmento para entidades: {e}")

        leyes         = detectadas.get("leyes", [])
        lugares       = detectadas.get("lugares", [])
        instituciones = detectadas.get("instituciones", [])
        partidos      = detectadas.get("partidos", [])
        eventos       = detectadas.get("eventos", [])
        conceptos     = detectadas.get("conceptos", [])
        print(f"[DEBUG TOTAL] leyes={len(leyes)} lugares={len(lugares)} instituciones={len(instituciones)} partidos={len(partidos)} eventos={len(eventos)} conceptos={len(conceptos)}")



        for ley in leyes:
            if not ley or len(ley) < 3:
                continue
            existente = obtener_entidad_existente(ley, "ley")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente["foto_url"]
                fuente_origen = existente.get("fuente", "Caché Neo4j")
            else:
                explicacion, foto_url, fuente_origen, _ = _explicar_entidad(ley, "ley")
            id_entidad = guardar_entidad_dinamica(ley, "ley", explicacion, foto_url=foto_url, video_id=video_id, fuente=fuente_origen)
            entidades.append({
                "id":          id_entidad,
                "nombre":      ley,
                "tipo":        "ley",
                "explicacion": explicacion
            })



        for lugar in lugares:
            if not lugar or len(lugar) < 3:
                continue
            existente = obtener_entidad_existente(lugar, "lugar")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente["foto_url"]
                fuente_origen = existente.get("fuente", "Caché Neo4j")
            else:
                explicacion, foto_url, fuente_origen, _ = _explicar_entidad(lugar, "lugar")
            id_entidad = guardar_entidad_dinamica(lugar, "lugar", explicacion, foto_url=foto_url, video_id=video_id, fuente=fuente_origen)
            entidades.append({
                "id":          id_entidad,
                "nombre":      lugar,
                "tipo":        "lugar",
                "explicacion": explicacion
            })

        for institucion in instituciones:
            if not institucion or len(institucion) < 3:
                continue
            existente = obtener_entidad_existente(institucion, "institucion")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente["foto_url"]
                fuente_origen = existente.get("fuente", "Caché Neo4j")
            else:
                explicacion, foto_url, fuente_origen, _ = _explicar_entidad(institucion, "institucion")
            id_entidad = guardar_entidad_dinamica(institucion, "institucion", explicacion, foto_url=foto_url, video_id=video_id, fuente=fuente_origen)
            entidades.append({
                "id":          id_entidad,
                "nombre":      institucion,
                "tipo":        "institucion",
                "explicacion": explicacion
            })

        for partido_obj in partidos:
            if isinstance(partido_obj, dict):
                partido_nombre = partido_obj.get("nombre")
                alias_list = partido_obj.get("alias", [])
            else:
                partido_nombre = partido_obj
                alias_list = []
                
            if not partido_nombre or len(partido_nombre) < 2:
                continue
                
            existente = obtener_entidad_existente(partido_nombre, "partido")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente["foto_url"]
                fuente_origen = existente.get("fuente", "Caché Neo4j")
            else:
                explicacion, foto_url, fuente_origen, _ = _explicar_entidad(partido_nombre, "partido")
                
            alias_str = ", ".join(alias_list) if alias_list else None
            id_entidad = guardar_entidad_dinamica(partido_nombre, "partido", explicacion, foto_url=foto_url, video_id=video_id, alias=alias_str, fuente=fuente_origen)
            entidades.append({
                "id":          id_entidad,
                "nombre":      partido_nombre,
                "tipo":        "partido",
                "explicacion": explicacion
            })

        for evento in eventos:
            if not evento or len(evento) < 3:
                continue
            existente = obtener_entidad_existente(evento, "evento")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente["foto_url"]
                fuente_origen = existente.get("fuente", "Caché Neo4j")
            else:
                explicacion, foto_url, fuente_origen, _ = _explicar_entidad(evento, "evento")
            id_entidad = guardar_entidad_dinamica(evento, "evento", explicacion, foto_url=foto_url, video_id=video_id, fuente=fuente_origen)
            entidades.append({
                "id":          id_entidad,
                "nombre":      evento,
                "tipo":        "evento",
                "explicacion": explicacion
            })

        for concepto in conceptos:
            if not concepto or len(concepto) < 3:
                continue
            existente = obtener_entidad_existente(concepto, "concepto")
            if existente and existente.get("explicacion"):
                explicacion = existente["explicacion"]
                foto_url = existente.get("foto_url", "")
                fuente_origen = existente.get("fuente", "Caché Neo4j")
            else:
                explicacion, foto_url, fuente_origen, _ = _explicar_entidad(concepto, "concepto")
            id_entidad = guardar_entidad_dinamica(concepto, "concepto", explicacion, video_id=video_id, foto_url=foto_url, fuente=fuente_origen)
            entidades.append({
                "id":          id_entidad,
                "nombre":      concepto,
                "tipo":        "concepto",
                "explicacion": explicacion
            })

        return {"video_id": video_id, "entidades": entidades, "error": None}

    except Exception as e:
        import traceback
        print(f"[!] EXCEPCIÓN EN extraer_entidades: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return {"video_id": video_id, "entidades": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 4: CHUNKING AMPLIADO
# ─────────────────────────────────────────────────────────────

def obtener_intervencion_completa(video_id: str, ponente: str, inicio: float, fin: float) -> dict:
    """
    Dado un ponente y un intervalo de tiempo, recupera todos los fragmentos
    de ese ponente que estén cerca en el tiempo (±60 segundos de margen).

    Devuelve un diccionario con:
      - ponente           → nombre del ponente
      - inicio_mmss       → tiempo de inicio formateado
      - fin_mmss          → tiempo de fin formateado
      - texto_fragmento   → el fragmento exacto solicitado
      - contexto_completo → todos los fragmentos del ponente en ese bloque de tiempo
      - error             → mensaje de error si algo falló (None si todo fue bien)
    """

    MARGEN_SEGUNDOS = 60

    try:
        # Buscamos por video_id y franja de tiempo sin filtrar por 'ponente' aquí,
        # porque el 'ponente' recibido viene con el formato de la UI (ej. "Nombre (Partido)")
        # y Chroma guarda el ponente como "SPEAKER_00" o similar en su campo 'ponente'.
        resultados = collection.get(
            where={
                "$and": [
                    {"video_id": {"$eq": video_id}},
                    {"inicio":   {"$gte": max(0.0, inicio - MARGEN_SEGUNDOS)}},
                    {"fin":      {"$lte": fin + MARGEN_SEGUNDOS}}
                ]
            },
            include=["documents", "metadatas"]
        )

        documentos = resultados.get("documents", [])
        metadatos  = resultados.get("metadatas", [])

        # Filtramos en memoria por el nombre formateado
        pares_filtrados = []
        for doc, meta in zip(documentos, metadatos):
            if _nombre_mostrar(meta) == ponente:
                pares_filtrados.append((meta, doc))

        if not pares_filtrados:
            return {
                "ponente":           ponente,
                "inicio_mmss":       _segundos_a_mmss(inicio),
                "fin_mmss":          _segundos_a_mmss(fin),
                "texto_fragmento":   "",
                "contexto_completo": [],
                "error":             "No se encontraron fragmentos para esa intervención."
            }

        pares = sorted(pares_filtrados, key=lambda x: x[0].get("inicio", 0))

        # Usamos el nombre real (si se conoce) del primer fragmento encontrado
        nombre_real = _nombre_mostrar(pares[0][0])

        docs_a_fusionar = [p[1] for p in pares]
        metas_a_fusionar = [p[0] for p in pares]
        
        # Fusión de chunks contiguos (Mejora B)
        docs_fusionados, metas_fusionados = _fusionar_chunks_contiguos(docs_a_fusionar, metas_a_fusionar)

        texto_exacto = ""
        for meta, doc in zip(metas_fusionados, docs_fusionados):
            if meta["inicio"] <= fin and meta["fin"] >= inicio:
                texto_exacto = doc
                break

        contexto_completo = [
            {
                "inicio": _segundos_a_mmss(meta["inicio"]),
                "fin":    _segundos_a_mmss(meta["fin"]),
                "texto":  doc
            }
            for meta, doc in zip(metas_fusionados, docs_fusionados)
        ]

        return {
            "ponente":           nombre_real,
            "inicio_mmss":       _segundos_a_mmss(inicio),
            "fin_mmss":          _segundos_a_mmss(fin),
            "texto_fragmento":   texto_exacto,
            "contexto_completo": contexto_completo,
            "error":             None
        }

    except Exception as e:
        return {
            "ponente":           ponente,
            "inicio_mmss":       _segundos_a_mmss(inicio),
            "fin_mmss":          _segundos_a_mmss(fin),
            "texto_fragmento":   "",
            "contexto_completo": [],
            "error":             str(e)
        }


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 5: BÚSQUEDA TRANSVERSAL (varios vídeos a la vez)
# ─────────────────────────────────────────────────────────────
# A diferencia de buscar(), esta función NO filtra por video_id: compara
# la pregunta contra los fragmentos de TODOS los plenos cargados en la
# base de datos. Pensada para preguntas tipo "¿qué ha dicho el PSOE sobre
# la vivienda en los últimos plenos?", donde el usuario no elige un vídeo
# concreto sino que quiere ver todo lo relacionado, venga de donde venga.

# Más alto que TOP_K_TEMATICA porque aquí compite contenido de MUCHOS
# vídeos distintos a la vez, así que hacen falta más fragmentos para
# tener una imagen representativa.
TOP_K_TRANSVERSAL = 30

# Lista de partidos conocidos para detectar automáticamente si la pregunta
# menciona a alguno (así no hace falta que el frontend lo indique siempre
# a mano). Si tu base de datos usa otros nombres de partido en el campo
# "partido", añádelos aquí para que la detección automática funcione.
_PARTIDOS_CONOCIDOS = [
    "PSOE", "PP", "Vox", "Sumar", "ERC", "Junts", "EH Bildu", "Bildu",
    "PNV", "BNG", "CC", "UPN", "Podemos", "Ciudadanos", "Cs", "Mesa"
]


def _detectar_partido_en_pregunta(pregunta: str) -> str | None:
    """
    Busca si la pregunta menciona explícitamente el nombre de un partido
    conocido (ej. "PSOE", "Junts"). Si lo encuentra, devuelve el nombre
    tal cual está escrito en _PARTIDOS_CONOCIDOS, para usarlo como filtro
    de metadatos ("partido") en ChromaDB.

    Si no se detecta ninguno, devuelve None (no se filtra por partido y
    se busca entre todos los ponentes).
    """
    texto = pregunta.lower()
    for partido in _PARTIDOS_CONOCIDOS:
        if partido.lower() in texto:
            return partido
    return None


def _construir_contexto_transversal(documentos: list, metadatos: list) -> str:
    """
    Igual que _construir_contexto(), pero como aquí los fragmentos pueden
    venir de vídeos (plenos) distintos, cada línea añade también el título
    y la fecha del vídeo de origen entre corchetes, para que el LLM pueda
    distinguir de qué sesión habla cada fragmento y no las mezcle.
    """
    lineas = []
    for doc, meta in zip(documentos, metadatos):
        t_inicio = _segundos_a_mmss(meta["inicio"])
        t_fin    = _segundos_a_mmss(meta["fin"])
        nombre   = _nombre_mostrar(meta)
        titulo   = meta.get("titulo_video") or meta.get("video_id", "Vídeo desconocido")
        fecha    = meta.get("fecha_publicacion")
        cabecera = f"[{titulo} — {fecha}]" if fecha else f"[{titulo}]"
        lineas.append(f"{cabecera} {nombre} ({t_inicio} - {t_fin}): [{doc}]")
    return "\n".join(lineas)


def buscar_transversal(pregunta: str, partido: str = None, top_k: int = TOP_K_TRANSVERSAL) -> dict:
    """
    Como buscar(), pero busca en TODOS los vídeos de la colección a la vez,
    en lugar de limitarse a un único video_id. Pensada para preguntas tipo
    "¿qué ha dicho el PSOE sobre la vivienda en los últimos plenos?".

    Parámetros:
      - pregunta → pregunta del usuario.
      - partido  → si se indica (ej. "PSOE"), filtra los fragmentos para
                   que solo se busque entre intervenciones de ese partido.
                   Si se deja en None, se intenta detectar automáticamente
                   a partir del texto de la pregunta (ver
                   _detectar_partido_en_pregunta); si tampoco se detecta
                   nada, se busca entre TODOS los partidos.
      - top_k    → número de fragmentos a recuperar. Por defecto más alto
                   que en buscar() (TOP_K_TRANSVERSAL = 30), porque aquí
                   se compite con contenido de muchos vídeos distintos.

    La memoria de conversación de estas búsquedas se guarda POR SEPARADO
    de la de buscar() (que es por video_id), en un "hilo" propio según el
    filtro de partido usado. Así, si el usuario alterna entre preguntar
    sobre un vídeo concreto y hacer una búsqueda transversal, las dos
    conversaciones no se mezclan.

    Devuelve el mismo formato que buscar(), con dos añadidos en cada
    elemento de "fuentes_top_k": "video_id" y "titulo_video", ya que aquí
    los fragmentos pueden venir de vídeos distintos y conviene saber de
    cuál viene cada uno.
    """

    # 1. Si no nos dan un partido explícito, intentamos detectarlo en la pregunta
    if partido is None:
        partido_detectado = _detectar_partido_en_pregunta(pregunta)
        # Solo usar el partido detectado si no se nombra a la mesa explícitamente y a otro partido a la vez
        if partido_detectado and "mesa" in pregunta.lower() and partido_detectado != "Mesa":
            partido = None
        else:
            partido = partido_detectado

    if partido is None:
        if "mesa" not in pregunta.lower():
            where = {"partido": {"$ne": "Mesa"}}
        else:
            where = None
    else:
        where = {"partido": {"$eq": partido}}

    # 3. Búsqueda semántica en toda la colección.
    #    Pedimos el doble de top_k para poder filtrar ruido y aun así
    #    quedarnos con top_k fragmentos útiles.
    resultados = collection.query(
        query_texts=[pregunta],
        n_results=top_k * 2,
        where=where,
        include=["documents", "metadatas"]
    )
    documentos = resultados["documents"][0]
    metadatos  = resultados["metadatas"][0]

    documentos, metadatos = _filtrar_ruido(documentos, metadatos)
    documentos = documentos[:top_k]
    metadatos  = metadatos[:top_k]

    if not documentos:
        filtro_txt = f" del partido '{partido}'" if partido else ""
        return {
            "pregunta":      pregunta,
            "prompt":        "",
            "respuesta_llm": f"No se encontraron fragmentos relevantes{filtro_txt} en ningún vídeo.",
            "fuentes_top_k": []
        }

    # Fusión de chunks contiguos (Mejora B)
    documentos, metadatos = _fusionar_chunks_contiguos(documentos, metadatos)

    # 4. Construir contexto (con título/fecha de vídeo, al venir de varias sesiones)
    contexto = _construir_contexto_transversal(documentos, metadatos)

    # 5. Historial propio de las búsquedas transversales, separado por partido
    clave_historial = f"__transversal__{partido or 'todos'}"
    historial_previo = obtener_historial(clave_historial)

    mensaje_usuario = (
        f"Pregunta del usuario: {pregunta}\n\n"
        f"A continuación tienes fragmentos relevantes recogidos de VARIOS PLENOS "
        f"distintos (cada fragmento indica entre corchetes de qué sesión y fecha "
        f"proviene). Ten en cuenta que pueden ser sesiones de fechas diferentes: "
        f"si es relevante para la pregunta, organiza la respuesta por sesión o "
        f"señala si la postura cambió con el tiempo.\n\n"
        f"Fragmentos:\n{contexto}"
    )

    mensajes = historial_previo + [{"role": "user", "content": mensaje_usuario}]

    # Inyectar entidades conocidas para auto-etiquetado XML
    from src.motor_busqueda.db_neo4j import get_all_entities_for_prompt
    texto_entidades = get_all_entities_for_prompt()

    system = (
        "Eres un asistente especializado en sesiones parlamentarias españolas. "
        "Responde ÚNICAMENTE con información que esté en los fragmentos proporcionados; "
        "si la respuesta no está en los fragmentos, dilo claramente en vez de inventarla. "
        "Los fragmentos proceden de VARIOS PLENOS distintos, cada uno identificado por su "
        "título y fecha entre corchetes al principio de la línea:\n"
        "- Cuando cites algo, indica también de qué sesión o fecha proviene, no solo quién lo dijo.\n"
        "- Si el mismo tema aparece en varias sesiones, señala si la postura se mantiene igual "
        "o ha cambiado con el tiempo.\n"
        "- CITA siempre por nombre y, si se conoce, por partido o grupo parlamentario "
        "(ej. 'Pérez Masó (Junts) defendió que...'), en vez de decir simplemente 'un diputado dijo...'.\n"
        "- ATRIBUCIÓN ESTRICTA: solo atribuye una afirmación o postura al partido o persona que "
        "la dijo literalmente en su propio fragmento. No infieras ni des por hecho que un partido "
        "'apoya' o 'está de acuerdo con' algo solo porque otro ponente lo mencionó.\n"
        "- SÉ CONCISO pero completo: estructura la respuesta por sesión o por postura cuando "
        "haya varios aspectos distintos, en vez de un único párrafo genérico.\n"
        "- Puedes usar el historial de la conversación para dar respuestas de seguimiento coherentes.\n\n"
        "INSTRUCCIÓN MUY IMPORTANTE (AUTO-ETIQUETADO XML):\n"
        "Cuando menciones a alguna de las personas, partidos, leyes, eventos o conceptos listados abajo en tu respuesta, "
        "DEBES envolver su nombre en una etiqueta XML <entidad id=\"...\">nombre</entidad> usando el ID "
        "exacto que se te proporciona. Esto es vital para el frontend. Solo hazlo con las entidades que "
        "aparezcan en esta lista. Si una entidad sale varias veces, etiquétala todas las veces.\n\n"
        f"{texto_entidades}"
    )

    # 6. Llamar a Mistral
    respuesta_llm = _llamar_mistral(system, mensajes)

    # 7. Guardar en el historial propio de la búsqueda transversal
    if clave_historial not in _historial:
        _historial[clave_historial] = []
    _historial[clave_historial].append({"role": "user",      "content": mensaje_usuario})
    _historial[clave_historial].append({"role": "assistant", "content": respuesta_llm})
    _recortar_historial(clave_historial)

    # 8. Construir fuentes, añadiendo de qué vídeo viene cada una
    fuentes_top_k = []
    for doc, meta in zip(documentos, metadatos):
        fuente = {**meta}
        ponente_mostrado = _nombre_mostrar(meta)
        fuente["ponente"] = ponente_mostrado
        fuente["nombre"] = ponente_mostrado.split(' (')[0]
        fuente["texto"] = doc
        fuente["enlace_video"] = meta.get("url_exacta_tiempo", "")
        fuente["inicio_str"] = _segundos_a_mmss(meta["inicio"])
        fuente["fin_str"] = _segundos_a_mmss(meta["fin"])
        fuente["inicio_segundos"] = meta["inicio"]
        fuente["fin_segundos"] = meta["fin"]
        fuente["inicio"] = _segundos_a_mmss(meta["inicio"])
        fuente["fin"] = _segundos_a_mmss(meta["fin"])
        fuente["video_id"] = meta.get("video_id", "")
        fuente["titulo_video"] = meta.get("titulo_video", "")
        fuentes_top_k.append(fuente)

    return {
        "pregunta":      pregunta,
        "prompt":        mensaje_usuario,
        "respuesta_llm": respuesta_llm,
        "fuentes_top_k": fuentes_top_k
    }


def limpiar_historial_transversal(partido: str = None) -> None:
    """
    Borra el historial de una conversación transversal concreta
    (la del partido indicado, o la general si no se indica ninguno).
    Útil para "empezar de cero" una búsqueda transversal en el frontend.
    """
    limpiar_historial(f"__transversal__{partido or 'todos'}")