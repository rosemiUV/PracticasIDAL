"""
fusionador_pruebaV2.py
Módulo de partición y estructuración de texto (Chunking) para RAG.

Este script se encarga de coger la transcripción "bruta" generada por WhisperX 
(que viene separada por frases y oradores) y transformarla en bloques de texto 
perfectos (chunks) para que el buscador semántico pueda procesarlos.

Sus objetivos principales son:
1. Agrupar todo lo que dice una misma persona del tirón para no perder el contexto.
2. Trocear los discursos muy largos en fragmentos más pequeños (ej. máximo 100 palabras) 
   para no saturar la memoria del modelo de lenguaje (LLM).
3. Garantizar de forma estricta que un mismo fragmento NUNCA mezcle lo que dicen dos 
   personas distintas.
4. Empaquetar cada fragmento con todos sus metadatos (enlace al minuto exacto 
   del vídeo, nombre del orador, duración, etc.) y guardarlo en un JSON listo 
   para subir a la base de datos vectorial (ChromaDB).
"""

import json
from pathlib import Path


def _palabras_de_segmento(seg: dict) -> list:
    """
    Extrae lista de {texto, inicio, fin} de un segmento de WhisperX.
    Usa timestamps por palabra si están disponibles; si no, los distribuye linealmente.
    """
    words = seg.get("words", [])
    if words:
        resultado = []
        for w in words:
            # Algunos tokens de WhisperX no tienen start/end (puntuación, etc.)
            resultado.append({
                "texto": w.get("word", ""),
                "inicio": w.get("start", seg["start"]),
                "fin": w.get("end", seg["end"]),
            })
        return resultado

    # Fallback: distribuir linealmente por el texto del segmento
    palabras = seg["text"].strip().split()
    if not palabras:
        return []
    dur_por_palabra = (seg["end"] - seg["start"]) / len(palabras)
    return [
        {
            "texto": p,
            "inicio": seg["start"] + i * dur_por_palabra,
            "fin": seg["start"] + (i + 1) * dur_por_palabra,
        }
        for i, p in enumerate(palabras)
    ]


def fusionar_datos_para_rag(
    segmentos: list,
    video_id: str,
    url_video: str,
    titulo_video: str,
    fecha_publicacion: str,
    ruta_guardado: Path,
    max_palabras: int = 100,
) -> list:
    """
    Convierte segmentos de WhisperX (con campo 'speaker') en chunks para RAG.
    Garantía: ningún chunk mezcla speakers.
    Si un bloque de un mismo speaker supera max_palabras, se divide en sub-chunks.
    """

    # --- Paso 1: agrupar segmentos consecutivos del mismo speaker ---
    bloques = []  # [{ponente, palabras: [{texto, inicio, fin}]}]
    for seg in segmentos:
        ponente = seg.get("speaker", "DESCONOCIDO")
        palabras = _palabras_de_segmento(seg)
        if not palabras:
            continue
        if bloques and bloques[-1]["ponente"] == ponente:
            bloques[-1]["palabras"].extend(palabras)
        else:
            bloques.append({"ponente": ponente, "palabras": palabras})

    # --- Paso 2: dividir bloques largos en chunks de max_palabras ---
    chunks = []
    for bloque in bloques:
        todas = bloque["palabras"]
        for i in range(0, len(todas), max_palabras):
            ventana = todas[i : i + max_palabras]
            if not ventana:
                continue

            texto = " ".join(p["texto"] for p in ventana).strip()
            inicio = ventana[0]["inicio"]
            fin = ventana[-1]["fin"]
            chunk_id = f"{video_id}_{int(inicio)}_{int(fin)}"

            chunks.append({
                "chunk_id": chunk_id,
                "video_id": video_id,
                "titulo_video": titulo_video,
                "fecha_publicacion": fecha_publicacion,
                "url_video": url_video,
                "url_exacta_tiempo": f"{url_video}&t={int(inicio)}s",
                "ponente": bloque["ponente"],
                "inicio": round(inicio, 3),
                "fin": round(fin, 3),
                "duracion": round(fin - inicio, 3),
                "texto": texto,
            })

    ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_guardado, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return chunks
