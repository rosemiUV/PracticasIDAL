#!/usr/bin/env python3
"""
extractor_votaciones.py
Extracción robusta de votaciones parlamentarias.
Guarda los resultados individualmente por video en BPI/data/
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from bisect import bisect_right
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. Utilidades de bajo nivel: numeros en texto y limpieza
# ---------------------------------------------------------------------------

_PALABRAS_CERO = {"ninguno", "ninguna", "ningún", "ningun", "cero"}

_PALABRAS_NUMERO = {
    "un": 1, "uno": 1, "una": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciséis": 16, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50,
}

_NUM_TOKEN = (
    r"(?:\d{1,4}|ningun[oa]?s?|ningún|cero|"
    + "|".join(sorted(_PALABRAS_NUMERO, key=len, reverse=True))
    + r")"
)


def _token_a_entero(token: str) -> Optional[int]:
    if token is None:
        return None
    t = token.strip().lower()
    if t in _PALABRAS_CERO:
        return 0
    if t.isdigit():
        return int(t)
    if t in _PALABRAS_NUMERO:
        return _PALABRAS_NUMERO[t]
    return None


def _limpiar_espacios(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


# ---------------------------------------------------------------------------
# 2. Extraccion de metricas numericas
# ---------------------------------------------------------------------------

_PAT_KEYWORD = {
    "votos_emitidos": re.compile(r"votos?\s+emitidos?", re.IGNORECASE),
    "a_favor": re.compile(r"a\s+favor", re.IGNORECASE),
    "en_contra": re.compile(r"en\s+contra", re.IGNORECASE),
    "abstenciones": re.compile(r"abstenci[oó]n(?:es)?", re.IGNORECASE),
}
_PAT_NUMERO = re.compile(rf"\b{_NUM_TOKEN}\b", re.IGNORECASE)

_DISTANCIA_MAXIMA = 25


def _extraer_metricas_de_segmento(segmento: str) -> dict[str, Optional[int]]:
    eventos_kw_todos = []  
    for nombre, patron in _PAT_KEYWORD.items():
        for m in patron.finditer(segmento):
            eventos_kw_todos.append((m.start(), m.end(), nombre))

    resultado: dict[str, Optional[int]] = {k: None for k in _PAT_KEYWORD}
    if not eventos_kw_todos:
        return resultado

    inicio_resultados = min(e[0] for e in eventos_kw_todos)

    eventos_num = [
        [m.start(), m.end(), _token_a_entero(m.group())]
        for m in _PAT_NUMERO.finditer(segmento)
        if m.start() >= inicio_resultados
    ]
    eventos_num = [e for e in eventos_num if e[2] is not None]

    por_metrica: dict[str, list[tuple[int, int, str]]] = {}
    for ev in eventos_kw_todos:
        por_metrica.setdefault(ev[2], []).append(ev)

    eventos_kw = []
    UMBRAL_DUPLICADO = 20  

    for nombre, ocurrencias in por_metrica.items():
        ocurrencias.sort(key=lambda e: e[0])
        primera = ocurrencias[0]
        eventos_kw.append(primera)

        for duplicado in ocurrencias[1:]:
            kw_ini, kw_fin, _ = duplicado
            mejor_idx = None
            mejor_dist = None
            for idx, (num_ini, num_fin, _valor) in enumerate(eventos_num):
                if num_ini >= kw_fin:
                    dist = num_ini - kw_fin
                elif num_fin <= kw_ini:
                    dist = kw_ini - num_fin
                else:
                    continue
                if dist > UMBRAL_DUPLICADO:
                    continue
                if mejor_dist is None or dist < mejor_dist:
                    mejor_dist = dist
                    mejor_idx = idx
            if mejor_idx is not None:
                eventos_num[mejor_idx][2] = None  

    eventos_num = [e for e in eventos_num if e[2] is not None]
    numeros_disponibles = list(range(len(eventos_num)))  

    for kw_inicio, kw_fin, nombre_metrica in sorted(eventos_kw, key=lambda e: e[0]):
        mejor_idx = None
        mejor_distancia = None
        for idx in numeros_disponibles:
            num_inicio, num_fin, _valor = eventos_num[idx]
            if num_inicio >= kw_fin:
                distancia = num_inicio - kw_fin       
            elif num_fin <= kw_inicio:
                distancia = kw_inicio - num_fin        
            else:
                continue  
            if distancia > _DISTANCIA_MAXIMA:
                continue
            if mejor_distancia is None or distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_idx = idx

        if mejor_idx is not None:
            resultado[nombre_metrica] = eventos_num[mejor_idx][2]
            numeros_disponibles.remove(mejor_idx)

    return resultado


# ---------------------------------------------------------------------------
# 3. Segmentacion por votacion individual
# ---------------------------------------------------------------------------

_DISPARADORES = [
    r"Vamos\s+ahora\s+a\s+votar",
    r"Vamos\s+a\s+votar",
    r"Votamos\s+ahora",
    r"Votamos\s+separadamente(?:\s+por\s+puntos)?",
    r"Empezamos\s+votando",
    r"Empezamos\s+con\s+la\s+enmienda",
    r"Empezamos\s+la\s+votaci[oó]n",
    r"Pasamos\s+ahora\s+a\s+votar",
    r"Se\s+vota\s+en\s+sus\s+t[eé]rminos",
    r"Se\s+votan?\b",
    r"Procedemos\s+a\s+la\s+votaci[oó]n",
    r"Sometemos\s+a\s+votaci[oó]n",
    r"Votamos\s+conjuntamente",
    r"\bVotamos\s+el\b",
    r"\bVotamos\s+la\b",
    r"\bVotamos\s+los\b",
    r"\bVotamos\s+las\b",
]

_REGEX_DISPARADOR = re.compile(
    r"(?=\b(?:" + "|".join(_DISPARADORES) + r")\b)", flags=re.IGNORECASE
)

_REGEX_INICIO_RESULTADOS = re.compile(
    r"¿?\s*votos?\s+emitidos?\s*\??|¿?\s*(?:votos\s+)?a\s+favor\s*\??",
    flags=re.IGNORECASE,
)

_MULETILLAS_INICIALES = re.compile(
    r"^(?:ahora\s+|ya\s+|pues\s+)*"
    r"(?:la|el|los|las|de\s+la|del)?\s*",
    flags=re.IGNORECASE,
)


def _limpiar_objeto(objeto: str) -> str:
    objeto = _limpiar_espacios(objeto)
    objeto = re.sub(
        r"^(?:" + "|".join(_DISPARADORES) + r")\s*", "", objeto, flags=re.IGNORECASE
    )
    objeto = _MULETILLAS_INICIALES.sub("", objeto, count=1)
    objeto = re.sub(
        r"[,.]?\s*(?:empezamos|empecemos|vamos|comenzamos)[.,]?\s*$",
        "",
        objeto,
        flags=re.IGNORECASE,
    )
    objeto = objeto.strip(" ,.;:")
    if objeto:
        objeto = objeto[0].upper() + objeto[1:]
    return objeto


_ORACIONES_DE_TRAMITE = re.compile(
    r"^(?:"
    r"votamos(?:\s+ahora)?|"
    r"empezamos(?:\s+la\s+votaci[oó]n)?|"
    r"empecemos|"
    r"vamos\s+a\s+votar(?:\s+ahora)?|"
    r"vamos\s+ahora\s+a\s+votar|"
    r"se\s+vota[n]?(?:\s+en\s+sus\s+t[eé]rminos)?|"
    r"sometemos\s+a\s+votaci[oó]n|"
    r"procedemos\s+a\s+la\s+votaci[oó]n|"
    r"pasamos\s+ahora\s+a\s+votar"
    r")$",
    flags=re.IGNORECASE,
)


def _es_oracion_de_tramite(oracion: str) -> bool:
    limpio = oracion.strip(" ¿?¡!.,;:")
    if not limpio:
        return True
    return bool(_ORACIONES_DE_TRAMITE.match(limpio))


def _extraer_objeto_contextual(contexto: str) -> str:
    contexto = contexto.strip()
    if not contexto:
        return ""

    oraciones = re.split(r"(?<=[.!?])\s+", contexto)
    for oracion in reversed(oraciones):
        oracion = oracion.strip()
        if not oracion or _es_oracion_de_tramite(oracion):
            continue
        return _limpiar_objeto(oracion)

    return ""


_REGEX_RESOLUCION = re.compile(
    r"(no\s+se\s+aprueba|queda\s+aprobad[oa]|se\s+aprueba|no\s+se\s+admite|"
    r"se\s+admite|hay\s+empate|queda\s+abocad[oa]|"
    r"queda\s+rechazad[oa]|se\s+rechaza|"
    r"no\s+prospera|prospera|"
    r"queda\s+decaíd[oa]|decae|"
    r"se\s+retira|"
    r"queda\s+desestimad[oa]|se\s+desestima|"
    r"queda\s+convalidad[oa]|se\s+convalida|"
    r"queda\s+derogad[oa]|se\s+deroga)",
    flags=re.IGNORECASE,
)


def _extraer_resolucion(segmento: str) -> Optional[str]:
    m = _REGEX_RESOLUCION.search(segmento)
    return _limpiar_espacios(m.group(1)).lower() if m else None


# ---------------------------------------------------------------------------
# 4. Estructura de datos de salida
# ---------------------------------------------------------------------------

@dataclass
class Votacion:
    video_id: str
    fecha: str
    chunk_id_origen: str
    objeto: str
    votos_emitidos: Optional[int]
    a_favor: Optional[int]
    en_contra: Optional[int]
    abstenciones: Optional[int]
    resolucion: Optional[str]
    integridad_ok: Optional[bool]
    inicio: Optional[float]
    fin: Optional[float]
    url_exacta_tiempo: Optional[str]
    texto_segmento: str = field(repr=False)


def _validar_integridad(v: Votacion, tolerancia: int = 1) -> Optional[bool]:
    valores = [v.a_favor, v.en_contra, v.abstenciones]
    if v.votos_emitidos is None or any(x is None for x in valores):
        return None
    return abs(sum(valores) - v.votos_emitidos) <= tolerancia


# ---------------------------------------------------------------------------
# 5. Construccion del buffer continuo por video
# ---------------------------------------------------------------------------

@dataclass
class _OffsetChunk:
    inicio_offset: int
    chunk: dict


def _construir_buffer_mesa(chunks: list[dict]) -> tuple[str, list[_OffsetChunk]]:
    partes: list[str] = []
    offsets: list[_OffsetChunk] = []
    cursor = 0

    chunks_ordenados = sorted(
        chunks, key=lambda c: c.get("inicio", 0) or 0
    )

    for chunk in chunks_ordenados:
        try:
            partido = (chunk.get("partido") or "").strip().lower()
            texto = chunk.get("texto") or ""
            if not texto:
                continue
            if partido != "mesa":
                continue

            offsets.append(_OffsetChunk(inicio_offset=cursor, chunk=chunk))
            partes.append(texto)
            cursor += len(texto) + 1  
        except Exception:
            continue

    buffer_completo = " ".join(partes)
    return buffer_completo, offsets


def _chunk_para_offset(offset: int, offsets: list[_OffsetChunk]) -> Optional[dict]:
    if not offsets:
        return None
    puntos = [o.inicio_offset for o in offsets]
    idx = bisect_right(puntos, offset) - 1
    idx = max(0, min(idx, len(offsets) - 1))
    return offsets[idx].chunk


# ---------------------------------------------------------------------------
# 6. Pipeline principal de extraccion
# ---------------------------------------------------------------------------

def extraer_votaciones_de_video(chunks: list[dict]) -> list[Votacion]:
    resultados: list[Votacion] = []

    if not chunks:
        return resultados

    video_id = chunks[0].get("video_id", "desconocido")
    fecha = chunks[0].get("fecha_publicacion", "desconocida") 

    buffer_completo, offsets = _construir_buffer_mesa(chunks)
    if not buffer_completo:
        return resultados

    posiciones = [m.start() for m in _REGEX_DISPARADOR.finditer(buffer_completo)]

    _DISTANCIA_MAX_ANUNCIO = 300
    for m in re.finditer(r"¿?\s*votos?\s+emitidos?", buffer_completo, flags=re.IGNORECASE):
        pos = m.start()
        cubierta = any(0 <= pos - p <= _DISTANCIA_MAX_ANUNCIO for p in posiciones)
        if not cubierta:
            posiciones.append(pos)

    if not posiciones:
        return resultados

    posiciones = sorted(set(posiciones))
    posiciones.append(len(buffer_completo))  

    puntero_contexto = 0

    for i in range(len(posiciones) - 1):
        try:
            inicio_seg = posiciones[i]
            fin_seg = posiciones[i + 1]
            segmento = buffer_completo[inicio_seg:fin_seg]

            if not segmento.strip():
                continue

            m_resultado = _REGEX_INICIO_RESULTADOS.search(segmento)
            pos_resultados_abs = inicio_seg + (
                m_resultado.start() if m_resultado else len(segmento)
            )

            metricas = _extraer_metricas_de_segmento(segmento)
            votos_emitidos = metricas["votos_emitidos"]
            a_favor = metricas["a_favor"]
            en_contra = metricas["en_contra"]
            abstenciones = metricas["abstenciones"]
            resolucion_match = _REGEX_RESOLUCION.search(segmento)
            resolucion = (
                _limpiar_espacios(resolucion_match.group(1)).lower()
                if resolucion_match
                else None
            )

            if all(v is None for v in (votos_emitidos, a_favor, en_contra, abstenciones)):
                continue

            contexto = buffer_completo[puntero_contexto:pos_resultados_abs]
            objeto = _extraer_objeto_contextual(contexto)

            if resolucion_match:
                fin_resolucion_local = resolucion_match.end()
                m_punto_siguiente = re.search(
                    r"[.!?]", segmento[fin_resolucion_local:]
                )
                if m_punto_siguiente:
                    fin_resolucion_local += m_punto_siguiente.end()
                else:
                    fin_resolucion_local = len(segmento)
                puntero_contexto = inicio_seg + fin_resolucion_local
            else:
                puntero_contexto = fin_seg

            chunk_origen = _chunk_para_offset(inicio_seg, offsets) or {}

            votacion = Votacion(
                video_id=video_id,
                fecha=fecha,
                chunk_id_origen=chunk_origen.get("chunk_id", "desconocido"),
                objeto=objeto or "(objeto no identificado)",
                votos_emitidos=votos_emitidos,
                a_favor=a_favor,
                en_contra=en_contra,
                abstenciones=abstenciones,
                resolucion=resolucion,
                integridad_ok=None,
                inicio=chunk_origen.get("inicio"),
                fin=chunk_origen.get("fin"),
                url_exacta_tiempo=chunk_origen.get("url_exacta_tiempo"),
                texto_segmento=_limpiar_espacios(segmento)[:400],
            )
            votacion.integridad_ok = _validar_integridad(votacion)
            resultados.append(votacion)

        except Exception as exc:  
            print(
                f"  [!] Aviso: fallo al procesar un segmento de {video_id}: {exc}",
                file=sys.stderr,
            )
            continue

    return resultados


# ---------------------------------------------------------------------------
# 7. Analisis batch de una carpeta completa y guardado
# ---------------------------------------------------------------------------

def procesar_archivo(ruta: Path) -> list[Votacion]:
    try:
        with ruta.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[!] No se pudo leer {ruta.name}: {exc}", file=sys.stderr)
        return []

    if not isinstance(data, list):
        print(f"[!] Formato inesperado en {ruta.name} (se esperaba una lista)", file=sys.stderr)
        return []

    chunks_validos = []
    for c in data:
        try:
            if isinstance(c, dict) and c.get("texto"):
                chunks_validos.append(c)
        except Exception:
            continue

    try:
        return extraer_votaciones_de_video(chunks_validos)
    except Exception as exc:
        print(f"[!] Fallo procesando {ruta.name}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return []


def guardar_resultados(votaciones: list[Votacion], ruta_salida: Path) -> None:
    datos_formateados = []
    
    for v in votaciones:
        datos_formateados.append({
            "Video_id": v.video_id,
            "Fecha": v.fecha,
            "Tema votado": v.objeto,
            "Total": v.votos_emitidos,
            "A favor": v.a_favor,
            "En contra": v.en_contra,
            "Abstenciones": v.abstenciones,
            "Aprobado": v.resolucion.capitalize() if v.resolucion else "Desconocido",
            "Fragmento_video": v.url_exacta_tiempo
        })

    with ruta_salida.open("w", encoding="utf-8") as f:
        json.dump(datos_formateados, f, ensure_ascii=False, indent=2)


def _buscar_carpeta_automaticamente(nombre_objetivo: str = "resultados_finales") -> Optional[Path]:
    raiz = Path.cwd()
    try:
        for candidata in raiz.rglob(nombre_objetivo):
            if candidata.is_dir():
                return candidata
    except (OSError, PermissionError):
        pass
    return None

def _buscar_jsons_automaticamente(raiz: Optional[Path] = None) -> list[Path]:
    raiz = raiz or Path.cwd()
    encontrados: list[Path] = []
    try:
        for candidato in raiz.rglob("*.json"):
            try:
                with candidato.open(encoding="utf-8") as f:
                    muestra = json.load(f)
                if (
                    isinstance(muestra, list)
                    and muestra
                    and isinstance(muestra[0], dict)
                    and "texto" in muestra[0]
                    and "chunk_id" in muestra[0]
                ):
                    encontrados.append(candidato)
            except Exception:
                continue
    except (OSError, PermissionError):
        pass
    return encontrados


def procesar_carpeta(carpeta: str | Path = "resultados_finales") -> list[Votacion]:
    carpeta = Path(carpeta)
    todas: list[Votacion] = []
    archivos: list[Path] = []

    if carpeta.exists() and carpeta.is_dir():
        archivos = sorted(carpeta.glob("*.json"))

    if not archivos:
        encontrada = _buscar_carpeta_automaticamente(carpeta.name)
        if encontrada is not None:
            carpeta = encontrada
            archivos = sorted(carpeta.glob("*.json"))

    if not archivos:
        archivos = sorted(_buscar_jsons_automaticamente())

    if not archivos:
        print(f"[!] No se ha podido localizar ningun archivo .json valido a partir de '{Path.cwd()}'.", file=sys.stderr)
        return todas

    # -------------------------------------------------------------
    # NUEVA LÓGICA: Determinar carpeta BPI/data/
    # Buscamos 2 niveles arriba desde este script (src -> BPI)
    # -------------------------------------------------------------
    bpi_dir = Path(__file__).resolve().parents[2]
    data_dir = bpi_dir / "data" / "votaciones"
    
    # Creamos la carpeta data si no existe
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Los resultados se guardaran en: {data_dir}\n")

    for archivo in archivos:
        print(f"-> Procesando {archivo.name} ...")
        votaciones = procesar_archivo(archivo)
        
        if votaciones:
            # Obtener el ID del video del primer objeto detectado
            video_id = votaciones[0].video_id
            
            # Formar la ruta de salida
            ruta_salida = data_dir / f"votaciones_extraidas_{video_id}.json"
            
            # Guardar resultados
            guardar_resultados(votaciones, ruta_salida)
            print(f"   [+] {len(votaciones)} votacion(es) guardada(s) en {ruta_salida.name}")
            
            todas.extend(votaciones)
        else:
            print("   [-] No se detectaron votaciones. No se generará archivo para este vídeo.")

    return todas


# ---------------------------------------------------------------------------
# 8. Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "resultados_finales"
    
    print("\n=== INICIANDO EXTRACCIÓN DE VOTACIONES ===")
    procesar_carpeta(carpeta)
    print("=== PROCESO COMPLETADO ===\n")


if __name__ == "__main__":
    main()