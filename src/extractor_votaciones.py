#!/usr/bin/env python3
"""
extractor_votaciones.py
Extraccion robusta de votaciones parlamentarias a partir de chunks
transcritos/diarizados (formato "resultados_finales/*.json").

Estrategia general
-------------------
1. Los chunks de la Mesa (partido == "Mesa") suelen contener varias
   votaciones seguidas ("Votamos ahora...", "Empezamos votando...").
   Para evitar que una votacion quede cortada porque cae justo en el
   limite entre dos chunks consecutivos, se reconstruye un "buffer"
   continuo por video concatenando SOLO los chunks de la Mesa en su
   orden temporal, guardando el offset de cada chunk dentro del buffer.
2. Sobre ese buffer se segmenta con una regex de disparadores
   (lookahead) que identifica el inicio de cada votacion individual.
3. Para cada segmento se extrae:
     - el "objeto" (que se vota), limpiando muletillas iniciales,
     - los resultados numericos (votos emitidos, a favor, en contra,
       abstenciones), tolerando ordenes distintos y "ninguno/ninguna"
       como sinonimo de 0, y evitando confundir numeros de leyes /
       articulos / anios con resultados de la votacion.
4. Se valida la integridad (favor + contra + abstenciones == emitidos)
   y se recupera, para cada votacion, el chunk_id / timestamp / url
   de origen mediante el offset dentro del buffer.
5. Todo el pipeline es defensivo: campos ausentes o filas corruptas no
   detienen el analisis batch de la carpeta.
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

# La transcripcion automatica a veces escribe "ninguno/ninguna/ningun" en
# lugar de "0". Tambien aparecen numeros escritos con palabras en casos
# sueltos ("cero", "un", "una") que tratamos igual por robustez.
_PALABRAS_CERO = {"ninguno", "ninguna", "ningún", "ningun", "cero"}

# La Mesa a veces dicta cifras pequenas con palabras en lugar de digitos
# (p.ej. "seis abstenciones"). Se cubre el rango habitual 0-20 mas las
# decenas redondas mas frecuentes en resultados de votacion.
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
    """Convierte un token numerico, una palabra-cero o un numero en palabras a int."""
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
# 2. Extraccion de metricas numericas (a favor / en contra / abstenciones /
#    votos emitidos) de forma bidireccional y tolerante al desorden del ASR.
# ---------------------------------------------------------------------------

# Enfoque: en lugar de una regex "forward/backward" por metrica (fragil
# cuando el orden se invierte de forma ambigua, p.ej. "ninguno en contra,
# 166 abstenciones", donde un numero cercano podria asociarse por error a
# la metrica vecina), se tokeniza el segmento en una secuencia ordenada de
# eventos [PALABRA_CLAVE_METRICA, NUMERO] y se empareja cada palabra clave
# con el NUMERO MAS CERCANO (por distancia en caracteres) que aun no haya
# sido consumido por otra metrica. Esto refleja como habla la Mesa: cada
# resultado numerico casi siempre esta pegado a su etiqueta, sea delante o
# detras, y nunca se reutiliza el mismo numero para dos metricas distintas.

_PAT_KEYWORD = {
    "votos_emitidos": re.compile(r"votos?\s+emitidos?", re.IGNORECASE),
    "a_favor": re.compile(r"a\s+favor", re.IGNORECASE),
    "en_contra": re.compile(r"en\s+contra", re.IGNORECASE),
    "abstenciones": re.compile(r"abstenci[oó]n(?:es)?", re.IGNORECASE),
}
_PAT_NUMERO = re.compile(rf"\b{_NUM_TOKEN}\b", re.IGNORECASE)

# Distancia maxima (en caracteres) entre una palabra clave y su numero para
# considerarlos asociados. Evita enlazar con un numero de ley/articulo/anio
# que quedo lejos, en otra frase.
_DISTANCIA_MAXIMA = 25


def _extraer_metricas_de_segmento(segmento: str) -> dict[str, Optional[int]]:
    """
    Devuelve {'votos_emitidos': int|None, 'a_favor': int|None,
    'en_contra': int|None, 'abstenciones': int|None} para un segmento de
    votacion, emparejando cada palabra clave con el numero disponible mas
    cercano (delante o detras), sin reutilizar numeros entre metricas.
    """
    eventos_kw_todos = []  # (inicio, fin, nombre_metrica)
    for nombre, patron in _PAT_KEYWORD.items():
        for m in patron.finditer(segmento):
            eventos_kw_todos.append((m.start(), m.end(), nombre))

    resultado: dict[str, Optional[int]] = {k: None for k in _PAT_KEYWORD}
    if not eventos_kw_todos:
        return resultado

    # El ASR a veces tartamudea y repite una palabra clave (p.ej. "338 a
    # favor, 338 a favor" o "a favor, ninguno a favor"). Cada metrica solo
    # deberia aparecer una vez por votacion, asi que nos quedamos unicamente
    # con su PRIMERA aparicion; las repeticiones se descartan para que no
    # "roben" por error el numero de una metrica distinta.
    primera_aparicion: dict[str, tuple[int, int, str]] = {}
    for evento in eventos_kw_todos:
        nombre = evento[2]
        if nombre not in primera_aparicion or evento[0] < primera_aparicion[nombre][0]:
            primera_aparicion[nombre] = evento
    eventos_kw = list(primera_aparicion.values())

    # Solo se consideran numeros a partir del inicio de la seccion de
    # resultados (la primera palabra clave detectada), para no confundir
    # numeros del "objeto" de la votacion (p.ej. "enmienda numero 33",
    # "punto 5") con resultados numericos de la votacion.
    inicio_resultados = min(e[0] for e in eventos_kw)

    eventos_num = [
        (m.start(), m.end(), _token_a_entero(m.group()))
        for m in _PAT_NUMERO.finditer(segmento)
        if m.start() >= inicio_resultados
    ]
    eventos_num = [e for e in eventos_num if e[2] is not None]

    numeros_disponibles = list(range(len(eventos_num)))  # indices no consumidos

    # Se procesan las palabras clave en el orden en que aparecen en el texto.
    for kw_inicio, kw_fin, nombre_metrica in sorted(eventos_kw, key=lambda e: e[0]):
        mejor_idx = None
        mejor_distancia = None
        for idx in numeros_disponibles:
            num_inicio, num_fin, _valor = eventos_num[idx]
            if num_inicio >= kw_fin:
                distancia = num_inicio - kw_fin       # numero DESPUES de la palabra clave
            elif num_fin <= kw_inicio:
                distancia = kw_inicio - num_fin        # numero ANTES de la palabra clave
            else:
                continue  # solapamiento inesperado, se ignora
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

# Frases disparadoras que la Mesa usa habitualmente para anunciar el inicio
# de CADA votacion individual dentro de un bloque largo. El lookahead
# permite usar re.split conservando el disparador al inicio del segmento
# siguiente.
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
    r"\bVotamos\s+el\b",
    r"\bVotamos\s+la\b",
    r"\bVotamos\s+los\b",
    r"\bVotamos\s+las\b",
    r"Procedemos\s+a\s+la\s+votaci[oó]n",
    r"Sometemos\s+a\s+votaci[oó]n",
    r"Votamos\s+conjuntamente"
]

_REGEX_DISPARADOR = re.compile(
    r"(?=\b(?:" + "|".join(_DISPARADORES) + r")\b)", flags=re.IGNORECASE
)

# Palabras clave que marcan el inicio de la seccion de resultados dentro de
# un segmento; se usan para saber donde CORTAR el "objeto" de la votacion,
# de forma que nunca se cuelen los numeros del resultado en el titulo.
_REGEX_INICIO_RESULTADOS = re.compile(
    r"¿?\s*votos?\s+emitidos?\s*\??|¿?\s*a\s+favor\s*\??", flags=re.IGNORECASE
)

# Muletillas/articulos iniciales que se deben limpiar del objeto extraido,
# p.ej. "ahora la proposicion..." -> "proposicion...".
_MULETILLAS_INICIALES = re.compile(
    r"^(?:ahora\s+|ya\s+|pues\s+)*"
    r"(?:la|el|los|las|de\s+la|del)?\s*",
    flags=re.IGNORECASE,
)


def _limpiar_objeto(objeto: str) -> str:
    objeto = _limpiar_espacios(objeto)
    # quitar disparador inicial (verbo) si quedo pegado, p.ej. "votamos ahora"
    objeto = re.sub(
        r"^(?:" + "|".join(_DISPARADORES) + r")\s*", "", objeto, flags=re.IGNORECASE
    )
    objeto = _MULETILLAS_INICIALES.sub("", objeto, count=1)
    # Muletillas finales frecuentes que el ASR pega justo antes de los
    # resultados numericos (p.ej. "...punto 5 empezamos" -> "...punto 5").
    objeto = re.sub(
        r"\s*(?:empezamos|empecemos|vamos|comenzamos)\s*$",
        "",
        objeto,
        flags=re.IGNORECASE,
    )
    objeto = objeto.strip(" ,.;:")
    if objeto:
        objeto = objeto[0].upper() + objeto[1:]
    return objeto


def _extraer_objeto(segmento: str) -> str:
    """
    Aisla la descripcion de lo que se vota: desde el inicio del segmento
    (justo despues del disparador) hasta el comienzo de los resultados
    numericos o el primer punto, lo que ocurra antes.
    """
    m_resultado = _REGEX_INICIO_RESULTADOS.search(segmento)
    limite = m_resultado.start() if m_resultado else len(segmento)

    candidato = segmento[:limite]

    # Si hay un punto final claro antes del limite de resultados, usarlo
    # (evita arrastrar frases posteriores no relacionadas con el objeto).
    m_punto = re.search(r"[.!]", candidato)
    if m_punto and m_punto.start() > 3:
        candidato = candidato[: m_punto.start()]

    return _limpiar_objeto(candidato)


_REGEX_RESOLUCION = re.compile(
    r"(no\s+se\s+aprueba|queda\s+aprobad[oa]|se\s+aprueba|no\s+se\s+admite|"
    r"se\s+admite|hay\s+empate|queda\s+abocad[oa])",
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
    """
    Compara la suma de favor+contra+abstenciones contra 'votos emitidos'.
    Devuelve None si no hay datos suficientes para validar (no se detiene
    el proceso, simplemente se marca como no verificable).
    """
    valores = [v.a_favor, v.en_contra, v.abstenciones]
    if v.votos_emitidos is None or any(x is None for x in valores):
        return None
    return abs(sum(valores) - v.votos_emitidos) <= tolerancia


# ---------------------------------------------------------------------------
# 5. Construccion del buffer continuo por video (chunks de la Mesa)
# ---------------------------------------------------------------------------

@dataclass
class _OffsetChunk:
    inicio_offset: int
    chunk: dict


def _construir_buffer_mesa(chunks: list[dict]) -> tuple[str, list[_OffsetChunk]]:
    """
    Concatena, en orden, el texto de los chunks cuyo 'partido' identifica a
    la Mesa (o cuyo 'nombre' corresponde a la Presidencia), preservando el
    offset de inicio de cada chunk dentro del buffer resultante. Esto evita
    perder votaciones que quedan partidas justo en el limite entre dos
    chunks consecutivos.
    """
    partes: list[str] = []
    offsets: list[_OffsetChunk] = []
    cursor = 0

    # Se ordena defensivamente por 'inicio' si esta disponible.
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
            cursor += len(texto) + 1  # +1 por el separador que anadimos
        except Exception:
            # Un chunk corrupto no debe tirar abajo la construccion del buffer.
            continue

    buffer_completo = " ".join(partes)
    return buffer_completo, offsets


def _chunk_para_offset(offset: int, offsets: list[_OffsetChunk]) -> Optional[dict]:
    """Localiza, via busqueda binaria, el chunk de origen de una posicion del buffer."""
    if not offsets:
        return None
    puntos = [o.inicio_offset for o in offsets]
    idx = bisect_right(puntos, offset) - 1
    idx = max(0, min(idx, len(offsets) - 1))
    return offsets[idx].chunk


# ---------------------------------------------------------------------------
# 6. Pipeline principal de extraccion para un video (lista de chunks)
# ---------------------------------------------------------------------------

def extraer_votaciones_de_video(chunks: list[dict]) -> list[Votacion]:
    resultados: list[Votacion] = []

    if not chunks:
        return resultados

    video_id = chunks[0].get("video_id", "desconocido")

    buffer_completo, offsets = _construir_buffer_mesa(chunks)
    if not buffer_completo:
        return resultados

    # Posiciones de cada disparador conocido dentro del buffer.
    posiciones = [m.start() for m in _REGEX_DISPARADOR.finditer(buffer_completo)]

    # Red de seguridad: si aparece un bloque de resultados ("votos
    # emitidos") que NO viene precedido, a una distancia razonable, por
    # ninguno de los disparadores conocidos, es señal de que el anuncio
    # de esa votacion se hizo con una frase que nuestra lista no cubre.
    # En ese caso se anade esa posicion como punto de corte adicional para
    # no fusionar dos votaciones distintas dentro de un mismo segmento
    # (aunque el "objeto" de esa votacion huerfana quede sin identificar).
    _DISTANCIA_MAX_ANUNCIO = 300
    for m in re.finditer(r"¿?\s*votos?\s+emitidos?", buffer_completo, flags=re.IGNORECASE):
        pos = m.start()
        cubierta = any(0 <= pos - p <= _DISTANCIA_MAX_ANUNCIO for p in posiciones)
        if not cubierta:
            posiciones.append(pos)

    if not posiciones:
        return resultados

    posiciones = sorted(set(posiciones))
    posiciones.append(len(buffer_completo))  # centinela final

    for i in range(len(posiciones) - 1):
        try:
            inicio_seg = posiciones[i]
            fin_seg = posiciones[i + 1]
            segmento = buffer_completo[inicio_seg:fin_seg]

            if not segmento.strip():
                continue

            objeto = _extraer_objeto(segmento)
            metricas = _extraer_metricas_de_segmento(segmento)
            votos_emitidos = metricas["votos_emitidos"]
            a_favor = metricas["a_favor"]
            en_contra = metricas["en_contra"]
            abstenciones = metricas["abstenciones"]
            resolucion = _extraer_resolucion(segmento)

            # Si no se detecto ningun resultado numerico, lo mas probable
            # es que el disparador no correspondiera a una votacion real
            # (ruido/falso positivo) -> se descarta el segmento.
            if all(v is None for v in (votos_emitidos, a_favor, en_contra, abstenciones)):
                continue

            chunk_origen = _chunk_para_offset(inicio_seg, offsets) or {}

            votacion = Votacion(
                video_id=video_id,
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

        except Exception as exc:  # programacion defensiva: nunca frenar el batch
            print(
                f"  [!] Aviso: fallo al procesar un segmento de {video_id}: {exc}",
                file=sys.stderr,
            )
            continue

    return resultados


# ---------------------------------------------------------------------------
# 7. Analisis batch de una carpeta completa
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

    # Los chunks validos deben tener al menos 'texto'; se descartan sin
    # detener el resto del archivo.
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


def procesar_carpeta(carpeta: str | Path = "resultados_finales") -> list[Votacion]:
    carpeta = Path(carpeta)
    todas: list[Votacion] = []

    if not carpeta.exists():
        print(f"[!] La carpeta '{carpeta}' no existe.", file=sys.stderr)
        return todas

    archivos = sorted(carpeta.glob("*.json"))
    if not archivos:
        print(f"[!] No se encontraron archivos .json en '{carpeta}'.", file=sys.stderr)
        return todas

    for archivo in archivos:
        print(f"-> Procesando {archivo.name} ...")
        votaciones = procesar_archivo(archivo)
        print(f"   {len(votaciones)} votacion(es) detectada(s).")
        todas.extend(votaciones)

    return todas


def guardar_resultados(votaciones: list[Votacion], salida: str | Path = "votaciones_extraidas.json") -> None:
    salida = Path(salida)
    datos = [asdict(v) for v in votaciones]
    with salida.open("w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado: {salida} ({len(votaciones)} votaciones)")


# ---------------------------------------------------------------------------
# 8. Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "resultados_finales"
    votaciones = procesar_carpeta(carpeta)

    print("\n=== RESUMEN ===")
    for v in votaciones:
        minutos = int(v.inicio // 60)
        horas = int(minutos // 60)
        minutos = minutos % 60
        segundos = int(v.inicio % 60)
        timestamp = f"{horas:02d}:{minutos:02d}:{segundos:02d}"
        estado_integridad = (
            "OK" if v.integridad_ok else ("SIN VERIFICAR" if v.integridad_ok is None else "INCONSISTENTE")
        )
        print(
            f"[{v.video_id} | {v.chunk_id_origen}] {v.objeto} [{timestamp}]\n"
            f"    Emitidos={v.votos_emitidos} A favor={v.a_favor} "
            f"En contra={v.en_contra} Abstenciones={v.abstenciones} "
            f"-> {v.resolucion or '¿?'} (integridad: {estado_integridad})"
        )

    guardar_resultados(votaciones)


if __name__ == "__main__":
    main()
