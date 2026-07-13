"""
identificador_comisiones.py
Extensión del pipeline BPI para sesiones de Comisión (no Pleno).

No duplica el diccionario de diputados, el fingerprinting, la fusión ni
LLM-DESC/LLM-DICT — todo eso se reutiliza importando identificador_speakers.py
como módulo. Solo se implementa lo que genuinamente cambia en comisiones:

  1. Portavoces presentados SIN "tiene la palabra":
     "Por el Grupo Parlamentario X, señor/a Y." / "Portavoz de X, el señor Y."
  2. Figura compareciente/ministro/defensor del pueblo — habla muchas veces
     en bloques no contiguos, casi nunca por nombre completo repetido, y NO
     tiene partido parlamentario (se le asigna un partido-etiqueta: "Gobierno",
     "Defensor del Pueblo", "Compareciente externo").
  3. Patrones de Mesa propios de comisión (recuento de votos a mano alzada,
     "suspendemos la comisión", etc.) — IMPORTANTE: en comisión NO se asume
     que preside Francina Armengol (eso es solo para el Pleno); el presidente
     de una comisión es normalmente otro diputado distinto y no lo conocemos
     de antemano, así que se etiqueta partido="Mesa" con nombre=None.

REQUIERE: identificador_speakers.py en la misma carpeta o en el PYTHONPATH.
Si tu fichero principal tiene otro nombre, ajusta la línea de import de abajo.

Uso:
    python identificador_comisiones.py ruta/al/chunks.json
    python identificador_comisiones.py ruta/al/chunks.json --sin-llm

    # Detección automática pleno/comisión por título (recomendado para el
    # watcher del pipeline, en vez de llamar a este script o al de pleno
    # a mano):
    from identificador_comisiones import identificar_video_auto
    identificar_video_auto(ruta_json)
"""
import re
import sys
from pathlib import Path

from src.transcriptor_diarizador import identificador_speakers as pleno

# =============================================================
# PATRONES PROPIOS DE COMISIÓN
# =============================================================

# Portavoces presentados sin "tiene la palabra" — patrón dominante en
# comisiones frente al de pleno. (confianza, patrón)
PATRONES_PRESENTACION_COMISION = [
    (r"por el grupo(?:\s+parlamentario)?\s+[\wÀ-ÿ\s]+?,\s*(?:el|la\s+)?se[ñn]or[a]?\s+"
     r"([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+(?:\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+){0,3})", 0.85),
    (r"portavoz de\s+[\wÀ-ÿ\s]+?,\s*(?:el|la\s+)?se[ñn]or[a]?\s+"
     r"([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+(?:\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+){0,3})", 0.85),
    (r"tiene\s+palabras?\s+(?:el|la)\s+se[ñn]or[a]?\s+"
     r"([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+(?:\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+){0,3})", 0.80),
]

# Mesa de comisión: procedimiento, sin nombre propio implicado en el patrón.
PATRONES_MESA_COMISION = [
    r"(?:pueden\s+)?levant(?:ar|en)\s+la\s+mano",
    r"\bvotos?\s+a\s+favor\b",
    r"cu[aá]ntos\s+(?:son|est[aá]n)",
    r"suspendemos\s+la\s+comisi[oó]n",
    r"vamos\s+a\s+dar\s+comienzo\s+a\s+la\s+sesi[oó]n",
    r"queda\s+aprobada\s+la\s+proposici[oó]n",
]

# Figura compareciente / ministro / defensor del pueblo -> (partido-etiqueta,
# nombre-etiqueta genérico). El nombre genérico existe para que SÍ se pueda
# fijar el fingerprint (actualizar_fingerprint exige nombre no nulo) y así la
# identidad se propague por todos sus chunks, aunque no tengamos su nombre
# propio — si más adelante aparece su nombre real, identificar_video_comision
# lo reconcilia al final (ver _reconciliar_nombres_genericos).
ROLES_EXTERNOS = {
    "ministro": ("Gobierno", "Ministro"),
    "ministra": ("Gobierno", "Ministra"),
    "secretario de estado": ("Gobierno", "Secretario de Estado"),
    "secretaria de estado": ("Gobierno", "Secretaria de Estado"),
    "defensor del pueblo": ("Defensor del Pueblo", "Defensor del Pueblo"),
    "defensora del pueblo": ("Defensor del Pueblo", "Defensora del Pueblo"),
    "compareciente": ("Compareciente externo", "Compareciente"),
}
GENERIC_ROLE_NAMES = {n for _, n in ROLES_EXTERNOS.values()}
# Protege estas etiquetas de rol de LLM-DICT (pleno.py): son intencionales,
# no nombres deformados — sin esto, LLM-DICT intenta "corregirlas" y puede
# fusionar al Defensor del Pueblo con un diputado real cualquiera (bug real
# detectado: "Defensor del Pueblo" → "Ángel Ibáñez Hernando" (PP), 103 chunks).
pleno.NOMBRES_PLACEHOLDER_NO_CORREGIR.update(GENERIC_ROLE_NAMES)
_PATRON_ROL_EXTERNO = re.compile(
    r"\b(ministro|ministra|secretari[oa] de estado|defensor(?:a)? del pueblo|compareciente)\b",
    re.IGNORECASE,
)
_PATRON_NOMBRE_CON_TITULO = re.compile(
    r"\b(?:don|do[ñn]a)\s+([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+(?:\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑñ'\-]+){0,3})",
)
# IMPORTANTE: estos patrones de introducción del rol se buscan SOLO en el
# chunk ANTERIOR (nunca en el propio chunk del hablante actual). Un portavoz
# que dice "gracias, señor ministro" dentro de SU PROPIA intervención no debe
# heredar el rol — solo cuenta cuando es la MESA, en el turno previo, dándole
# la palabra al titular del rol.
_PATRON_ROL_INTRODUCCION = re.compile(
    r"(?:tiene\s+la\s+palabra|damos\s+la\s+palabra|vuelve\s+a\s+tomar\s+la\s+palabra)"
    r".{0,40}?(ministro|ministra|secretari[oa] de estado|defensor(?:a)? del pueblo|compareciente)",
    re.IGNORECASE,
)
_PATRON_ROL_ANUNCIO = re.compile(
    r"(?:sesi[oó]n\s+de\s+)?comparecencia\s+(?:del|de\s+la)\s+"
    r"(ministro|ministra|secretari[oa] de estado|defensor(?:a)? del pueblo)",
    re.IGNORECASE,
)
_PATRON_ROL_CONTESTAR = re.compile(r"contestar\s+a\s+(?:los\s+diferentes\s+)?portavoces", re.IGNORECASE)


def detectar_patrones_heuristicos_comision(chunks: list, idx: int) -> dict:
    """Heurística para sesiones de Comisión. Prueba primero los patrones
    específicos de comisión; si ninguno encaja, delega en la heurística de
    pleno (autopresentación, cierre "muchas gracias, señor X", etc., que son
    iguales en ambos contextos)."""
    texto_actual = chunks[idx]["texto"]
    textos = []
    if idx > 0:
        textos.append(chunks[idx - 1]["texto"])
    textos.append(texto_actual)
    ventana = " ".join(textos)

    # ── 1. Mesa de comisión (procedimiento) ─────────────────────────────
    for patron in pleno.PATRONES_PRESIDENCIA_MESA:
        if re.search(patron, texto_actual, re.IGNORECASE):
            return {
                "nombre": None,  # NO asumir que preside Armengol: en comisión
                "partido": "Mesa",  # preside otro diputado, no lo sabemos a priori
                "confianza": 0.90,
                "fragmento_detectado": f"[presidencia_mesa_comision] {patron}",
                "es_presentacion": False,
                "nombre_es_apellido_unico": False,
            }
    for patron in PATRONES_MESA_COMISION:
        if re.search(patron, texto_actual, re.IGNORECASE):
            return {
                "nombre": None,
                "partido": "Mesa",
                "confianza": 0.85,
                "fragmento_detectado": f"[mesa_comision] {patron}",
                "es_presentacion": False,
                "nombre_es_apellido_unico": False,
            }

    # ── 2. Figura compareciente / ministro / defensor del pueblo ────────
    # Solo se dispara si el chunk ANTERIOR (la Mesa) da la palabra al rol
    # ahora mismo — nunca por menciones del rol dentro del propio chunk del
    # hablante actual (evita que un portavoz que dice "gracias, ministro"
    # se quede él mismo etiquetado como el ministro).
    if idx > 0:
        texto_anterior = chunks[idx - 1]["texto"]
        m_intro = _PATRON_ROL_INTRODUCCION.search(texto_anterior)
        m_anuncio = _PATRON_ROL_ANUNCIO.search(texto_anterior)
        m_contestar = _PATRON_ROL_CONTESTAR.search(texto_anterior)
        rol_capturado = None
        fragmento = None
        if m_intro:
            rol_capturado, fragmento = m_intro.group(1), m_intro.group(0)
        elif m_anuncio:
            rol_capturado, fragmento = m_anuncio.group(1), m_anuncio.group(0)
        elif m_contestar:
            m_rol_generico = _PATRON_ROL_EXTERNO.search(texto_anterior)
            if m_rol_generico:
                rol_capturado, fragmento = m_rol_generico.group(1), m_contestar.group(0)

        if not rol_capturado:
            # Respaldo más amplio: si el chunk anterior menciona un rol
            # externo en CUALQUIER parte (no pegado a un verbo concreto de
            # dar la palabra) Y también nombra a alguien con "don/doña" en
            # ese mismo chunk, se trata como introducción. Cubre frases
            # reales que no encajaban arriba, como "por respeto también a
            # la compareciente... vamos a dar la palabra a doña Raquel
            # Montón" o "nuestra compareciente, doña Beatriz Jordi
            # Aguirre" — ninguna de las dos usa "tiene/damos la palabra"
            # pegado al rol, así que antes se perdía el nombre completo
            # real y solo quedaba el apellido vía cierre posterior.
            m_rol_amplio = _PATRON_ROL_EXTERNO.search(texto_anterior)
            m_nombre_amplio = _PATRON_NOMBRE_CON_TITULO.search(texto_anterior)
            if m_rol_amplio and m_nombre_amplio:
                rol_capturado = m_rol_amplio.group(1)
                fragmento = f"{m_rol_amplio.group(0)} ... {m_nombre_amplio.group(0)}"

        if rol_capturado:
            rol_norm = re.sub(r"\s+", " ", pleno._quitar_tildes(rol_capturado.lower()))
            partido_rol = nombre_rol_generico = None
            for clave, (etiqueta_partido, etiqueta_nombre) in ROLES_EXTERNOS.items():
                clave_norm = pleno._quitar_tildes(clave)
                if clave_norm in rol_norm or rol_norm in clave_norm:
                    partido_rol = etiqueta_partido
                    nombre_rol_generico = etiqueta_nombre
                    break
            if partido_rol:
                # Busca el nombre propio ("don/doña X") en el mismo chunk
                # anterior o en el actual (ambos son razonables aquí).
                m_nombre = _PATRON_NOMBRE_CON_TITULO.search(texto_anterior + " " + texto_actual)
                nombre_propio = None
                if m_nombre:
                    nombre_raw = m_nombre.group(1).strip()
                    nombre_dic, _ = pleno.normalizar_nombre(nombre_raw, solo_completos=False)
                    nombre_propio = nombre_dic or nombre_raw
                nombre_final = nombre_propio or nombre_rol_generico
                confianza = 0.95 if nombre_propio else 0.92
                return {
                    "nombre": nombre_final,
                    "nombre_raw": nombre_propio or nombre_rol_generico,
                    "partido": partido_rol,
                    "confianza": confianza,
                    "fragmento_detectado": fragmento[:120],
                    "es_presentacion": True,
                    "nombre_es_apellido_unico": False,
                }

    # ── 3. Portavoces sin "tiene la palabra" ────────────────────────────
    for patron, confianza in PATRONES_PRESENTACION_COMISION:
        m = re.search(patron, ventana, re.IGNORECASE)
        if m:
            nombre_raw_capturado = m.group(1).strip()
            nombre_encontrado, es_apellido_unico = pleno.normalizar_nombre(
                nombre_raw_capturado, solo_completos=False
            )
            if nombre_encontrado is None:
                continue  # cargo/palabra suelta capturada por error, no un nombre real
            partido_encontrado = pleno._partido_desde_texto_mesa(ventana)
            return {
                "nombre": nombre_encontrado,
                "nombre_raw": nombre_raw_capturado,
                "partido": partido_encontrado,
                "confianza": confianza,
                "fragmento_detectado": m.group(0)[:120],
                "es_presentacion": True,
                "nombre_es_apellido_unico": es_apellido_unico,
            }

    # ── 4. Todo lo demás (autopresentación, cierre, "solo partido"...) ──
    # es igual que en pleno, así que delegamos en su heurística.
    return pleno.detectar_patrones_heuristicos(chunks, idx)


# =============================================================
# DETECCIÓN AUTOMÁTICA PLENO / COMISIÓN
# =============================================================

def es_comision(chunks: list) -> bool:
    """Detecta si el vídeo es una sesión de Comisión mirando titulo_video.
    Cubre 'Comisión de...' y 'Comisión Mixta de...'."""
    if not chunks:
        return False
    titulo = (chunks[0].get("titulo_video") or "").lower()
    return "comisi" in pleno._quitar_tildes(titulo)


def _reconciliar_nombres_genericos(ruta_json) -> int:
    """Tras identificar, si algún SPEAKER_ID quedó con la etiqueta genérica de
    rol (p. ej. "Ministro") en algunos chunks pero su nombre propio se resolvió
    en OTROS chunks suyos (porque la Mesa dijo "don/doña X" en otro momento),
    sustituye la etiqueta genérica por el nombre real en todos sus chunks.
    Si hay más de un nombre propio distinto para el mismo SPEAKER_ID, no toca
    nada (es una ambigüedad real, no un caso de genérico-vs-real)."""
    ruta_json = Path(ruta_json)
    chunks = pleno.cargar_chunks(ruta_json)
    por_ponente = {}
    for c in chunks:
        por_ponente.setdefault(c.get("ponente"), []).append(c)

    cambios = 0
    for ponente, lista in por_ponente.items():
        nombres_especificos = {
            c["nombre"] for c in lista
            if c.get("nombre") and c["nombre"] not in GENERIC_ROLE_NAMES
        }
        hay_generico = any(c.get("nombre") in GENERIC_ROLE_NAMES for c in lista)
        if hay_generico and len(nombres_especificos) == 1:
            nombre_real = next(iter(nombres_especificos))
            for c in lista:
                if c.get("nombre") in GENERIC_ROLE_NAMES:
                    c["nombre"] = nombre_real
                    c["metodo_id"] = f"{c.get('metodo_id')}+reconciliado_comision"
                    cambios += 1

    if cambios:
        pleno.guardar_json(chunks, ruta_json)
        print(f"[COMISIONES] Reconciliados {cambios} chunk(s): etiqueta de rol → nombre real")
    return cambios


def identificar_video_comision(
    ruta_json_entrada,
    ruta_json_salida=None,
    usar_llm: bool = True,
    umbral_llm: float = 0.90,
):
    """Fuerza la heurística de comisiones, sea cual sea el título."""
    ruta_salida = pleno.identificar_video(
        ruta_json_entrada, ruta_json_salida, usar_llm, umbral_llm,
        funcion_heuristica=detectar_patrones_heuristicos_comision,
    )
    _reconciliar_nombres_genericos(ruta_salida)
    return ruta_salida


def identificar_video_auto(
    ruta_json_entrada,
    ruta_json_salida=None,
    usar_llm: bool = True,
    umbral_llm: float = 0.90,
):
    """Punto de entrada recomendado para el pipeline: decide solo, por el
    título del vídeo, si aplicar la heurística de pleno o la de comisión."""
    chunks = pleno.cargar_chunks(Path(ruta_json_entrada))
    if es_comision(chunks):
        print("[AUTO] Detectada sesión de Comisión — heurística de comisiones")
        return identificar_video_comision(ruta_json_entrada, ruta_json_salida, usar_llm, umbral_llm)
    print("[AUTO] Detectada sesión de Pleno — heurística estándar")
    return pleno.identificar_video(ruta_json_entrada, ruta_json_salida, usar_llm, umbral_llm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python identificador_comisiones.py <ruta_json> [--sin-llm]")
        sys.exit(1)
    ruta = sys.argv[1]
    identificar_video_auto(ruta, usar_llm="--sin-llm" not in sys.argv)