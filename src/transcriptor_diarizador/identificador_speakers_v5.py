"""
identificador_speakers_v2_14.py
Bloque de identificación de speakers para el pipeline "Buscador Plenario Inteligente".

Versión 2.14 — Partido del diccionario expuesto a LLM-DICT:

  CAMBIO PRINCIPAL — NOMBRE_A_PARTIDO extrae, leyendo el propio código
  fuente (los comentarios "# PNV", "# Senador", etc. junto a cada entrada
  de ALIASES_NOMBRE_COMPLETO), un mapa nombre canónico -> partido para las
  1703 entradas del diccionario (~1130 con partido real, ~570 senadores/
  gobierno con solo el rol). La lista de candidatos que recibe LLM-DICT
  ahora muestra "Nombre (Partido)" en vez de solo el nombre. Además, si la
  Mesa no menciona el partido explícitamente, se usa como respaldo el
  partido verificado del diccionario para el nombre ya resuelto (ignorando
  "Senador"/"Gobierno", que no son partidos reales) antes de aceptar lo que
  diga el LLM. Esto añade una segunda red de seguridad, independiente de la
  extracción por palabra clave del texto de Mesa (v2.13): para los 3 casos
  reales de esa versión donde faltaba el partido, el diccionario YA tenía
  el dato correcto (Otero Gabirondo→EH Bildu, Rentería Lasanta→PNV, Rego
  Candamil→Mixto).

Versión 2.13 — Grounding de partido + muestreo de chunks con cierre incluido:

  CAMBIO PRINCIPAL — Detectados con datos reales de producción: (1) el LLM
  ignoraba a veces el partido que la propia Mesa decía explícitamente y
  "anclaba" en el partido más frecuente del contexto (ej. asignó PSOE a un
  diputado de PNV, uno de Bildu y uno del Grupo Mixto, los tres con el
  partido correcto dicho literalmente por la Mesa). Ahora
  _partido_desde_texto_mesa() extrae el partido por palabra clave del texto
  de Mesa en código (no depende del LLM) y SOBREESCRIBE lo que diga el LLM
  cuando hay mención explícita, en LLM-DESC y LLM-DICT. (2) indices[:4]
  siempre cogía los primeros chunks del bloque, pero la autoidentificación
  de partido a menudo está en el CIERRE del discurso (último chunk), que en
  bloques de más de 4 chunks nunca se incluía. Ahora se cogen los 3
  primeros + el último.

Versión 2.12 — Fix de truncado en texto_speaker (LLM-DESC y LLM-DICT):

  CAMBIO PRINCIPAL — texto_speaker se construía uniendo hasta 4 chunks y
  cortando la CONCATENACIÓN a 600/500 caracteres. Si el primer chunk del
  bloque ya superaba ese límite (caso real detectado: orador de Vox cuyo
  primer chunk tenía 1038 caracteres), los chunks siguientes —incluida una
  frase de autoidentificación de partido como "somos nosotros, Vox..."—
  nunca llegaban al prompt. Ahora cada chunk se trunca por separado antes
  de unirlos (400 car. en LLM-DESC, 300 en LLM-DICT), así el corte nunca se
  come chunks completos posteriores del mismo bloque.

Versión 2.11 — Few-shots + candidatos de diccionario para LLM-DICT:

  CAMBIO PRINCIPAL — Ambos prompts (LLM-DESC y LLM-DICT) incluyen ahora 4
  ejemplos few-shot con casos reales (incluido el fallo real de "Bellugera
  Balañá" -> "Pilar Vallugera Balañà" que se detectó en producción), y uno
  de anti-alucinación explícito para enseñar a devolver null cuando no hay
  información suficiente. Además, LLM-DICT ya NO se apoya solo en el
  conocimiento del modelo: antes de llamarlo, _candidatos_diccionario() hace
  una búsqueda difusa local (difflib) contra los ~661 nombres canónicos del
  diccionario y le pasa al LLM solo los 5-6 más parecidos textualmente al
  nombre deformado, en vez del diccionario entero (inviable: ~11 500 tokens,
  por encima del límite de 6 000 TPM del free tier de Groq).

Versión 2.10 — Migrado de Gemini a Groq:

  CAMBIO PRINCIPAL — LLM-DESC y LLM-DICT ya no llaman a Gemini, llaman a la
  API de Groq (requiere `pip install groq` y la variable de entorno
  GROQ_API_KEY). Modelo usado: llama-3.1-8b-instant, que en el free tier de
  Groq permite el mayor número de llamadas diarias del catálogo Llama
  (14 400 RPD / 30 RPM / 6 000 TPM), muy por encima de llama-3.3-70b-versatile
  (1 000 RPD). La lógica de prompts, umbrales y actualización de fingerprints
  no cambia — solo el proveedor del LLM (_llamar_groq reemplaza a
  _llamar_gemini_nuevo).

Versión 2.9 — Salida en consola para LLM-DESC/LLM-DICT:

  CAMBIO PRINCIPAL — logging.basicConfig solo escribe al fichero de log, no
  a la consola, así que las llamadas al LLM y sus resultados eran invisibles
  en tiempo real. Ahora identificar_desconocidos_con_llm() y
  verificar_nombres_fuera_diccionario_con_llm() imprimen en consola cada
  llamada que hacen y si el resultado provoca un cambio (✓) o no (✗/⚠).

Versión 2.8 — LLM-1 y LLM-2 eliminados por completo:

  CAMBIO PRINCIPAL — Se retiran identificar_con_gemini() (LLM-1) y
  verificar_heuristica_con_llm() (LLM-2). Ambas funciones llamaban a un
  objeto `modelo` que nunca se definía (resto de la SDK antigua
  google-generativeai, nunca migrado a la SDK nueva google-genai), por lo
  que LLM-1 fallaba en cada llamada con NameError ("name 'modelo' is not
  defined") y gastaba cuota de Gemini sin producir resultados. LLM-2 ya
  estaba desactivado con `if False:`.
  Ahora la única resolución por LLM ocurre después del bucle principal,
  vía LLM-DESC (identificar_desconocidos_con_llm) y LLM-DICT
  (verificar_nombres_fuera_diccionario_con_llm), que sí usan la SDK nueva
  correctamente. Los chunks que antes intentaban LLM-1 en el bucle
  principal ahora quedan DESCONOCIDO/AMBIGUO y los recoge LLM-DESC.

Versión 2.7 — LLM-2 reducido sobre v2.6:

  CAMBIO PRINCIPAL — Diccionario exhaustivo XV Legislatura.
    Diccionario generado automáticamente desde datos oficiales:
      350 diputados activos (Congreso XV Legislatura)
      269 senadores activos (Senado XV Legislatura)
      23 miembros del gobierno
      + variantes Whisper y aliases críticos de versiones anteriores
    Total: ~1700 entradas en ALIASES_NOMBRE_COMPLETO
    Estrategia de aliases:
      - ALIASES_NOMBRE_COMPLETO: nombre completo + apellidos compuestos
        (siempre seguros, se buscan antes que apellido único)
      - ALIASES_APELLIDO_UNICO: solo apellidos genuinamente únicos en
        el Congreso actual (lista reducida y curada)
    Con el diccionario ampliado, la mayoría de casos se resuelven en
    el paso 1 de normalizar_nombre (nombre completo exacto) sin llegar
    al alias de apellido único ni al LLM-2.

  Nota sobre fusión cross-ID (Prioridad 2):
    La fusión actual YA es general y correcta — agrupa todos los SPEAKER_IDs
    con el mismo nombre canónico. El problema de Rocío de Meer (SPEAKER_03/04)
    era falta de semilla, no fallo de fusión. El diccionario ampliado lo resuelve
    dando más rutas de entrada al nombre correcto.

  Conserva todos los cambios de v2.4/v2.3/v2.2/v2.1.

Uso:
    python identificador_speakers_v2_14.py ruta/al/chunks.json
    python identificador_speakers_v2_14.py ruta/al/chunks.json --sin-llm
"""

import re
import json
import time
import logging
import unicodedata
import sys
import difflib
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =============================================================
# DICCIONARIOS
# =============================================================

CARGOS_FIJOS = {
    "presidenta del congreso": "Francina Armengol",
    "presidente del senado": "Pedro Rollán",
    "presidente del gobierno": "Pedro Sánchez",
    "candidato a la presidencia": "Pedro Sánchez",
}

GRUPOS_PARLAMENTARIOS = {
    "grupo parlamentario popular": "PP",
    "grupo parlamentario socialista": "PSOE",
    "grupo parlamentario vox": "Vox",
    "grupo parlamentario plurinacional sumar": "Sumar",
    "grupo parlamentario republicano": "ERC",
    "grupo parlamentario junts per catalunya": "Junts",
    "grupo parlamentario euskal herria bildu": "EH Bildu",
    "grupo parlamentario vasco": "PNV",
    "grupo parlamentario mixto": None,
}

# Aliases multi-token seguros: siempre se aplican (orden 1 y 2 en normalizar_nombre)
ALIASES_NOMBRE_COMPLETO = {

    # ── Nombres clave con variantes Whisper (prioridad) ────────────────────
    "baquiero montero": "Maribel Vaquero Montero",  # PNV
    "baquiero": "Maribel Vaquero Montero",  # PNV
    "mejias sanchez": "Carina Mejías Sánchez",  # Vox
    "minguez macho": "Verónica Míguez Macho",  # PSOE
    "jordi raul": "Laia Jordà",  # ERC
    "jorda rau": "Laia Jordà",  # ERC
    "laia jorda": "Laia Jordà",  # ERC
    "escorbe torra": "Juan José Aizcorbe Torra",  # Vox
    "aizcorbe torra": "Juan José Aizcorbe Torra",  # Vox
    "de meer": "Rocío de Meer",  # Vox
    "de mer": "Rocío de Meer",  # Vox
    "demer": "Rocío de Meer",  # Vox
    "de mers": "Rocío de Meer",  # Vox
    "de meer mendez": "Rocío de Meer",  # Vox
    "rocio de mer": "Rocío de Meer",  # Vox
    "mariboso": "Jaime de Olano Vela",  # PP
    "moriboso": "Jaime de Olano Vela",  # PP
    "inagritu": "Jon Iñarritu García",  # EH Bildu
    "fijoo": "Alberto Núñez Feijóo",  # PP
    "fijo": "Alberto Núñez Feijóo",  # PP
    "feijo": "Alberto Núñez Feijóo",  # PP
    "patetoledo": "Cayetana Álvarez de Toledo",  # PP
    "madre simil": "Noèlia Madrera Simil",  # Junts
    "madre de la simil": "Noèlia Madrera Simil",  # Junts
    "rego campbell": "Néstor Rego Candamil",  # Mixto
    "catalan i deras": "Alberto Catalán Higueras",  # Mixto
    "cuca gamarra": "Cuca Gamarra",  # PP
    "patxi lopez": "Patxi López",  # PSOE
    "irene montero": "Irene Montero",  # Sumar
    "pedro sanchez": "Pedro Sánchez",  # PSOE
    "yolanda diaz": "Yolanda Díaz",  # Sumar
    "gabriel rufian": "Gabriel Rufián",  # ERC
    "mertxe aizpurua": "Mertxe Aizpurua Arzallus",  # EH Bildu
    "ione belarra": "Ione Belarra Urteaga",  # Sumar
    "miriam nogueras": "Míriam Nogueras i Camero",  # Junts
    "oscar puente": "Óscar Puente Santiago",  # Gobierno
    "felix bolanos": "Félix Bolaños García",  # Gobierno
    "arnaldo otegi": "Arnaldo Otegi",  # EH Bildu
    "carles puigdemont": "Carles Puigdemont",  # Junts
    "aitor esteban": "Aitor Esteban Bravo",  # PNV
    "ramos esteban": "César Ramos Esteban",  # PSOE
    "armengol": "Francina Armengol Socias",  # PSOE
    "rollan": "Pedro Rollán",  # PP
    "abascal": "Santiago Abascal Conde",  # Vox
    "lastra": "Adriana Lastra",  # PSOE
    "batet": "Meritxell Batet",  # PSOE
    "gamarra": "Cuca Gamarra",  # PP
    "garriga": "Ignacio Garriga",  # Vox
    "monasterio": "Rocío Monasterio",  # Vox
    "galindo": "Enrique Santiago Romero",  # Sumar
    "maraver": "Agustín Santos Maraver",  # Sumar
    "belarra": "Ione Belarra Urteaga",  # Sumar
    "urteaga": "Ione Belarra Urteaga",  # Sumar
    "feijoo": "Alberto Núñez Feijóo",  # PP
    "mico mico": "Àgueda Micó Micó",  # Sumar
    "agueda mico": "Àgueda Micó Micó",  # Sumar
    "madrera simil": "Noèlia Madrera Simil",  # Junts
    "montesinos de miguel": "Macarena Montesinos de Miguel",  # PP
    "montesinos": "Macarena Montesinos de Miguel",  # PP
    "alvarez de toledo": "Cayetana Álvarez de Toledo",  # PP
    "espinosa de los monteros": "Iván Espinosa de los Monteros",  # Vox
    "rodriguez de millan": "María José Rodríguez de Millán Parro",  # Vox
    "millan parro": "María José Rodríguez de Millán Parro",  # Vox
    "valido garcia": "Cristina Valido García",  # Mixto
    "ortega smith": "Francisco Javier Ortega Smith-Molina",  # Vox
    "romero pozo": "Rafaela Romero Pozo",  # PSOE
    "minguez garcia": "Montse Mínguez García",  # PSOE
    "veronica martinez": "Verónica Martínez Barbero",  # Sumar
    "martinez barbero": "Verónica Martínez Barbero",  # Sumar
    "santiago romero": "Enrique Santiago Romero",  # Sumar
    "gil lazaro": "Ignacio Gil Lázaro",  # Vox
    "figaredo": "José María Figaredo Álvarez-Sala",  # Vox
    "vaquero montero": "Maribel Vaquero Montero",  # PNV
    "idoia vaquero": "Maribel Vaquero Montero",  # PNV
    "idoia vaquero montero": "Maribel Vaquero Montero",  # PNV

    # ── Diputados, Senadores y Gobierno (generado automáticamente) ─────────
    "aagesen munoz": "Sara Aagesen Muñoz",  # Gobierno
    "abades martinez": "Cristina Abades Martínez",  # PP
    "abascal conde": "Santiago Abascal Conde",  # Vox
    "abdelhakim abdeselam al lal": "Abdelhakim Abdeselam Al Lal",  # Senador
    "abigail garrido tinta": "Abigail Garrido Tinta",  # Senador
    "acedo reyes": "Sofía Acedo Reyes",  # PP
    "aceves galindo": "José Luis Aceves Galindo",  # PSOE
    "ada santana": "Ada Santana Aguilera",  # PSOE
    "ada santana aguilera": "Ada Santana Aguilera",  # PSOE
    "adolfo lander vera": "Adolfo Lander Vera",  # Senador
    "adrian gutierrez": "Miguel Ángel Adrián Gutiérrez",  # Senador
    "adriana maldonado": "Adriana Maldonado López",  # PSOE
    "adriana maldonado lopez": "Adriana Maldonado López",  # PSOE
    "adrio taracido": "María Adrio Taracido",  # PSOE
    "agirretxea urresti": "Joseba Andoni Agirretxea Urresti",  # PNV
    "agueda mico mico": "Àgueda Micó Micó",  # Sumar
    "aguera gago": "Cristina Agüera Gago",  # PP
    "aguirre gil de biedma": "Rocío Aguirre Gil de Biedma",  # Vox
    "agustin almodobar barcelo": "Agustín Almodóbar Barceló",  # Senador
    "agustin conde": "Agustín Conde Bajén",  # PP
    "agustin conde bajen": "Agustín Conde Bajén",  # PP
    "agustin parra": "Agustín Parra Gallego",  # PP
    "agustin parra gallego": "Agustín Parra Gallego",  # PP
    "agustin santos": "Agustín Santos Maraver",  # Sumar
    "agustin santos maraver": "Agustín Santos Maraver",  # Sumar
    "ahedo ceza": "Nerea Ahedo Ceza",  # Senador
    "aina vidal": "Aina Vidal Sáez",  # Sumar
    "aina vidal saez": "Aina Vidal Sáez",  # Sumar
    "ainhoa molina": "Ainhoa Molina León",  # PP
    "ainhoa molina leon": "Ainhoa Molina León",  # PP
    "aizpurua arzallus": "Mertxe Aizpurua Arzallus",  # EH Bildu
    "al lal": "Abdelhakim Abdeselam Al Lal",  # Senador
    "alba soldevilla": "Alba Soldevilla Novials",  # PSOE
    "alba soldevilla novials": "Alba Soldevilla Novials",  # PSOE
    "albaladejo gutierrez": "Marcos Albaladejo Gutiérrez",  # Senador
    "albares bueno": "José Manuel Albares Bueno",  # Gobierno
    "alberto asarta": "Alberto Asarta Cuevas",  # Vox
    "alberto asarta cuevas": "Alberto Asarta Cuevas",  # Vox
    "alberto catalan": "Alberto Catalán Higueras",  # Mixto
    "alberto catalan higueras": "Alberto Catalán Higueras",  # Mixto
    "alberto fabra": "Alberto Fabra Part",  # PP
    "alberto fabra part": "Alberto Fabra Part",  # PP
    "alberto ibanez": "Alberto Ibáñez Mezquita",  # Sumar
    "alberto ibanez mezquita": "Alberto Ibáñez Mezquita",  # Sumar
    "alberto mayoral": "Alberto Mayoral de Lamo",  # PSOE
    "alberto mayoral de": "Alberto Mayoral de Lamo",  # PSOE
    "alberto mayoral de lamo": "Alberto Mayoral de Lamo",  # PSOE
    "alberto nunez": "Alberto Núñez Feijóo",  # PP
    "alberto nunez feijoo": "Alberto Núñez Feijóo",  # PP
    "alberto rojo": "Alberto Rojo Blas",  # PSOE
    "alberto rojo blas": "Alberto Rojo Blas",  # PSOE
    "alcaraz martos": "Francisco José Alcaraz Martos",  # Vox
    "alda recas": "Alda Recas Martín",  # Sumar
    "alda recas martin": "Alda Recas Martín",  # Sumar
    "aldea gomez": "Rosa María Aldea Gómez",  # Senador
    "alejandro soler": "Alejandro Soler Mur",  # PSOE
    "alejandro soler mur": "Alejandro Soler Mur",  # PSOE
    "alejo joaquin miranda de larra arnaiz": "Alejo Joaquín Miranda de Larra Arnaiz",  # Senador
    "alfonso carlos macias": "Alfonso Carlos Macías Gata",  # PP
    "alfonso carlos macias gata": "Alfonso Carlos Macías Gata",  # PP
    "alfonso carlos moscoso gonzalez": "Alfonso Carlos Moscoso González",  # Senador
    "alfonso carlos serrano sanchez-capuchino": "Alfonso Carlos Serrano Sánchez-Capuchino",  # Senador
    "alfonso cendon": "Javier Alfonso Cendón",  # PSOE
    "alfonso garcia rodriguez": "Alfonso García Rodríguez",  # Senador
    "alfonso gil invernon": "Alfonso Gil Invernón",  # Senador
    "alfonso rodriguez": "Alfonso Rodríguez Gómez de Celis",  # PSOE
    "alfonso rodriguez gomez": "Alfonso Rodríguez Gómez de Celis",  # PSOE
    "alfonso rodriguez gomez de celis": "Alfonso Rodríguez Gómez de Celis",  # PSOE
    "alfonso silvestre": "Alma Alfonso Silvestre",  # PP
    "alia aguado": "María Pilar Alía Aguado",  # PP
    "alicia alvarez": "Alicia Álvarez González",  # PSOE
    "alicia alvarez gonzalez": "Alicia Álvarez González",  # PSOE
    "alicia garcia rodriguez": "Alicia García Rodríguez",  # Senador
    "alma alfonso": "Alma Alfonso Silvestre",  # PP
    "alma alfonso silvestre": "Alma Alfonso Silvestre",  # PP
    "almiron ruiz": "Oriol Almirón Ruiz",  # PSOE
    "almodobar barcelo": "Agustín Almodóbar Barceló",  # Senador
    "almodovar sanchez": "Emilia Almodóvar Sánchez",  # PSOE
    "alonso cantorne": "Fèlix Alonso Cantorné",  # Sumar
    "alonso coronel": "Javier Valentín Alonso Coronel",  # Senador
    "alonso perez": "José Ángel Alonso Pérez",  # Senador
    "alos lopez": "Ana Isabel Alós López",  # PP
    "alvarez de toledo peralta-ramos": "Cayetana Álvarez de Toledo Peralta-Ramos",  # PP
    "alvarez fanjul": "Beatriz Álvarez Fanjul",  # PP
    "alvarez gonzalez": "Alicia Álvarez González",  # PSOE
    "alvaro morales": "Álvaro Morales Álvarez",  # PSOE
    "alvaro morales alvarez": "Álvaro Morales Álvarez",  # PSOE
    "alvaro perez": "Álvaro Pérez López",  # PP
    "alvaro perez lopez": "Álvaro Pérez López",  # PP
    "alvaro vidal": "Francesc-Marc Álvaro Vidal",  # ERC
    "amador marques": "Amador Marqués Atés",  # PSOE
    "amador marques ates": "Amador Marqués Atés",  # PSOE
    "amaro huelva betanzos": "Amaro Huelva Betanzos",  # Senador
    "amores garcia": "Juan Ramón Amores García",  # Senador
    "amparo torres valencoso": "Amparo Torres Valencoso",  # Senador
    "ana belen vazquez": "Ana Belén Vázquez Blanco",  # PP
    "ana belen vazquez blanco": "Ana Belén Vázquez Blanco",  # PP
    "ana cobo": "Ana Cobo Carmona",  # PSOE
    "ana cobo carmona": "Ana Cobo Carmona",  # PSOE
    "ana gonzalez": "Ana González Herdaro",  # PSOE
    "ana gonzalez herdaro": "Ana González Herdaro",  # PSOE
    "ana isabel alos": "Ana Isabel Alós López",  # PP
    "ana isabel alos lopez": "Ana Isabel Alós López",  # PP
    "ana maria beltran villalba": "Ana María Beltrán Villalba",  # Senador
    "ana martinez": "Ana Martínez Labella",  # PP
    "ana martinez labella": "Ana Martínez Labella",  # PP
    "ana martinez zaragoza": "Ana Martínez Zaragoza",  # Senador
    "ana redondo garcia": "Ana Redondo García",  # Gobierno
    "andala ubbi": "Teslem Andala Ubbi",  # Sumar
    "andrea canelo": "Andrea Canelo Matito",  # PSOE
    "andrea canelo matito": "Andrea Canelo Matito",  # PSOE
    "andrea fernandez": "Andrea Fernández Benéitez",  # PSOE
    "andrea fernandez beneitez": "Andrea Fernández Benéitez",  # PSOE
    "andres alberto rodriguez": "Andrés Alberto Rodríguez Almeida",  # Vox
    "andres alberto rodriguez almeida": "Andrés Alberto Rodríguez Almeida",  # Vox
    "andres anon": "Carmen Andrés Añón",  # PSOE
    "andreu martin": "Andreu Martín Martínez",  # PSOE
    "andreu martin martinez": "Andreu Martín Martínez",  # PSOE
    "andreu rodriguez": "Concepción Andreu Rodríguez",  # Senador
    "angel ibanez": "Ángel Ibáñez Hernando",  # PP
    "angel ibanez hernando": "Ángel Ibáñez Hernando",  # PP
    "angel lopez": "Ángel López Maraver",  # Vox
    "angel lopez maraver": "Ángel López Maraver",  # Vox
    "angel luis gonzalez munoz": "Ángel Luis González Muñoz",  # Senador
    "angel pelayo gordillo moreno": "Ángel Pelayo Gordillo Moreno",  # Senador
    "angel victor torres perez": "Ángel Víctor Torres Pérez",  # Gobierno
    "aniceto javier armas gonzalez": "Aniceto Javier Armas González",  # Senador
    "antidio fagundez": "Antidio Fagúndez Campo",  # PSOE
    "antidio fagundez campo": "Antidio Fagúndez Campo",  # PSOE
    "anton cacho": "Javier Antón Cacho",  # Senador
    "antonia lopez moya": "Antonia López Moya",  # Senador
    "antonio cavacasillas": "Antonio Cavacasillas Rodríguez",  # PP
    "antonio cavacasillas rodriguez": "Antonio Cavacasillas Rodríguez",  # PP
    "antonio gutierrez limones": "Antonio Gutiérrez Limones",  # Senador
    "antonio luengo zapata": "Antonio Luengo Zapata",  # Senador
    "antonio magdaleno alegria": "Antonio Magdaleno Alegría",  # Senador
    "antonio martinez": "Antonio Martínez Gómez",  # PP
    "antonio martinez gomez": "Antonio Martínez Gómez",  # PP
    "antonio martinez rodriguez": "Antonio Martínez Rodríguez",  # Senador
    "antonio munoz martinez": "Antonio Muñoz Martínez",  # Senador
    "antonio poveda zapata": "Antonio Poveda Zapata",  # Senador
    "antonio roman": "Antonio Román Jasanada",  # PP
    "antonio roman jasanada": "Antonio Román Jasanada",  # PP
    "antonio silvan rodriguez": "Antonio Silván Rodríguez",  # Senador
    "antunano colina": "Pablo Antuñano Colina",  # PSOE
    "araceli martinez esteban": "Araceli Martínez Esteban",  # Senador
    "aragones mendiguchia": "Carlos Aragonés Mendiguchía",  # PP
    "aranda garcia": "Mariola Aranda García",  # Senador
    "aranda lassa": "José Manuel Aranda Lassa",  # Senador
    "aranda vargas": "Francisco Aranda Vargas",  # PSOE
    "arcadi espana garcia": "Arcadi España García",  # Gobierno
    "arenas bocanegra": "Francisco Javier Arenas Bocanegra",  # Senador
    "arevalo gomez": "Nidia María Arévalo Gómez",  # Senador
    "argota castro": "Trinidad Carmen Argota Castro",  # PSOE
    "arguelles garcia": "Silverio Argüelles García",  # PP
    "armario gonzalez": "Blanca Armario González",  # Vox
    "armas gonzalez": "Aniceto Javier Armas González",  # Senador
    "armengol socias": "Francina Armengol Socias",  # PSOE
    "armijo navas": "José Alberto Armijo Navas",  # Senador
    "arnau ramirez": "Arnau Ramírez Carner",  # PSOE
    "arnau ramirez carner": "Arnau Ramírez Carner",  # PSOE
    "arocha correa": "Marta Arocha Correa",  # Senador
    "arriba sanchez": "Bienvenido de Arriba Sánchez",  # Senador
    "arribas maroto": "Manuel Arribas Maroto",  # PSOE
    "artemi rallo": "Artemi Rallo Lombarte",  # PSOE
    "artemi rallo lombarte": "Artemi Rallo Lombarte",  # PSOE
    "asarta cuevas": "Alberto Asarta Cuevas",  # Vox
    "aurora nacarino-brabo": "Aurora Nacarino-Brabo Jiménez",  # PP
    "aurora nacarino-brabo jimenez": "Aurora Nacarino-Brabo Jiménez",  # PP
    "avila gutierrez": "Juan Manuel Ávila Gutiérrez",  # Senador
    "aznar teruel": "Evarist Aznar Teruel",  # PP
    "azorin salar": "Lázaro Azorín Salar",  # PSOE
    "azpitarte perez": "Vicente Azpitarte Pérez",  # Senador
    "badia casas": "Eloi Badia Casas",  # Sumar
    "bague roura": "Joan Baptista Bagué Roura",  # Senador
    "bailac ardanuy": "Sara Bailac Ardanuy",  # Senador
    "ballester feliu": "Carmen Ballester Feliu",  # Senador
    "balseiro orol": "José Manuel Balseiro Orol",  # Senador
    "baltar blanco": "José Manuel Baltar Blanco",  # Senador
    "barasoain rodrigo": "Sergio Barasoain Rodrigo",  # Senador
    "barcos berruezo": "Miren Uxue Barcos Berruezo",  # Senador
    "barreiro fernandez": "José Manuel Barreiro Fernández",  # Senador
    "barrio baroja": "Carmelo Barrio Baroja",  # PP
    "barrios tejero": "José María Barrios Tejero",  # Senador
    "bartolome madrid": "Bartolomé Madrid Olmo",  # PP
    "bartolome madrid olmo": "Bartolomé Madrid Olmo",  # PP
    "bayon rolo": "Juan Andrés Bayón Rolo",  # PP
    "beamonte mesa": "Luis María Beamonte Mesa",  # PP
    "beatriz alvarez": "Beatriz Álvarez Fanjul",  # PP
    "beatriz alvarez fanjul": "Beatriz Álvarez Fanjul",  # PP
    "beatriz jimenez": "Beatriz Jiménez Linuesa",  # PP
    "beatriz jimenez linuesa": "Beatriz Jiménez Linuesa",  # PP
    "begona nasarre": "Begoña Nasarre Oliva",  # PSOE
    "begona nasarre oliva": "Begoña Nasarre Oliva",  # PSOE
    "belarra urteaga": "Ione Belarra Urteaga",  # Sumar
    "belda perez-pedrero": "Enrique Belda Pérez-Pedrero",  # PP
    "belen hoyo": "Belén Hoyo Juliá",  # PP
    "belen hoyo julia": "Belén Hoyo Juliá",  # PP
    "bella verano": "Bella Verano Domínguez",  # PP
    "bella verano dominguez": "Bella Verano Domínguez",  # PP
    "belmonte gomez": "Rafael Benigno Belmonte Gómez",  # PP
    "belmonte sanchez": "Teresa María Belmonte Sánchez",  # Senador
    "beltran villalba": "Ana María Beltrán Villalba",  # Senador
    "bendodo benasayag": "Elías Bendodo Benasayag",  # PP
    "benjamin prieto valencia": "Benjamín Prieto Valencia",  # Senador
    "bermudez carrillo": "Francisco Javier Bermúdez Carrillo",  # Senador
    "bermudez de castro fernandez": "José Antonio Bermúdez de Castro Fernández",  # PP
    "bernabe perez": "Francisco Martín Bernabé Pérez",  # Senador
    "bideguren gabantxo": "Idurre Bideguren Gabantxo",  # Senador
    "bienvenido de arriba sanchez": "Bienvenido de Arriba Sánchez",  # Senador
    "blanca armario": "Blanca Armario González",  # Vox
    "blanca armario gonzalez": "Blanca Armario González",  # Vox
    "blanca cercas": "Blanca Cercas Mena",  # PSOE
    "blanca cercas mena": "Blanca Cercas Mena",  # PSOE
    "blanch fulcara": "Luisa Blanch Fulcarà",  # Senador
    "blanco arrue": "Gabriel Blanco Arrúe",  # PSOE
    "blanco garrido": "María Mar Blanco Garrido",  # Senador
    "blanquer alcaraz": "Patricia Blanquer Alcaraz",  # PSOE
    "boada danes": "Júlia Boada Danés",  # Sumar
    "bolanos garcia": "Félix Bolaños García",  # PSOE
    "bone amela": "Carlos Luis Boné Amela",  # Senador
    "bonilla dominguez": "María Jesús Bonilla Domínguez",  # Senador
    "borja semper": "Borja Sémper Pascual",  # PP
    "borja semper pascual": "Borja Sémper Pascual",  # PP
    "borrego cortes": "Isabel María Borrego Cortés",  # PP
    "borrego rodriguez": "Manuel Borrego Rodríguez",  # Senador
    "bravo baena": "Juan Bravo Baena",  # PP
    "bravo sanchez": "Miriam Bravo Sánchez",  # Senador
    "brigida pachon": "Brígida Pachón Martín",  # PSOE
    "brigida pachon martin": "Brígida Pachón Martín",  # PSOE
    "brio gonzalez": "Esther Basilia del Brío González",  # Senador
    "briones morales": "Rocío Briones Morales",  # Senador
    "bueno campanario": "Eva Patricia Bueno Campanario",  # Senador
    "bueno vargas": "Simón Valentín Bueno Vargas",  # Senador
    "buj sanchez": "María Emma Buj Sánchez",  # Senador
    "bustinduy amador": "Pablo Bustinduy Amador",  # Gobierno
    "caballero martinez": "María Mar Caballero Martínez",  # Senador
    "cabezon casas": "Tomás Cabezón Casas",  # PP
    "cacho isla": "Iván Cacho Isla",  # PSOE
    "caicedo bernabe": "Jesús Caicedo Bernabé",  # Senador
    "calvo gomez": "Pilar Calvo Gómez",  # Junts
    "camacho borrego": "Francisco Joaquín Camacho Borrego",  # Senador
    "camino minana": "Víctor Camino Miñana",  # PSOE
    "campos asensi": "Jorge Campos Asensi",  # Vox
    "campoy monreal": "Javier Campoy Monreal",  # Senador
    "camps devesa": "Gerardo Camps Devesa",  # Senador
    "candela lopez": "Candela López Tagliafico",  # Sumar
    "candela lopez tagliafico": "Candela López Tagliafico",  # Sumar
    "canelo matito": "Andrea Canelo Matito",  # PSOE
    "cantalapiedra alvarez": "María de las Mercedes Cantalapiedra Álvarez",  # PP
    "cantenys arboli": "Consol Cantenys Arbolí",  # Senador
    "carazo hermoso": "Eduardo Carazo Hermoso",  # PP
    "carballedo berlanga": "María Eugenia Carballedo Berlanga",  # PP
    "carbonell tatay": "Fernando Carbonell Tatay",  # Senador
    "caridad rives": "Caridad Rives Arcayna",  # PSOE
    "caridad rives arcayna": "Caridad Rives Arcayna",  # PSOE
    "carina mejias": "Carina Mejías Sánchez",  # Vox
    "carina mejias sanchez": "Carina Mejías Sánchez",  # Vox
    "carla delgado gomez": "Carla Delgado Gómez",  # Senador
    "carlos alberto sanchez": "Carlos Alberto Sánchez Ojeda",  # PP
    "carlos alberto sanchez ojeda": "Carlos Alberto Sánchez Ojeda",  # PP
    "carlos alfonso polanco rebolleda": "Carlos Alfonso Polanco Rebolleda",  # Senador
    "carlos aragones": "Carlos Aragonés Mendiguchía",  # PP
    "carlos aragones mendiguchia": "Carlos Aragonés Mendiguchía",  # PP
    "carlos cuerpo caballero": "Carlos Cuerpo Caballero",  # Gobierno
    "carlos flores": "Carlos Flores Juberías",  # Vox
    "carlos flores juberias": "Carlos Flores Juberías",  # Vox
    "carlos garcia": "Carlos García Adanero",  # PP
    "carlos garcia adanero": "Carlos García Adanero",  # PP
    "carlos hernandez": "Carlos Hernández Quero",  # Vox
    "carlos hernandez quero": "Carlos Hernández Quero",  # Vox
    "carlos javier floriano": "Carlos Javier Floriano Corrales",  # PP
    "carlos javier floriano corrales": "Carlos Javier Floriano Corrales",  # PP
    "carlos luis bone amela": "Carlos Luis Boné Amela",  # Senador
    "carlos martin": "Carlos Martín Urriza",  # Sumar
    "carlos martin urriza": "Carlos Martín Urriza",  # Sumar
    "carlos rojas": "Carlos Rojas García",  # PP
    "carlos rojas garcia": "Carlos Rojas García",  # PP
    "carlos simarro": "Carlos Simarro Vicens",  # PP
    "carlos simarro vicens": "Carlos Simarro Vicens",  # PP
    "carlos yecora roca": "Carlos Yécora Roca",  # Senador
    "carmelo barrio": "Carmelo Barrio Baroja",  # PP
    "carmelo barrio baroja": "Carmelo Barrio Baroja",  # PP
    "carmelo romero hernandez": "Carmelo Romero Hernández",  # Senador
    "carmen andres": "Carmen Andrés Añón",  # PSOE
    "carmen andres anon": "Carmen Andrés Añón",  # PSOE
    "carmen ballester feliu": "Carmen Ballester Feliu",  # Senador
    "carmen belen lopez zapata": "Carmen Belén López Zapata",  # Senador
    "carmen funez": "Carmen Fúnez de Gregorio",  # PP
    "carmen funez de": "Carmen Fúnez de Gregorio",  # PP
    "carmen funez de gregorio": "Carmen Fúnez de Gregorio",  # PP
    "carmen martinez": "Carmen Martínez Ramírez",  # PSOE
    "carmen martinez ramirez": "Carmen Martínez Ramírez",  # PSOE
    "carmen navarro": "Carmen Navarro Lacoba",  # PP
    "carmen navarro lacoba": "Carmen Navarro Lacoba",  # PP
    "carmen pagador lopez": "Carmen Pagador López",  # Senador
    "carmen torralba valiente": "Carmen Torralba Valiente",  # Senador
    "carnero garcia": "Jesús Julio Carnero García",  # Senador
    "caro adanero": "Jesús Caro Adanero",  # Senador
    "casal miguez": "Verónica María Casal Míguez",  # Senador
    "casanueva jimenez": "Cristina Casanueva Jiménez",  # Senador
    "caso roiz": "Secundino Caso Roiz",  # Senador
    "castel fort": "Laura Castel Fort",  # Senador
    "castellon rubio": "Miguel Ángel Castellón Rubio",  # Senador
    "castilla alvarez": "María Carmen Castilla Álvarez",  # PSOE
    "castillo lopez": "Elena Castillo López",  # Senador
    "castillo rodriguez": "Lucas Castillo Rodríguez",  # Senador
    "catalan higueras": "Alberto Catalán Higueras",  # Mixto
    "cavacasillas rodriguez": "Antonio Cavacasillas Rodríguez",  # PP
    "cayetana alvarez": "Cayetana Álvarez de Toledo Peralta-Ramos",  # PP
    "cayetana alvarez de": "Cayetana Álvarez de Toledo Peralta-Ramos",  # PP
    "cayetana alvarez de toledo peralta-ramos": "Cayetana Álvarez de Toledo Peralta-Ramos",  # PP
    "celaya brey": "Javier Celaya Brey",  # PP
    "celso luis delgado": "Celso Luis Delgado Arce",  # PP
    "celso luis delgado arce": "Celso Luis Delgado Arce",  # PP
    "cercas mena": "Blanca Cercas Mena",  # PSOE
    "cervera pinart": "Josep Maria Cervera Pinart",  # Junts
    "cesar alejandro mogo zaro": "César Alejandro Mogo Zaro",  # Senador
    "cesar joaquin ramos": "César Joaquín Ramos Esteban",  # PSOE
    "cesar joaquin ramos esteban": "César Joaquín Ramos Esteban",  # PSOE
    "cesar sanchez": "César Sánchez Pérez",  # PP
    "cesar sanchez perez": "César Sánchez Pérez",  # PP
    "chamorro delmo": "Ricardo Chamorro Delmo",  # Vox
    "chinea correa": "Fabián Chinea Correa",  # Senador
    "clavell lopez": "Óscar Clavell López",  # PP
    "clemente munoz": "Raquel Clemente Muñoz",  # PP
    "cobo carmona": "Ana Cobo Carmona",  # PSOE
    "cobo perez": "Noelia Cobo Pérez",  # PSOE
    "cobo vega": "Manuel Cobo Vega",  # PP
    "cofino fernandez": "Rafael Cofiño Fernández",  # Sumar
    "colome garcia": "Gabriel Colomé García",  # Senador
    "concepcion andreu rodriguez": "Concepción Andreu Rodríguez",  # Senador
    "concepcion gamarra": "Concepción Gamarra Ruiz-Clavijo",  # PP
    "concepcion gamarra ruiz-clavijo": "Concepción Gamarra Ruiz-Clavijo",  # PP
    "conde bajen": "Agustín Conde Bajén",  # PP
    "conde lopez": "Francisco José Conde López",  # PP
    "conesa coma": "Ignasi Conesa Coma",  # PSOE
    "consol cantenys arboli": "Consol Cantenys Arbolí",  # Senador
    "cortes carballo": "Mario Cortés Carballo",  # PP
    "corujo berriel": "María Dolores Corujo Berriel",  # PSOE
    "cotelo balmaseda": "Mar Cotelo Balmaseda",  # Senador
    "crespin rubio": "Rafaela Crespín Rubio",  # PSOE
    "crespo iglesias": "José Crespo Iglesias",  # Senador
    "cristina abades": "Cristina Abades Martínez",  # PP
    "cristina abades martinez": "Cristina Abades Martínez",  # PP
    "cristina aguera": "Cristina Agüera Gago",  # PP
    "cristina aguera gago": "Cristina Agüera Gago",  # PP
    "cristina casanueva jimenez": "Cristina Casanueva Jiménez",  # Senador
    "cristina diaz moreno": "Cristina Díaz Moreno",  # Senador
    "cristina lopez": "Cristina López Zamora",  # PSOE
    "cristina lopez zamora": "Cristina López Zamora",  # PSOE
    "cristina moreno": "Cristina Moreno Borrás",  # PP
    "cristina moreno borras": "Cristina Moreno Borrás",  # PP
    "cristina moreno fernandez": "Cristina Moreno Fernández",  # Senador
    "cristina narbona": "Cristina Narbona Ruiz",  # PSOE
    "cristina narbona ruiz": "Cristina Narbona Ruiz",  # PSOE
    "cristina teniente": "Cristina Teniente Sánchez",  # PP
    "cristina teniente sanchez": "Cristina Teniente Sánchez",  # PP
    "cristina valido": "Cristina Valido García",  # Mixto
    "cristina valido garcia": "Cristina Valido García",  # Mixto
    "cristobal garre": "Cristóbal Garre Murcia",  # PP
    "cristobal garre murcia": "Cristóbal Garre Murcia",  # PP
    "cristobal marques palliser": "Cristóbal Marqués Palliser",  # Senador
    "cruset domenech": "Josep Maria Cruset Domènech",  # Junts
    "cruz rodriguez": "Manuel Cruz Rodríguez",  # Senador
    "cruz santana": "Gabriel Cruz Santana",  # PSOE
    "cruz-guzman garcia": "María Soledad Cruz-Guzmán García",  # PP
    "cuerpo caballero": "Carlos Cuerpo Caballero",  # Gobierno
    "cuesta alonso": "Severiano Ángel Cuesta Alonso",  # Senador
    "cuesta rodriguez": "María Cuesta Rodríguez",  # PP
    "cuevas larrosa": "Raúl Cuevas Larrosa",  # PP
    "dalmau blanco": "Miguel Carmelo Dalmau Blanco",  # Senador
    "daniel perez": "Daniel Pérez Osma",  # PP
    "daniel perez osma": "Daniel Pérez Osma",  # PP
    "daniel senderos": "Daniel Senderos Oraá",  # PSOE
    "daniel senderos oraa": "Daniel Senderos Oraá",  # PSOE
    "darocas marin": "Estela del Carmen Darocas Marín",  # Senador
    "david garcia": "David García Gomis",  # Vox
    "david garcia gomis": "David García Gomis",  # Vox
    "david matute perez": "David Matute Pérez",  # Senador
    "david serrada": "David Serrada Pariente",  # PSOE
    "david serrada pariente": "David Serrada Pariente",  # PSOE
    "de castro": "María Teresa Mallada de Castro",  # Senador
    "de la rosa baena": "Olvido De la Rosa Baena",  # PSOE
    "de las cuevas cortes": "Félix De las Cuevas Cortés",  # PP
    "de los santos gonzalez": "Jaime Miguel De los Santos González",  # PP
    "de luna tobarra": "Llanos De Luna Tobarra",  # PP
    "de otazu": "Fernando Adolfo Gutiérrez Díaz de Otazu",  # Senador
    "de rosa torner": "Fernando De Rosa Torner",  # PP
    "del valle rodriguez": "Emilio Jesús Del Valle Rodríguez",  # Vox
    "delgado arce": "Celso Luis Delgado Arce",  # PP
    "delgado gomez": "Carla Delgado Gómez",  # Senador
    "delgado-taramona hernandez": "Jimena Delgado-Taramona Hernández",  # PP
    "diana morant ripoll": "Diana Morant Ripoll",  # Gobierno
    "diaz marin": "Raúl Díaz Marín",  # PSOE
    "diaz moreno": "Cristina Díaz Moreno",  # Senador
    "diaz pacheco": "Susana Díaz Pacheco",  # Senador
    "diaz perez": "Yolanda Díaz Pérez",  # Sumar
    "dionis ona martin": "Dionís Oña Martín",  # Senador
    "diouf dioh": "Luc Andre Diouf Dioh",  # PSOE
    "divar conde": "María del Rocío Dívar Conde",  # Senador
    "duarte lopez": "Olaia Duarte López",  # Senador
    "eduard pujol bonell": "Eduard Pujol Bonell",  # Senador
    "eduardo carazo": "Eduardo Carazo Hermoso",  # PP
    "eduardo carazo hermoso": "Eduardo Carazo Hermoso",  # PP
    "edurne uriarte": "Edurne Uriarte Bengoechea",  # PP
    "edurne uriarte bengoechea": "Edurne Uriarte Bengoechea",  # PP
    "elejabarrieta diaz": "Gorka Elejabarrieta Díaz",  # Senador
    "elena castillo lopez": "Elena Castillo López",  # Senador
    "elena vila gomez": "Elena Vila Gómez",  # Senador
    "elias bendodo": "Elías Bendodo Benasayag",  # PP
    "elias bendodo benasayag": "Elías Bendodo Benasayag",  # PP
    "elisa garrido": "Elisa Garrido Jiménez",  # PSOE
    "elisa garrido jimenez": "Elisa Garrido Jiménez",  # PSOE
    "elma saiz delgado": "Elma Saiz Delgado",  # Gobierno
    "eloi badia": "Eloi Badia Casas",  # Sumar
    "eloi badia casas": "Eloi Badia Casas",  # Sumar
    "eloy suarez lamata": "Eloy Suárez Lamata",  # Senador
    "elvira velasco": "Elvira Velasco Morillo",  # PP
    "elvira velasco morillo": "Elvira Velasco Morillo",  # PP
    "emilia almodovar": "Emilia Almodóvar Sánchez",  # PSOE
    "emilia almodovar sanchez": "Emilia Almodóvar Sánchez",  # PSOE
    "emilio jesus del": "Emilio Jesús Del Valle Rodríguez",  # Vox
    "emilio jesus del valle": "Emilio Jesús Del Valle Rodríguez",  # Vox
    "emilio jesus del valle rodriguez": "Emilio Jesús Del Valle Rodríguez",  # Vox
    "emilio jose navarro castanedo": "Emilio José Navarro Castanedo",  # Senador
    "emilio saez": "Emilio Sáez Cruz",  # PSOE
    "emilio saez cruz": "Emilio Sáez Cruz",  # PSOE
    "engracia rivera": "Engracia Rivera Arias",  # Sumar
    "engracia rivera arias": "Engracia Rivera Arias",  # Sumar
    "enric xavier morera catala": "Enric Xavier Morera Català",  # Senador
    "enrique belda": "Enrique Belda Pérez-Pedrero",  # PP
    "enrique belda perez-pedrero": "Enrique Belda Pérez-Pedrero",  # PP
    "enrique fernando santiago": "Enrique Fernando Santiago Romero",  # Sumar
    "enrique fernando santiago romero": "Enrique Fernando Santiago Romero",  # Sumar
    "enrique ruiz escudero": "Enrique Ruiz Escudero",  # Senador
    "entrena avila": "José Entrena Ávila",  # Senador
    "ernest urtasun domenech": "Ernest Urtasun Domènech",  # Gobierno
    "espadas cejas": "Juan Espadas Cejas",  # Senador
    "espana garcia": "Arcadi España García",  # Gobierno
    "esperanza reynal": "Esperanza Reynal Reillo",  # PP
    "esperanza reynal reillo": "Esperanza Reynal Reillo",  # PP
    "estarrona elizondo": "Josu Estarrona Elizondo",  # Senador
    "estefania beltran de heredia arroniz": "Estefanía Beltrán de Heredia Arroniz",  # Senador
    "estela del carmen darocas marin": "Estela del Carmen Darocas Marín",  # Senador
    "ester munoz": "Ester Muñoz de la Iglesia",  # PP
    "ester munoz de": "Ester Muñoz de la Iglesia",  # PP
    "ester munoz de la iglesia": "Ester Muñoz de la Iglesia",  # PP
    "esteve juan": "María Dolores Esteve Juan",  # Senador
    "esther basilia del brio gonzalez": "Esther Basilia del Brío González",  # Senador
    "esther gil": "Esther Gil de Reboleño Lastortres",  # Sumar
    "esther gil de": "Esther Gil de Reboleño Lastortres",  # Sumar
    "esther gil de reboleno lastortres": "Esther Gil de Reboleño Lastortres",  # Sumar
    "esther llamazares": "Esther Llamazares Domingo",  # PP
    "esther llamazares domingo": "Esther Llamazares Domingo",  # PP
    "esther pena": "Esther Peña Camarero",  # PSOE
    "esther pena camarero": "Esther Peña Camarero",  # PSOE
    "esther rodriguez": "Esther Rodríguez Suárez",  # PSOE
    "esther rodriguez suarez": "Esther Rodríguez Suárez",  # PSOE
    "estrems fayos": "Etna Estrems Fayos",  # ERC
    "etna estrems": "Etna Estrems Fayos",  # ERC
    "etna estrems fayos": "Etna Estrems Fayos",  # ERC
    "etxano varela": "María Dolores Etxano Varela",  # Senador
    "eva maria redondo gamero": "Eva María Redondo Gamero",  # Senador
    "eva ortiz vilella": "Eva Ortiz Vilella",  # Senador
    "eva patricia bueno campanario": "Eva Patricia Bueno Campanario",  # Senador
    "evarist aznar": "Evarist Aznar Teruel",  # PP
    "evarist aznar teruel": "Evarist Aznar Teruel",  # PP
    "fabian chinea correa": "Fabián Chinea Correa",  # Senador
    "fabra part": "Alberto Fabra Part",  # PP
    "fagundez campo": "Antidio Fagúndez Campo",  # PSOE
    "fajardo palarea": "Francisco Manuel Fajardo Palarea",  # Senador
    "faneca lopez": "María Luisa Faneca López",  # PSOE
    "felix alonso": "Fèlix Alonso Cantorné",  # Sumar
    "felix alonso cantorne": "Fèlix Alonso Cantorné",  # Sumar
    "felix bolanos garcia": "Félix Bolaños García",  # PSOE
    "felix de": "Félix De las Cuevas Cortés",  # PP
    "felix de las": "Félix De las Cuevas Cortés",  # PP
    "felix de las cuevas cortes": "Félix De las Cuevas Cortés",  # PP
    "fernandez alvarez": "María Fernández Álvarez",  # Senador
    "fernandez beneitez": "Andrea Fernández Benéitez",  # PSOE
    "fernandez blanco": "José Fernández Blanco",  # Senador
    "fernandez diaz": "José Manuel Fernández Díaz",  # Senador
    "fernandez garcia": "Íñigo Fernández García",  # Senador
    "fernandez gonzalez": "María Mercedes Fernández González",  # PP
    "fernandez hernandez": "Pedro Fernández Hernández",  # Vox
    "fernandez perez": "Francisco José Fernández Pérez",  # Senador
    "fernandez rios": "Tomás Fernández Ríos",  # Vox
    "fernando adolfo gutierrez diaz de otazu": "Fernando Adolfo Gutiérrez Díaz de Otazu",  # Senador
    "fernando carbonell tatay": "Fernando Carbonell Tatay",  # Senador
    "fernando de": "Fernando De Rosa Torner",  # PP
    "fernando de rosa": "Fernando De Rosa Torner",  # PP
    "fernando de rosa torner": "Fernando De Rosa Torner",  # PP
    "fernando grande-marlaska gomez": "Fernando Grande-Marlaska Gómez",  # Gobierno
    "fernando martinez-maillo toribio": "Fernando Martínez-Maíllo Toribio",  # Senador
    "fernando priego chacon": "Fernando Priego Chacón",  # Senador
    "ferran verdejo": "Ferran Verdejo Vicente",  # PSOE
    "ferran verdejo vicente": "Ferran Verdejo Vicente",  # PSOE
    "ferrer martinez": "Juanjo Ferrer Martínez",  # Senador
    "figaredo alvarez-sala": "José María Figaredo Álvarez-Sala",  # Vox
    "flores garcia": "María Dolores Flores García",  # Senador
    "flores juberias": "Carlos Flores Juberías",  # Vox
    "floriano corrales": "Carlos Javier Floriano Corrales",  # PP
    "folch blanc": "Javier José Folch Blanc",  # PP
    "foronda vaquero": "Salvador de Foronda Vaquero",  # Senador
    "francesc xavier ten costa": "Francesc Xavier Ten Costa",  # Senador
    "francesc-marc alvaro": "Francesc-Marc Álvaro Vidal",  # ERC
    "francesc-marc alvaro vidal": "Francesc-Marc Álvaro Vidal",  # ERC
    "francina armengol": "Francina Armengol Socias",  # PSOE
    "francina armengol socias": "Francina Armengol Socias",  # PSOE
    "francisco aranda": "Francisco Aranda Vargas",  # PSOE
    "francisco aranda vargas": "Francisco Aranda Vargas",  # PSOE
    "francisco javier arenas bocanegra": "Francisco Javier Arenas Bocanegra",  # Senador
    "francisco javier bermudez carrillo": "Francisco Javier Bermúdez Carrillo",  # Senador
    "francisco javier lacalle lacalle": "Francisco Javier Lacalle Lacalle",  # Senador
    "francisco javier marquez sanchez": "Francisco Javier Márquez Sánchez",  # Senador
    "francisco javier ortega": "Francisco Javier Ortega Smith-Molina",  # Vox
    "francisco javier ortega smith-molina": "Francisco Javier Ortega Smith-Molina",  # Vox
    "francisco joaquin camacho borrego": "Francisco Joaquín Camacho Borrego",  # Senador
    "francisco jose alcaraz": "Francisco José Alcaraz Martos",  # Vox
    "francisco jose alcaraz martos": "Francisco José Alcaraz Martos",  # Vox
    "francisco jose conde": "Francisco José Conde López",  # PP
    "francisco jose conde lopez": "Francisco José Conde López",  # PP
    "francisco jose fernandez perez": "Francisco José Fernández Pérez",  # Senador
    "francisco manuel fajardo palarea": "Francisco Manuel Fajardo Palarea",  # Senador
    "francisco martin bernabe perez": "Francisco Martín Bernabé Pérez",  # Senador
    "francisco sierra": "Francisco Sierra Caballero",  # Sumar
    "francisco sierra caballero": "Francisco Sierra Caballero",  # Sumar
    "franco gonzalez": "Silvia Franco González",  # PP
    "franco pardo": "José Manuel Franco Pardo",  # Senador
    "fullaondo la cruz": "Marije Fullaondo la Cruz",  # EH Bildu
    "funez de gregorio": "Carmen Fúnez de Gregorio",  # PP
    "gabriel blanco": "Gabriel Blanco Arrúe",  # PSOE
    "gabriel blanco arrue": "Gabriel Blanco Arrúe",  # PSOE
    "gabriel colome garcia": "Gabriel Colomé García",  # Senador
    "gabriel cruz": "Gabriel Cruz Santana",  # PSOE
    "gabriel cruz santana": "Gabriel Cruz Santana",  # PSOE
    "gabriel rufian romero": "Gabriel Rufián Romero",  # ERC
    "galicia jaramillo": "Vidal Galicia Jaramillo",  # Senador
    "gallardo barrena": "Pedro Ignacio Gallardo Barrena",  # PP
    "gallego neira": "Rosa María Gallego Neira",  # Senador
    "gamarra ruiz-clavijo": "Concepción Gamarra Ruiz-Clavijo",  # PP
    "garcia adanero": "Carlos García Adanero",  # PP
    "garcia chavarria": "María Montserrat García Chavarría",  # PSOE
    "garcia diego": "Juan Carlos García Diego",  # Senador
    "garcia felix": "Manuel García Félix",  # PP
    "garcia gomez": "Mónica García Gómez",  # Gobierno
    "garcia gomis": "David García Gomis",  # Vox
    "garcia gurrutxaga": "María Luisa García Gurrutxaga",  # PSOE
    "garcia herrero": "María Ángeles García Herrero",  # Senador
    "garcia lopez": "Maribel García López",  # PSOE
    "garcia moris": "Roberto García Morís",  # PSOE
    "garcia navarro": "Miriam García Navarro",  # Senador
    "garcia rodriguez": "Alfonso García Rodríguez",  # Senador
    "garcia vega": "Jorge García Vega",  # Senador
    "garcia-escudero marquez": "Pío García-Escudero Márquez",  # Senador
    "garcia-pelayo jurado": "María José García-Pelayo Jurado",  # Senador
    "garre murcia": "Cristóbal Garre Murcia",  # PP
    "garrido jimenez": "Elisa Garrido Jiménez",  # PSOE
    "garrido tinta": "Abigail Garrido Tinta",  # Senador
    "garrido valenzuela": "Irene Garrido Valenzuela",  # PP
    "gaseni blanch": "Jordi Gaseni Blanch",  # Senador
    "gavin i valls": "Isidre Gavin i Valls",  # Junts
    "gerardo camps devesa": "Gerardo Camps Devesa",  # Senador
    "gerardo pisarello": "Gerardo Pisarello Prados",  # Sumar
    "gerardo pisarello prados": "Gerardo Pisarello Prados",  # Sumar
    "gil de reboleno lastortres": "Esther Gil de Reboleño Lastortres",  # Sumar
    "gil invernon": "Alfonso Gil Invernón",  # Senador
    "gomez enriquez": "Paloma Gómez Enríquez",  # Senador
    "gomez pina": "Luis Antonio Gómez Piña",  # PSOE
    "gonzalez benito": "Raquel González Benito",  # Senador
    "gonzalez camacho": "Juan Manuel González Camacho",  # Senador
    "gonzalez fernandez": "Víctor González Fernández",  # Senador
    "gonzalez gracia": "Juan Antonio González Gracia",  # PSOE
    "gonzalez herdaro": "Ana González Herdaro",  # PSOE
    "gonzalez lopez": "Nahuel González López",  # Sumar
    "gonzalez menendez": "Pablo González Menéndez",  # Senador
    "gonzalez munoz": "Ángel Luis González Muñoz",  # Senador
    "gonzalez vazquez": "Marta González Vázquez",  # PP
    "gonzalez-robatto perote": "Jacobo González-Robatto Perote",  # Vox
    "gonzalo jesus robles orozco": "Gonzalo Jesús Robles Orozco",  # Senador
    "gonzalo redondo": "Gonzalo Redondo Cárdenas",  # PSOE
    "gonzalo redondo cardenas": "Gonzalo Redondo Cárdenas",  # PSOE
    "gordillo moreno": "Ángel Pelayo Gordillo Moreno",  # Senador
    "gorka elejabarrieta diaz": "Gorka Elejabarrieta Díaz",  # Senador
    "gracia blanco": "Marta Gracia Blanco",  # PSOE
    "gracia ferrer": "Miguel Gracia Ferrer",  # Senador
    "grados caro": "Mónica Grados Caro",  # Senador
    "granados ruiz": "Rafael Granados Ruiz",  # Senador
    "grande-marlaska gomez": "Fernando Grande-Marlaska Gómez",  # Gobierno
    "granollers cunillera": "Inés Granollers Cunillera",  # ERC
    "guardiola salmeron": "Mirian Guardiola Salmerón",  # PP
    "guerra sanchez": "Lorena Guerra Sánchez",  # Senador
    "guia marques": "Rafael Guía Marqués",  # Senador
    "guijarro garcia": "Txema Guijarro García",  # Sumar
    "guillermo hita": "Guillermo Hita Téllez",  # PSOE
    "guillermo hita tellez": "Guillermo Hita Téllez",  # PSOE
    "guillermo mariscal": "Guillermo Mariscal Anaya",  # PP
    "guillermo mariscal anaya": "Guillermo Mariscal Anaya",  # PP
    "guinart moreno": "Lídia Guinart Moreno",  # PSOE
    "gutierrez limones": "Antonio Gutiérrez Limones",  # Senador
    "gutierrez prieto": "Sergio Gutiérrez Prieto",  # PSOE
    "gutierrez santiago": "Víctor Gutiérrez Santiago",  # PSOE
    "hector palencia": "Héctor Palencia Rubio",  # PP
    "hector palencia rubio": "Héctor Palencia Rubio",  # PP
    "heredia arroniz": "Estefanía Beltrán de Heredia Arroniz",  # Senador
    "hereu boher": "Jordi Hereu Boher",  # Gobierno
    "herminio rufino sancho": "Herminio Rufino Sancho Íñiguez",  # PSOE
    "herminio rufino sancho iniguez": "Herminio Rufino Sancho Íñiguez",  # PSOE
    "hernandez cerezo": "Paloma Hernández Cerezo",  # Senador
    "hernandez quero": "Carlos Hernández Quero",  # Vox
    "hernandez rodriguez": "Inmaculada Hernández Rodríguez",  # Senador
    "hernando fraile": "Rafael Antonio Hernando Fraile",  # PP
    "hernando garcia": "José Manuel Hernando García",  # Senador
    "herrera garcia": "Milena Herrera García",  # PSOE
    "herrero bono": "José Alberto Herrero Bono",  # PP
    "hila vargas": "José Francisco Hila Vargas",  # Senador
    "hispan iglesias de ussel": "Pablo Hispán Iglesias de Ussel",  # PP
    "hita tellez": "Guillermo Hita Téllez",  # PSOE
    "hoces iniguez": "Ignacio Hoces Íñiguez",  # Vox
    "hoyo julia": "Belén Hoyo Juliá",  # PP
    "huelva betanzos": "Amaro Huelva Betanzos",  # Senador
    "ibanez hernando": "Ángel Ibáñez Hernando",  # PP
    "ibanez mezquita": "Alberto Ibáñez Mezquita",  # Sumar
    "idoia sagastizabal": "Idoia Sagastizabal Unzetabarrenetxea",  # PNV
    "idoia sagastizabal unzetabarrenetxea": "Idoia Sagastizabal Unzetabarrenetxea",  # PNV
    "idurre bideguren gabantxo": "Idurre Bideguren Gabantxo",  # Senador
    "ignacio gil": "Ignacio Gil Lázaro",  # Vox
    "ignacio gil lazaro": "Ignacio Gil Lázaro",  # Vox
    "ignacio hoces": "Ignacio Hoces Íñiguez",  # Vox
    "ignacio hoces iniguez": "Ignacio Hoces Íñiguez",  # Vox
    "ignacio lopez": "Ignacio López Cano",  # PSOE
    "ignacio lopez cano": "Ignacio López Cano",  # PSOE
    "ignasi conesa": "Ignasi Conesa Coma",  # PSOE
    "ignasi conesa coma": "Ignasi Conesa Coma",  # PSOE
    "igotz lopez torre": "Igotz López Torre",  # Senador
    "inarritu garcia": "Jon Iñarritu García",  # EH Bildu
    "ines granollers": "Inés Granollers Cunillera",  # ERC
    "ines granollers cunillera": "Inés Granollers Cunillera",  # ERC
    "ines plaza": "Inés Plaza García",  # PSOE
    "ines plaza garcia": "Inés Plaza García",  # PSOE
    "iniesta egido": "Isabel Iniesta Egido",  # PSOE
    "inigo fernandez garcia": "Íñigo Fernández García",  # Senador
    "inmaculada hernandez rodriguez": "Inmaculada Hernández Rodríguez",  # Senador
    "inmaculada marin": "Inmaculada Marín Aparicio",  # PP
    "inmaculada marin aparicio": "Inmaculada Marín Aparicio",  # PP
    "inmaculada sanchez roca": "Inmaculada Sánchez Roca",  # Senador
    "ione belarra urteaga": "Ione Belarra Urteaga",  # Sumar
    "irene garrido": "Irene Garrido Valenzuela",  # PP
    "irene garrido valenzuela": "Irene Garrido Valenzuela",  # PP
    "irene jodar": "Irene Jódar Pérez",  # PSOE
    "irene jodar perez": "Irene Jódar Pérez",  # PSOE
    "isabel gema perez": "Isabel Gema Pérez Recuerda",  # PP
    "isabel gema perez recuerda": "Isabel Gema Pérez Recuerda",  # PP
    "isabel iniesta": "Isabel Iniesta Egido",  # PSOE
    "isabel iniesta egido": "Isabel Iniesta Egido",  # PSOE
    "isabel maria borrego": "Isabel María Borrego Cortés",  # PP
    "isabel maria borrego cortes": "Isabel María Borrego Cortés",  # PP
    "isabel maria moreno mohamed": "Isabel María Moreno Mohamed",  # Senador
    "isabel maria perez": "Isabel María Pérez Ortiz",  # PSOE
    "isabel maria perez ortiz": "Isabel María Pérez Ortiz",  # PSOE
    "isabel pozueta": "Isabel Pozueta Fernández",  # EH Bildu
    "isabel pozueta fernandez": "Isabel Pozueta Fernández",  # EH Bildu
    "isabel rodriguez garcia": "Isabel Rodríguez García",  # Gobierno
    "isaura leal": "Isaura Leal Fernández",  # PSOE
    "isaura leal fernandez": "Isaura Leal Fernández",  # PSOE
    "isidre gavin": "Isidre Gavin i Valls",  # Junts
    "isidre gavin i": "Isidre Gavin i Valls",  # Junts
    "isidre gavin i valls": "Isidre Gavin i Valls",  # Junts
    "israel roberto perez jimenez": "Israel Roberto Pérez Jiménez",  # Senador
    "ivan cacho": "Iván Cacho Isla",  # PSOE
    "ivan cacho isla": "Iván Cacho Isla",  # PSOE
    "jacobo gonzalez-robatto": "Jacobo González-Robatto Perote",  # Vox
    "jacobo gonzalez-robatto perote": "Jacobo González-Robatto Perote",  # Vox
    "jaime eduardo de olano": "Jaime Eduardo de Olano Vela",  # PP
    "jaime eduardo de olano vela": "Jaime Eduardo de Olano Vela",  # PP
    "jaime miguel de": "Jaime Miguel De los Santos González",  # PP
    "jaime miguel de los": "Jaime Miguel De los Santos González",  # PP
    "jaime miguel de los santos gonzalez": "Jaime Miguel De los Santos González",  # PP
    "jaime morales garcia": "Jaime Morales García",  # Senador
    "jan pomes lopez": "Jan Pomés López",  # Senador
    "jaume llorens monzonis": "Jaume Llorens Monzonís",  # Senador
    "javier alfonso": "Javier Alfonso Cendón",  # PSOE
    "javier alfonso cendon": "Javier Alfonso Cendón",  # PSOE
    "javier anton cacho": "Javier Antón Cacho",  # Senador
    "javier campoy monreal": "Javier Campoy Monreal",  # Senador
    "javier celaya": "Javier Celaya Brey",  # PP
    "javier celaya brey": "Javier Celaya Brey",  # PP
    "javier ignacio maroto aranzabal": "Javier Ignacio Maroto Aranzábal",  # Senador
    "javier jimenez santamaria": "Javier Jiménez Santamaría",  # Senador
    "javier jose folch": "Javier José Folch Blanc",  # PP
    "javier jose folch blanc": "Javier José Folch Blanc",  # PP
    "javier merino": "Javier Merino Martínez",  # PP
    "javier merino martinez": "Javier Merino Martínez",  # PP
    "javier noriega": "Javier Noriega Gómez",  # PP
    "javier noriega gomez": "Javier Noriega Gómez",  # PP
    "javier rodriguez": "Javier Rodríguez Palacios",  # PSOE
    "javier rodriguez palacios": "Javier Rodríguez Palacios",  # PSOE
    "javier sanchez": "Javier Sánchez Serna",  # Sumar
    "javier sanchez serna": "Javier Sánchez Serna",  # Sumar
    "javier valentin alonso coronel": "Javier Valentín Alonso Coronel",  # Senador
    "jerez antequera": "Juan Carlos Jerez Antequera",  # PSOE
    "jerez juan": "Miguel Ángel Jerez Juan",  # Senador
    "jesus caicedo bernabe": "Jesús Caicedo Bernabé",  # Senador
    "jesus caro adanero": "Jesús Caro Adanero",  # Senador
    "jesus julio carnero garcia": "Jesús Julio Carnero García",  # Senador
    "jesus mayoral": "Jesús Mayoral Pérez",  # PSOE
    "jesus mayoral perez": "Jesús Mayoral Pérez",  # PSOE
    "jimena delgado-taramona": "Jimena Delgado-Taramona Hernández",  # PP
    "jimena delgado-taramona hernandez": "Jimena Delgado-Taramona Hernández",  # PP
    "jimenez linuesa": "Beatriz Jiménez Linuesa",  # PP
    "jimenez santamaria": "Javier Jiménez Santamaría",  # Senador
    "joan baptista bague roura": "Joan Baptista Bagué Roura",  # Senador
    "joan josep queralt jimenez": "Joan Josep Queralt Jiménez",  # Senador
    "joan mesquida": "Joan Mesquida Mayans",  # PP
    "joan mesquida mayans": "Joan Mesquida Mayans",  # PP
    "joaquin martinez": "Joaquín Martínez Salmerón",  # PSOE
    "joaquin martinez salmeron": "Joaquín Martínez Salmerón",  # PSOE
    "joaquin melgarejo": "Joaquín Melgarejo Moreno",  # PP
    "joaquin melgarejo moreno": "Joaquín Melgarejo Moreno",  # PP
    "joaquin robles": "Joaquín Robles López",  # Vox
    "joaquin robles lopez": "Joaquín Robles López",  # Vox
    "jodar perez": "Irene Jódar Pérez",  # PSOE
    "jon inarritu": "Jon Iñarritu García",  # EH Bildu
    "jon inarritu garcia": "Jon Iñarritu García",  # EH Bildu
    "jonay quintero": "Jonay Quintero Hernández",  # PSOE
    "jonay quintero hernandez": "Jonay Quintero Hernández",  # PSOE
    "jorda i roura": "Teresa Jordà i Roura",  # ERC
    "jordi gaseni blanch": "Jordi Gaseni Blanch",  # Senador
    "jordi hereu boher": "Jordi Hereu Boher",  # Gobierno
    "jordi salvador": "Jordi Salvador i Duch",  # ERC
    "jordi salvador i": "Jordi Salvador i Duch",  # ERC
    "jordi salvador i duch": "Jordi Salvador i Duch",  # ERC
    "jorge campos": "Jorge Campos Asensi",  # Vox
    "jorge campos asensi": "Jorge Campos Asensi",  # Vox
    "jorge domingo martinez antolin": "Jorge Domingo Martínez Antolín",  # Senador
    "jorge garcia vega": "Jorge García Vega",  # Senador
    "jose alberto armijo navas": "José Alberto Armijo Navas",  # Senador
    "jose alberto herrero": "José Alberto Herrero Bono",  # PP
    "jose alberto herrero bono": "José Alberto Herrero Bono",  # PP
    "jose angel alonso perez": "José Ángel Alonso Pérez",  # Senador
    "jose antonio bermudez": "José Antonio Bermúdez de Castro Fernández",  # PP
    "jose antonio bermudez de": "José Antonio Bermúdez de Castro Fernández",  # PP
    "jose antonio bermudez de castro fernandez": "José Antonio Bermúdez de Castro Fernández",  # PP
    "jose antonio monago terraza": "José Antonio Monago Terraza",  # Senador
    "jose antonio rodriguez": "José Antonio Rodríguez Salas",  # PSOE
    "jose antonio rodriguez salas": "José Antonio Rodríguez Salas",  # PSOE
    "jose antonio valbuena alonso": "José Antonio Valbuena Alonso",  # Senador
    "jose crespo iglesias": "José Crespo Iglesias",  # Senador
    "jose enrique nunez": "José Enrique Núñez Guijarro",  # PP
    "jose enrique nunez guijarro": "José Enrique Núñez Guijarro",  # PP
    "jose entrena avila": "José Entrena Ávila",  # Senador
    "jose fernandez blanco": "José Fernández Blanco",  # Senador
    "jose francisco hila vargas": "José Francisco Hila Vargas",  # Senador
    "jose ignacio landaluce calleja": "José Ignacio Landaluce Calleja",  # Senador
    "jose latorre ruiz": "José Latorre Ruiz",  # Senador
    "jose losada": "José Losada Fernández",  # PSOE
    "jose losada fernandez": "José Losada Fernández",  # PSOE
    "jose luis aceves": "José Luis Aceves Galindo",  # PSOE
    "jose luis aceves galindo": "José Luis Aceves Galindo",  # PSOE
    "jose manuel albares bueno": "José Manuel Albares Bueno",  # Gobierno
    "jose manuel aranda lassa": "José Manuel Aranda Lassa",  # Senador
    "jose manuel balseiro orol": "José Manuel Balseiro Orol",  # Senador
    "jose manuel baltar blanco": "José Manuel Baltar Blanco",  # Senador
    "jose manuel barreiro fernandez": "José Manuel Barreiro Fernández",  # Senador
    "jose manuel de la vega carrera": "José Manuel de la Vega Carrera",  # Senador
    "jose manuel fernandez diaz": "José Manuel Fernández Díaz",  # Senador
    "jose manuel franco pardo": "José Manuel Franco Pardo",  # Senador
    "jose manuel hernando garcia": "José Manuel Hernando García",  # Senador
    "jose manuel rey varela": "José Manuel Rey Varela",  # Senador
    "jose manuel rodriguez gonzalez": "José Manuel Rodríguez González",  # Senador
    "jose manuel tofino perez": "José Manuel Tofiño Pérez",  # Senador
    "jose manuel velasco": "José Manuel Velasco Retamosa",  # PP
    "jose manuel velasco retamosa": "José Manuel Velasco Retamosa",  # PP
    "jose maria barrios tejero": "José María Barrios Tejero",  # Senador
    "jose maria figaredo": "José María Figaredo Álvarez-Sala",  # Vox
    "jose maria figaredo alvarez-sala": "José María Figaredo Álvarez-Sala",  # Vox
    "jose maria oleaga zalvidea": "José María Oleaga Zalvidea",  # Senador
    "jose maria sanchez": "José María Sánchez García",  # Vox
    "jose maria sanchez garcia": "José María Sánchez García",  # Vox
    "jose ramirez": "José Ramírez del Río",  # Vox
    "jose ramirez del": "José Ramírez del Río",  # Vox
    "jose ramirez del rio": "José Ramírez del Río",  # Vox
    "jose ramon diez de revenga albacete": "José Ramón Díez de Revenga Albacete",  # Senador
    "jose vicente mari": "José Vicente Marí Bosó",  # PP
    "jose vicente mari boso": "José Vicente Marí Bosó",  # PP
    "jose zaragoza": "José Zaragoza Alonso",  # PSOE
    "jose zaragoza alonso": "José Zaragoza Alonso",  # PSOE
    "joseba andoni agirretxea": "Joseba Andoni Agirretxea Urresti",  # PNV
    "joseba andoni agirretxea urresti": "Joseba Andoni Agirretxea Urresti",  # PNV
    "josep maria cervera": "Josep Maria Cervera Pinart",  # Junts
    "josep maria cervera pinart": "Josep Maria Cervera Pinart",  # Junts
    "josep maria cruset": "Josep Maria Cruset Domènech",  # Junts
    "josep maria cruset domenech": "Josep Maria Cruset Domènech",  # Junts
    "josep pages": "Josep Pagès i Massó",  # Junts
    "josep pages i": "Josep Pagès i Massó",  # Junts
    "josep pages i masso": "Josep Pagès i Massó",  # Junts
    "josep pare": "Josep Paré Aregall",  # PSOE
    "josep pare aregall": "Josep Paré Aregall",  # PSOE
    "josu estarrona elizondo": "Josu Estarrona Elizondo",  # Senador
    "juan andres bayon": "Juan Andrés Bayón Rolo",  # PP
    "juan andres bayon rolo": "Juan Andrés Bayón Rolo",  # PP
    "juan antonio gonzalez": "Juan Antonio González Gracia",  # PSOE
    "juan antonio gonzalez gracia": "Juan Antonio González Gracia",  # PSOE
    "juan antonio lorenzo": "Juan Antonio Lorenzo Cazorla",  # PSOE
    "juan antonio lorenzo cazorla": "Juan Antonio Lorenzo Cazorla",  # PSOE
    "juan antonio rojas": "Juan Antonio Rojas Manrique",  # PP
    "juan antonio rojas manrique": "Juan Antonio Rojas Manrique",  # PP
    "juan antonio sagredo marco": "Juan Antonio Sagredo Marco",  # Senador
    "juan antonio valero": "Juan Antonio Valero Morales",  # Sumar
    "juan antonio valero morales": "Juan Antonio Valero Morales",  # Sumar
    "juan bautista milian querol": "Juan Bautista Milián Querol",  # Senador
    "juan bravo": "Juan Bravo Baena",  # PP
    "juan bravo baena": "Juan Bravo Baena",  # PP
    "juan carlos garcia diego": "Juan Carlos García Diego",  # Senador
    "juan carlos jerez": "Juan Carlos Jerez Antequera",  # PSOE
    "juan carlos jerez antequera": "Juan Carlos Jerez Antequera",  # PSOE
    "juan carlos ruiz": "Juan Carlos Ruiz Boix",  # PSOE
    "juan carlos ruiz boix": "Juan Carlos Ruiz Boix",  # PSOE
    "juan carlos serrano lopez": "Juan Carlos Serrano López",  # Senador
    "juan diego requena": "Juan Diego Requena Ruiz",  # PP
    "juan diego requena ruiz": "Juan Diego Requena Ruiz",  # PP
    "juan espadas cejas": "Juan Espadas Cejas",  # Senador
    "juan francisco serrano": "Juan Francisco Serrano Martínez",  # PSOE
    "juan francisco serrano martinez": "Juan Francisco Serrano Martínez",  # PSOE
    "juan jose aizcorbe": "Juan José Aizcorbe Torra",  # Vox
    "juan jose aizcorbe torra": "Juan José Aizcorbe Torra",  # Vox
    "juan jose matari saez": "Juan José Matarí Sáez",  # Senador
    "juan jose sanz vitorio": "Juan José Sanz Vitorio",  # Senador
    "juan lobato gandarias": "Juan Lobato Gandarias",  # Senador
    "juan luis pedreno": "Juan Luis Pedreño Molina",  # PP
    "juan luis pedreno molina": "Juan Luis Pedreño Molina",  # PP
    "juan manuel avila gutierrez": "Juan Manuel Ávila Gutiérrez",  # Senador
    "juan manuel gonzalez camacho": "Juan Manuel González Camacho",  # Senador
    "juan pablo martin martin": "Juan Pablo Martín Martín",  # Senador
    "juan ramon amores garcia": "Juan Ramón Amores García",  # Senador
    "juanjo ferrer martinez": "Juanjo Ferrer Martínez",  # Senador
    "julia boada": "Júlia Boada Danés",  # Sumar
    "julia boada danes": "Júlia Boada Danés",  # Sumar
    "julia maria liberal liberal": "Julia María Liberal Liberal",  # Senador
    "julia parra": "Julia Parra Aparicio",  # PP
    "julia parra aparicio": "Julia Parra Aparicio",  # PP
    "julian nieva delgado": "Julián Nieva Delgado",  # Senador
    "kilian sanchez san juan": "Kilian Sánchez San Juan",  # Senador
    "la fuente": "María Yolanda Ibarrola de la Fuente",  # Senador
    "la torre": "María Montserrat Rivas de la Torre",  # Senador
    "lacalle lacalle": "Francisco Javier Lacalle Lacalle",  # Senador
    "lago penas": "Manuel Lago Peñas",  # Sumar
    "lamua estanol": "Marc Lamuà Estañol",  # PSOE
    "landaluce calleja": "José Ignacio Landaluce Calleja",  # Senador
    "lander martinez": "Lander Martínez Hierro",  # Sumar
    "lander martinez hierro": "Lander Martínez Hierro",  # Sumar
    "lander vera": "Adolfo Lander Vera",  # Senador
    "larra arnaiz": "Alejo Joaquín Miranda de Larra Arnaiz",  # Senador
    "latorre ruiz": "José Latorre Ruiz",  # Senador
    "laura castel fort": "Laura Castel Fort",  # Senador
    "laura maria lima": "Laura María Lima García",  # PP
    "laura maria lima garcia": "Laura María Lima García",  # PP
    "laura vergara": "Laura Vergara Román",  # Sumar
    "laura vergara roman": "Laura Vergara Román",  # Sumar
    "laureano leon rodriguez": "Laureano León Rodríguez",  # Senador
    "lazaro azorin": "Lázaro Azorín Salar",  # PSOE
    "lazaro azorin salar": "Lázaro Azorín Salar",  # PSOE
    "leal fernandez": "Isaura Leal Fernández",  # PSOE
    "legarda uriarte": "Mikel Legarda Uriarte",  # PNV
    "lemus rubiales": "Rafael Damián Lemus Rubiales",  # Senador
    "leon rodriguez": "Laureano León Rodríguez",  # Senador
    "leopoldo jeronimo sierra gallardo": "Leopoldo Jerónimo Sierra Gallardo",  # Senador
    "leyte coello": "María del Carmen Leyte Coello",  # Senador
    "liberal liberal": "Julia María Liberal Liberal",  # Senador
    "lidia guinart": "Lídia Guinart Moreno",  # PSOE
    "lidia guinart moreno": "Lídia Guinart Moreno",  # PSOE
    "lima garcia": "Laura María Lima García",  # PP
    "limon bayo": "María Eugenia Limón Bayo",  # Senador
    "llamazares domingo": "Esther Llamazares Domingo",  # PP
    "llanos de": "Llanos De Luna Tobarra",  # PP
    "llanos de luna": "Llanos De Luna Tobarra",  # PP
    "llanos de luna tobarra": "Llanos De Luna Tobarra",  # PP
    "llorens monzonis": "Jaume Llorens Monzonís",  # Senador
    "lobato gandarias": "Juan Lobato Gandarias",  # Senador
    "lopez agueda": "Óscar López Águeda",  # Gobierno
    "lopez alvarez": "Patxi López Álvarez",  # PSOE
    "lopez cano": "Ignacio López Cano",  # PSOE
    "lopez maraver": "Ángel López Maraver",  # Vox
    "lopez moya": "Antonia López Moya",  # Senador
    "lopez tagliafico": "Candela López Tagliafico",  # Sumar
    "lopez torre": "Igotz López Torre",  # Senador
    "lopez zamora": "Cristina López Zamora",  # PSOE
    "lopez zapata": "Carmen Belén López Zapata",  # Senador
    "lorena guerra sanchez": "Lorena Guerra Sánchez",  # Senador
    "lorente anaya": "Macarena Lorente Anaya",  # PP
    "lorenzo cazorla": "Juan Antonio Lorenzo Cazorla",  # PSOE
    "losada fernandez": "José Losada Fernández",  # PSOE
    "lourdes mendez": "Lourdes Méndez Monasterio",  # Vox
    "lourdes mendez monasterio": "Lourdes Méndez Monasterio",  # Vox
    "luc andre diouf": "Luc Andre Diouf Dioh",  # PSOE
    "luc andre diouf dioh": "Luc Andre Diouf Dioh",  # PSOE
    "lucas castillo rodriguez": "Lucas Castillo Rodríguez",  # Senador
    "lucia yeves leal": "Lucía Yeves Leal",  # Senador
    "luengo zapata": "Antonio Luengo Zapata",  # Senador
    "luis alfonso rey": "Luis Alfonso Rey de las Heras",  # PSOE
    "luis alfonso rey de": "Luis Alfonso Rey de las Heras",  # PSOE
    "luis alfonso rey de las heras": "Luis Alfonso Rey de las Heras",  # PSOE
    "luis antonio gomez": "Luis Antonio Gómez Piña",  # PSOE
    "luis antonio gomez pina": "Luis Antonio Gómez Piña",  # PSOE
    "luis carlos sahuquillo": "Luis Carlos Sahuquillo García",  # PSOE
    "luis carlos sahuquillo garcia": "Luis Carlos Sahuquillo García",  # PSOE
    "luis javier santamaria ruiz": "Luis Javier Santamaría Ruiz",  # Senador
    "luis jesus uribe-etxebarria apalategui": "Luis Jesús Uribe-Etxebarria Apalategui",  # Senador
    "luis maria beamonte": "Luis María Beamonte Mesa",  # PP
    "luis maria beamonte mesa": "Luis María Beamonte Mesa",  # PP
    "luis martinez-portillo subero": "Luis Martínez-Portillo Subero",  # Senador
    "luis planas puchades": "Luis Planas Puchades",  # Gobierno
    "luis rogelio rodriguez-comendador perez": "Luis Rogelio Rodríguez-Comendador Pérez",  # Senador
    "luis tudanca fernandez": "Luis Tudanca Fernández",  # Senador
    "luisa blanch fulcara": "Luisa Blanch Fulcarà",  # Senador
    "luisa fernanda rudi ubeda": "Luisa Fernanda Rudi Úbeda",  # Senador
    "luisa sanz": "Luisa Sanz Martínez",  # PSOE
    "luisa sanz martinez": "Luisa Sanz Martínez",  # PSOE
    "luna morales": "María de los Ángeles Luna Morales",  # Senador
    "macarena lorente": "Macarena Lorente Anaya",  # PP
    "macarena lorente anaya": "Macarena Lorente Anaya",  # PP
    "macarena montesinos": "Macarena Montesinos de Miguel",  # PP
    "macarena montesinos de": "Macarena Montesinos de Miguel",  # PP
    "macarena montesinos de miguel": "Macarena Montesinos de Miguel",  # PP
    "macias gata": "Alfonso Carlos Macías Gata",  # PP
    "macias mateos": "María Teresa Macías Mateos",  # Senador
    "madrenas i mir": "Marta Madrenas i Mir",  # Junts
    "madrid olmo": "Bartolomé Madrid Olmo",  # PP
    "magdaleno alegria": "Antonio Magdaleno Alegría",  # Senador
    "maldonado lopez": "Adriana Maldonado López",  # PSOE
    "manuel albares bueno": "José Manuel Albares Bueno",  # Gobierno
    "manuel arribas": "Manuel Arribas Maroto",  # PSOE
    "manuel arribas maroto": "Manuel Arribas Maroto",  # PSOE
    "manuel borrego rodriguez": "Manuel Borrego Rodríguez",  # Senador
    "manuel cobo": "Manuel Cobo Vega",  # PP
    "manuel cobo vega": "Manuel Cobo Vega",  # PP
    "manuel cruz rodriguez": "Manuel Cruz Rodríguez",  # Senador
    "manuel garcia": "Manuel García Félix",  # PP
    "manuel garcia felix": "Manuel García Félix",  # PP
    "manuel lago": "Manuel Lago Peñas",  # Sumar
    "manuel lago penas": "Manuel Lago Peñas",  # Sumar
    "manuel mariscal": "Manuel Mariscal Zabala",  # Vox
    "manuel mariscal zabala": "Manuel Mariscal Zabala",  # Vox
    "manuel miras franqueira": "Manuel Mirás Franqueira",  # Senador
    "manuel santos ruiz rivas": "Manuel Santos Ruiz Rivas",  # Senador
    "mar cotelo balmaseda": "Mar Cotelo Balmaseda",  # Senador
    "marc lamua": "Marc Lamuà Estañol",  # PSOE
    "marc lamua estanol": "Marc Lamuà Estañol",  # PSOE
    "marco gual": "María Amparo Marco Gual",  # Senador
    "marcos albaladejo gutierrez": "Marcos Albaladejo Gutiérrez",  # Senador
    "marcos moyano": "María Dolores Marcos Moyano",  # Senador
    "marcos ortega": "Milagros Marcos Ortega",  # PP
    "margarita martin": "Margarita Martín Rodríguez",  # PSOE
    "margarita martin rodriguez": "Margarita Martín Rodríguez",  # PSOE
    "margarita robles fernandez": "Margarita Robles Fernández",  # Gobierno
    "mari boso": "José Vicente Marí Bosó",  # PP
    "maria adrio": "María Adrio Taracido",  # PSOE
    "maria adrio taracido": "María Adrio Taracido",  # PSOE
    "maria amparo marco gual": "María Amparo Marco Gual",  # Senador
    "maria angeles garcia herrero": "María Ángeles García Herrero",  # Senador
    "maria araceli poblador": "María Araceli Poblador Pacheco",  # PSOE
    "maria araceli poblador pacheco": "María Araceli Poblador Pacheco",  # PSOE
    "maria arenales serrano arguello": "María Arenales Serrano Argüello",  # Senador
    "maria asuncion mayo fernandez": "María Asunción Mayo Fernández",  # Senador
    "maria carmen castilla": "María Carmen Castilla Álvarez",  # PSOE
    "maria carmen castilla alvarez": "María Carmen Castilla Álvarez",  # PSOE
    "maria carmen da silva mendez": "María Carmen da Silva Méndez",  # Senador
    "maria carmen isabel pobo sanchez": "María Carmen Isabel Pobo Sánchez",  # Senador
    "maria carmen riolobos regadera": "María Carmen Riolobos Regadera",  # Senador
    "maria carmen sanchez": "María Carmen Sánchez Díaz",  # PSOE
    "maria carmen sanchez diaz": "María Carmen Sánchez Díaz",  # PSOE
    "maria cristina rubio blasco": "María Cristina Rubio Blasco",  # Senador
    "maria cuesta": "María Cuesta Rodríguez",  # PP
    "maria cuesta rodriguez": "María Cuesta Rodríguez",  # PP
    "maria de la cabeza ruiz": "María de la Cabeza Ruiz Solás",  # Vox
    "maria de la cabeza ruiz solas": "María de la Cabeza Ruiz Solás",  # Vox
    "maria de las mercedes cantalapiedra": "María de las Mercedes Cantalapiedra Álvarez",  # PP
    "maria de las mercedes cantalapiedra alvarez": "María de las Mercedes Cantalapiedra Álvarez",  # PP
    "maria de las nieves ramirez": "María de las Nieves Ramírez Moreno",  # PSOE
    "maria de las nieves ramirez moreno": "María de las Nieves Ramírez Moreno",  # PSOE
    "maria de los angeles luna morales": "María de los Ángeles Luna Morales",  # Senador
    "maria de los reyes romero": "María de los Reyes Romero Vilches",  # Vox
    "maria de los reyes romero vilches": "María de los Reyes Romero Vilches",  # Vox
    "maria del carmen leyte coello": "María del Carmen Leyte Coello",  # Senador
    "maria del carmen perez becerra": "María del Carmen Pérez Becerra",  # Senador
    "maria del carmen silva rego": "María del Carmen Silva Rego",  # Senador
    "maria del lirio martin garcia": "María del Lirio Martín García",  # Senador
    "maria del mar san martin ibarra": "María del Mar San Martín Ibarra",  # Senador
    "maria del mar sanchez": "María del Mar Sánchez Sierra",  # PP
    "maria del mar sanchez sierra": "María del Mar Sánchez Sierra",  # PP
    "maria del mar vazquez": "María del Mar Vázquez Jiménez",  # PP
    "maria del mar vazquez jimenez": "María del Mar Vázquez Jiménez",  # PP
    "maria del pilar zamora bastante": "María del Pilar Zamora Bastante",  # Senador
    "maria del rocio divar conde": "María del Rocío Dívar Conde",  # Senador
    "maria dolores corujo": "María Dolores Corujo Berriel",  # PSOE
    "maria dolores corujo berriel": "María Dolores Corujo Berriel",  # PSOE
    "maria dolores esteve juan": "María Dolores Esteve Juan",  # Senador
    "maria dolores etxano varela": "María Dolores Etxano Varela",  # Senador
    "maria dolores flores garcia": "María Dolores Flores García",  # Senador
    "maria dolores marcos moyano": "María Dolores Marcos Moyano",  # Senador
    "maria elisa vedrina": "María Elisa Vedrina Conesa",  # PP
    "maria elisa vedrina conesa": "María Elisa Vedrina Conesa",  # PP
    "maria elvira rodriguez": "María Elvira Rodríguez Herrer",  # PP
    "maria elvira rodriguez herrer": "María Elvira Rodríguez Herrer",  # PP
    "maria emma buj sanchez": "María Emma Buj Sánchez",  # Senador
    "maria eugenia carballedo": "María Eugenia Carballedo Berlanga",  # PP
    "maria eugenia carballedo berlanga": "María Eugenia Carballedo Berlanga",  # PP
    "maria eugenia limon bayo": "María Eugenia Limón Bayo",  # Senador
    "maria eva martin perez": "María Eva Martín Pérez",  # Senador
    "maria fernandez alvarez": "María Fernández Álvarez",  # Senador
    "maria isabel moreno duque": "María Isabel Moreno Duque",  # Senador
    "maria isabel prieto": "María Isabel Prieto Serrano",  # PP
    "maria isabel prieto serrano": "María Isabel Prieto Serrano",  # PP
    "maria jesus alvarez gonzalez": "María Jesús Álvarez González",  # Senador
    "maria jesus bonilla dominguez": "María Jesús Bonilla Domínguez",  # Senador
    "maria jesus moro": "María Jesús Moro Almaraz",  # PP
    "maria jesus moro almaraz": "María Jesús Moro Almaraz",  # PP
    "maria jose garcia-pelayo jurado": "María José García-Pelayo Jurado",  # Senador
    "maria jose ortega gomez": "María José Ortega Gómez",  # Senador
    "maria jose pardo pumar": "María José Pardo Pumar",  # Senador
    "maria jose rodriguez": "María José Rodríguez de Millán Parro",  # Vox
    "maria jose rodriguez de": "María José Rodríguez de Millán Parro",  # Vox
    "maria jose rodriguez de millan parro": "María José Rodríguez de Millán Parro",  # Vox
    "maria jose villalba chavarria": "María José Villalba Chavarría",  # Senador
    "maria lourdes ramirez": "María Lourdes Ramírez Martín",  # PP
    "maria lourdes ramirez martin": "María Lourdes Ramírez Martín",  # PP
    "maria luisa faneca": "María Luisa Faneca López",  # PSOE
    "maria luisa faneca lopez": "María Luisa Faneca López",  # PSOE
    "maria luisa garcia": "María Luisa García Gurrutxaga",  # PSOE
    "maria luisa garcia gurrutxaga": "María Luisa García Gurrutxaga",  # PSOE
    "maria luz martinez": "María Luz Martínez Seijo",  # PSOE
    "maria luz martinez seijo": "María Luz Martínez Seijo",  # PSOE
    "maria mar blanco garrido": "María Mar Blanco Garrido",  # Senador
    "maria mar caballero martinez": "María Mar Caballero Martínez",  # Senador
    "maria merce perea": "María Mercè Perea i Conillas",  # PSOE
    "maria merce perea i": "María Mercè Perea i Conillas",  # PSOE
    "maria merce perea i conillas": "María Mercè Perea i Conillas",  # PSOE
    "maria mercedes fernandez": "María Mercedes Fernández González",  # PP
    "maria mercedes fernandez gonzalez": "María Mercedes Fernández González",  # PP
    "maria montserrat garcia": "María Montserrat García Chavarría",  # PSOE
    "maria montserrat garcia chavarria": "María Montserrat García Chavarría",  # PSOE
    "maria montserrat rivas de la torre": "María Montserrat Rivas de la Torre",  # Senador
    "maria patricio zafra": "María Patricio Zafra",  # Senador
    "maria pilar alia": "María Pilar Alía Aguado",  # PP
    "maria pilar alia aguado": "María Pilar Alía Aguado",  # PP
    "maria sainz": "María Sainz Martín",  # PSOE
    "maria sainz martin": "María Sainz Martín",  # PSOE
    "maria salom coll": "María Salom Coll",  # Senador
    "maria sandra moneo": "María Sandra Moneo Díez",  # PP
    "maria sandra moneo diez": "María Sandra Moneo Díez",  # PP
    "maria soledad cruz-guzman": "María Soledad Cruz-Guzmán García",  # PP
    "maria soledad cruz-guzman garcia": "María Soledad Cruz-Guzmán García",  # PP
    "maria susana marques escoin": "María Susana Marqués Escoín",  # Senador
    "maria teresa macias mateos": "María Teresa Macías Mateos",  # Senador
    "maria teresa mallada de castro": "María Teresa Mallada de Castro",  # Senador
    "maria teresa pallares pique": "Maria Teresa Pallarès Piqué",  # Senador
    "maria teresa perez esteban": "María Teresa Pérez Esteban",  # Senador
    "maria teresa ruiz-sillero bernal": "María Teresa Ruiz-Sillero Bernal",  # Senador
    "maria torres": "María Torres Tejada",  # PP
    "maria torres tejada": "María Torres Tejada",  # PP
    "maria yolanda ibarrola de la fuente": "María Yolanda Ibarrola de la Fuente",  # Senador
    "maribel garcia": "Maribel García López",  # PSOE
    "maribel garcia lopez": "Maribel García López",  # PSOE
    "maribel sanchez": "Maribel Sánchez Torregrosa",  # PP
    "maribel sanchez torregrosa": "Maribel Sánchez Torregrosa",  # PP
    "maribel vaquero": "Maribel Vaquero Montero",  # PNV
    "maribel vaquero montero": "Maribel Vaquero Montero",  # PNV
    "marije fullaondo": "Marije Fullaondo la Cruz",  # EH Bildu
    "marije fullaondo la": "Marije Fullaondo la Cruz",  # EH Bildu
    "marije fullaondo la cruz": "Marije Fullaondo la Cruz",  # EH Bildu
    "marin aparicio": "Inmaculada Marín Aparicio",  # PP
    "mario cortes": "Mario Cortés Carballo",  # PP
    "mario cortes carballo": "Mario Cortés Carballo",  # PP
    "mario soler santos": "Mario Soler Santos",  # Senador
    "mario zubiaga garate": "Mario Zubiaga Garate",  # Senador
    "mariola aranda garcia": "Mariola Aranda García",  # Senador
    "mariscal anaya": "Guillermo Mariscal Anaya",  # PP
    "mariscal zabala": "Manuel Mariscal Zabala",  # Vox
    "maroto aranzabal": "Javier Ignacio Maroto Aranzábal",  # Senador
    "marques ates": "Amador Marqués Atés",  # PSOE
    "marques escoin": "María Susana Marqués Escoín",  # Senador
    "marques palliser": "Cristóbal Marqués Palliser",  # Senador
    "marquez sanchez": "Francisco Javier Márquez Sánchez",  # Senador
    "marta arocha correa": "Marta Arocha Correa",  # Senador
    "marta gonzalez": "Marta González Vázquez",  # PP
    "marta gonzalez vazquez": "Marta González Vázquez",  # PP
    "marta gracia": "Marta Gracia Blanco",  # PSOE
    "marta gracia blanco": "Marta Gracia Blanco",  # PSOE
    "marta jorgina saavedra domenech": "Marta Jorgina Saavedra Doménech",  # Senador
    "marta madrenas": "Marta Madrenas i Mir",  # Junts
    "marta madrenas i": "Marta Madrenas i Mir",  # Junts
    "marta madrenas i mir": "Marta Madrenas i Mir",  # Junts
    "marta trenzano": "Marta Trenzano Rubio",  # PSOE
    "marta trenzano rubio": "Marta Trenzano Rubio",  # PSOE
    "marta varela": "Marta Varela Pazos",  # PP
    "marta varela pazos": "Marta Varela Pazos",  # PP
    "martin angel torres valls": "Martín Ángel Torres Valls",  # Senador
    "martin blanco": "Nacho Martín Blanco",  # PP
    "martin dominguez": "Pedro Manuel Martín Domínguez",  # Senador
    "martin garcia": "Pedro Samuel Martín García",  # PP
    "martin ibarra": "María del Mar San Martín Ibarra",  # Senador
    "martin martin": "Juan Pablo Martín Martín",  # Senador
    "martin martinez": "Andreu Martín Martínez",  # PSOE
    "martin perez": "María Eva Martín Pérez",  # Senador
    "martin rodriguez": "Margarita Martín Rodríguez",  # PSOE
    "martin sans pairuto": "Martín Sans Pairutó",  # Senador
    "martin urriza": "Carlos Martín Urriza",  # Sumar
    "martina velarde": "Martina Velarde Gómez",  # Sumar
    "martina velarde gomez": "Martina Velarde Gómez",  # Sumar
    "martinez antolin": "Jorge Domingo Martínez Antolín",  # Senador
    "martinez esteban": "Araceli Martínez Esteban",  # Senador
    "martinez gomez": "Antonio Martínez Gómez",  # PP
    "martinez hierro": "Lander Martínez Hierro",  # Sumar
    "martinez labella": "Ana Martínez Labella",  # PP
    "martinez ramirez": "Carmen Martínez Ramírez",  # PSOE
    "martinez rodriguez": "Antonio Martínez Rodríguez",  # Senador
    "martinez salmeron": "Joaquín Martínez Salmerón",  # PSOE
    "martinez seijo": "María Luz Martínez Seijo",  # PSOE
    "martinez zaragoza": "Ana Martínez Zaragoza",  # Senador
    "martinez-maillo toribio": "Fernando Martínez-Maíllo Toribio",  # Senador
    "martinez-portillo subero": "Luis Martínez-Portillo Subero",  # Senador
    "matari saez": "Juan José Matarí Sáez",  # Senador
    "matute garcia de jalon": "Oskar Matute García de Jalón",  # EH Bildu
    "matute perez": "David Matute Pérez",  # Senador
    "mayo fernandez": "María Asunción Mayo Fernández",  # Senador
    "mayoral de lamo": "Alberto Mayoral de Lamo",  # PSOE
    "mayoral perez": "Jesús Mayoral Pérez",  # PSOE
    "medina santos": "Nuria Medina Santos",  # Senador
    "melania mur sangra": "Melania Mur Sangrá",  # Senador
    "melgarejo moreno": "Joaquín Melgarejo Moreno",  # PP
    "mellado sierra": "Valle Mellado Sierra",  # PSOE
    "mendez monasterio": "Lourdes Méndez Monasterio",  # Vox
    "mercadal baquero": "Pepe Mercadal Baquero",  # PSOE
    "mercedes otero": "Mercedes Otero García",  # PSOE
    "mercedes otero garcia": "Mercedes Otero García",  # PSOE
    "merino martinez": "Javier Merino Martínez",  # PP
    "mertxe aizpurua arzallus": "Mertxe Aizpurua Arzallus",  # EH Bildu
    "mesquida mayans": "Joan Mesquida Mayans",  # PP
    "miguel angel adrian gutierrez": "Miguel Ángel Adrián Gutiérrez",  # Senador
    "miguel angel castellon rubio": "Miguel Ángel Castellón Rubio",  # Senador
    "miguel angel de la rosa martin": "Miguel Ángel de la Rosa Martín",  # Senador
    "miguel angel jerez juan": "Miguel Ángel Jerez Juan",  # Senador
    "miguel angel nacarino muriel": "Miguel Ángel Nacarino Muriel",  # Senador
    "miguel angel paniagua": "Miguel Ángel Paniagua Núñez",  # PP
    "miguel angel paniagua nunez": "Miguel Ángel Paniagua Núñez",  # PP
    "miguel angel quintanilla": "Miguel Ángel Quintanilla Navarro",  # PP
    "miguel angel quintanilla navarro": "Miguel Ángel Quintanilla Navarro",  # PP
    "miguel angel sastre": "Miguel Ángel Sastre Uyá",  # PP
    "miguel angel sastre uya": "Miguel Ángel Sastre Uyá",  # PP
    "miguel carmelo dalmau blanco": "Miguel Carmelo Dalmau Blanco",  # Senador
    "miguel gracia ferrer": "Miguel Gracia Ferrer",  # Senador
    "miguel tellado": "Miguel Tellado Filgueira",  # PP
    "miguel tellado filgueira": "Miguel Tellado Filgueira",  # PP
    "mikel legarda": "Mikel Legarda Uriarte",  # PNV
    "mikel legarda uriarte": "Mikel Legarda Uriarte",  # PNV
    "mikel otero": "Mikel Otero Gabirondo",  # EH Bildu
    "mikel otero gabirondo": "Mikel Otero Gabirondo",  # EH Bildu
    "milagros marcos": "Milagros Marcos Ortega",  # PP
    "milagros marcos ortega": "Milagros Marcos Ortega",  # PP
    "milagros tolon jaime": "Milagros Tolón Jaime",  # Gobierno
    "milena herrera": "Milena Herrera García",  # PSOE
    "milena herrera garcia": "Milena Herrera García",  # PSOE
    "milian querol": "Juan Bautista Milián Querol",  # Senador
    "miras franqueira": "Manuel Mirás Franqueira",  # Senador
    "miren uxue barcos berruezo": "Miren Uxue Barcos Berruezo",  # Senador
    "miriam bravo sanchez": "Miriam Bravo Sánchez",  # Senador
    "miriam garcia navarro": "Miriam García Navarro",  # Senador
    "miriam nogueras i": "Míriam Nogueras i Camero",  # Junts
    "miriam nogueras i camero": "Míriam Nogueras i Camero",  # Junts
    "mirian guardiola": "Mirian Guardiola Salmerón",  # PP
    "mirian guardiola salmeron": "Mirian Guardiola Salmerón",  # PP
    "modesto pose": "Modesto Pose Mesura",  # PSOE
    "modesto pose mesura": "Modesto Pose Mesura",  # PSOE
    "mogo zaro": "César Alejandro Mogo Zaro",  # Senador
    "molina leon": "Ainhoa Molina León",  # PP
    "monago terraza": "José Antonio Monago Terraza",  # Senador
    "moneo diez": "María Sandra Moneo Díez",  # PP
    "monica garcia gomez": "Mónica García Gómez",  # Gobierno
    "monica grados caro": "Mónica Grados Caro",  # Senador
    "montavez aguillaume": "Vicente Montávez Aguillaume",  # PSOE
    "montse minguez": "Montse Mínguez García",  # PSOE
    "montse minguez garcia": "Montse Mínguez García",  # PSOE
    "moraleja gomez": "Tristana María Moraleja Gómez",  # PP
    "morales alvarez": "Álvaro Morales Álvarez",  # PSOE
    "morales garcia": "Jaime Morales García",  # Senador
    "morales quesada": "Ramón Morales Quesada",  # Senador
    "morant ripoll": "Diana Morant Ripoll",  # Gobierno
    "moreno borras": "Cristina Moreno Borrás",  # PP
    "moreno duque": "María Isabel Moreno Duque",  # Senador
    "moreno fernandez": "Mª Isabel Moreno Fernández",  # PSOE
    "moreno mohamed": "Isabel María Moreno Mohamed",  # Senador
    "morera catala": "Enric Xavier Morera Català",  # Senador
    "moro almaraz": "María Jesús Moro Almaraz",  # PP
    "moscoso gonzalez": "Alfonso Carlos Moscoso González",  # Senador
    "munoz abrines": "Pedro Muñoz Abrines",  # PP
    "munoz de la iglesia": "Ester Muñoz de la Iglesia",  # PP
    "munoz martinez": "Antonio Muñoz Martínez",  # Senador
    "mur sangra": "Melania Mur Sangrá",  # Senador
    "mª isabel moreno": "Mª Isabel Moreno Fernández",  # PSOE
    "mª isabel moreno fernandez": "Mª Isabel Moreno Fernández",  # PSOE
    "nacarino muriel": "Miguel Ángel Nacarino Muriel",  # Senador
    "nacarino-brabo jimenez": "Aurora Nacarino-Brabo Jiménez",  # PP
    "nacho martin": "Nacho Martín Blanco",  # PP
    "nacho martin blanco": "Nacho Martín Blanco",  # PP
    "nahuel gonzalez": "Nahuel González López",  # Sumar
    "nahuel gonzalez lopez": "Nahuel González López",  # Sumar
    "narbona ruiz": "Cristina Narbona Ruiz",  # PSOE
    "nasarre oliva": "Begoña Nasarre Oliva",  # PSOE
    "natalia ucero perez": "Natalia Ucero Pérez",  # Senador
    "navarro castanedo": "Emilio José Navarro Castanedo",  # Senador
    "navarro lacoba": "Carmen Navarro Lacoba",  # PP
    "navarro lopez": "Pedro Navarro López",  # PP
    "nerea ahedo ceza": "Nerea Ahedo Ceza",  # Senador
    "nerea renteria": "Nerea Renteria Lasanta",  # PNV
    "nerea renteria lasanta": "Nerea Renteria Lasanta",  # PNV
    "nestor rego": "Néstor Rego Candamil",  # Mixto
    "nestor rego candamil": "Néstor Rego Candamil",  # Mixto
    "nidia maria arevalo gomez": "Nidia María Arévalo Gómez",  # Senador
    "nieva delgado": "Julián Nieva Delgado",  # Senador
    "noelia cobo": "Noelia Cobo Pérez",  # PSOE
    "noelia cobo perez": "Noelia Cobo Pérez",  # PSOE
    "noemi santana": "Noemí Santana Perera",  # Sumar
    "noemi santana perera": "Noemí Santana Perera",  # Sumar
    "nogueras i camero": "Míriam Nogueras i Camero",  # Junts
    "noriega gomez": "Javier Noriega Gómez",  # PP
    "nunez feijoo": "Alberto Núñez Feijóo",  # PP
    "nunez guijarro": "José Enrique Núñez Guijarro",  # PP
    "nuria medina santos": "Nuria Medina Santos",  # Senador
    "nuria rovira costas": "Núria Rovira Costas",  # Senador
    "obdulia taboadela": "Obdulia Taboadela Álvarez",  # PSOE
    "obdulia taboadela alvarez": "Obdulia Taboadela Álvarez",  # PSOE
    "ogou i corbi": "Viviane Ogou i Corbi",  # Sumar
    "olaia duarte lopez": "Olaia Duarte López",  # Senador
    "olano vela": "Jaime Eduardo de Olano Vela",  # PP
    "oleaga zalvidea": "José María Oleaga Zalvidea",  # Senador
    "olvido de": "Olvido De la Rosa Baena",  # PSOE
    "olvido de la": "Olvido De la Rosa Baena",  # PSOE
    "olvido de la rosa baena": "Olvido De la Rosa Baena",  # PSOE
    "ona martin": "Dionís Oña Martín",  # Senador
    "oriol almiron": "Oriol Almirón Ruiz",  # PSOE
    "oriol almiron ruiz": "Oriol Almirón Ruiz",  # PSOE
    "ortega gomez": "María José Ortega Gómez",  # Senador
    "ortega smith-molina": "Francisco Javier Ortega Smith-Molina",  # Vox
    "ortiz vilella": "Eva Ortiz Vilella",  # Senador
    "oscar clavell": "Óscar Clavell López",  # PP
    "oscar clavell lopez": "Óscar Clavell López",  # PP
    "oscar lopez agueda": "Óscar López Águeda",  # Gobierno
    "oscar puente santiago": "Óscar Puente Santiago",  # PSOE
    "oscar ramajo": "Óscar Ramajo Prada",  # PP
    "oscar ramajo prada": "Óscar Ramajo Prada",  # PP
    "oskar matute": "Oskar Matute García de Jalón",  # EH Bildu
    "oskar matute garcia": "Oskar Matute García de Jalón",  # EH Bildu
    "oskar matute garcia de jalon": "Oskar Matute García de Jalón",  # EH Bildu
    "otero gabirondo": "Mikel Otero Gabirondo",  # EH Bildu
    "otero garcia": "Mercedes Otero García",  # PSOE
    "otero rodriguez": "Patricia Otero Rodríguez",  # PSOE
    "pablo antunano": "Pablo Antuñano Colina",  # PSOE
    "pablo antunano colina": "Pablo Antuñano Colina",  # PSOE
    "pablo bustinduy amador": "Pablo Bustinduy Amador",  # Gobierno
    "pablo gonzalez menendez": "Pablo González Menéndez",  # Senador
    "pablo hispan": "Pablo Hispán Iglesias de Ussel",  # PP
    "pablo hispan iglesias": "Pablo Hispán Iglesias de Ussel",  # PP
    "pablo hispan iglesias de ussel": "Pablo Hispán Iglesias de Ussel",  # PP
    "pablo perez": "Pablo Pérez Coronado",  # PP
    "pablo perez coronado": "Pablo Pérez Coronado",  # PP
    "pablo saez": "Pablo Sáez Alonso-Muñumer",  # Vox
    "pablo saez alonso-munumer": "Pablo Sáez Alonso-Muñumer",  # Vox
    "pachon martin": "Brígida Pachón Martín",  # PSOE
    "pagador lopez": "Carmen Pagador López",  # Senador
    "pages i masso": "Josep Pagès i Massó",  # Junts
    "palencia rubio": "Héctor Palencia Rubio",  # PP
    "pallares pique": "Maria Teresa Pallarès Piqué",  # Senador
    "paloma gomez enriquez": "Paloma Gómez Enríquez",  # Senador
    "paloma hernandez cerezo": "Paloma Hernández Cerezo",  # Senador
    "paloma ines sanz jeronimo": "Paloma Inés Sanz Jerónimo",  # Senador
    "paloma martin martin": "Paloma Martín Martín",  # Senador
    "paniagua nunez": "Miguel Ángel Paniagua Núñez",  # PP
    "pardo pumar": "María José Pardo Pumar",  # Senador
    "pare aregall": "Josep Paré Aregall",  # PSOE
    "parra aparicio": "Julia Parra Aparicio",  # PP
    "parra gallego": "Agustín Parra Gallego",  # PP
    "pascual rocamora": "Sandra Pascual Rocamora",  # PP
    "pasion gador romero garcia": "Pasión Gador Romero García",  # Senador
    "patricia blanquer": "Patricia Blanquer Alcaraz",  # PSOE
    "patricia blanquer alcaraz": "Patricia Blanquer Alcaraz",  # PSOE
    "patricia otero": "Patricia Otero Rodríguez",  # PSOE
    "patricia otero rodriguez": "Patricia Otero Rodríguez",  # PSOE
    "patricia rodriguez": "Patricia Rodríguez Calleja",  # PP
    "patricia rodriguez calleja": "Patricia Rodríguez Calleja",  # PP
    "patricia rueda": "Patricia Rueda Perelló",  # Vox
    "patricia rueda perello": "Patricia Rueda Perelló",  # Vox
    "patricio zafra": "María Patricio Zafra",  # Senador
    "patxi lopez alvarez": "Patxi López Álvarez",  # PSOE
    "paula alicia somalo garcia": "Paula Alicia Somalo García",  # Senador
    "pedreno molina": "Juan Luis Pedreño Molina",  # PP
    "pedro fernandez": "Pedro Fernández Hernández",  # Vox
    "pedro fernandez hernandez": "Pedro Fernández Hernández",  # Vox
    "pedro ignacio gallardo": "Pedro Ignacio Gallardo Barrena",  # PP
    "pedro ignacio gallardo barrena": "Pedro Ignacio Gallardo Barrena",  # PP
    "pedro manuel martin dominguez": "Pedro Manuel Martín Domínguez",  # Senador
    "pedro manuel rollan ojeda": "Pedro Manuel Rollán Ojeda",  # Senador
    "pedro manuel sangines gutierrez": "Pedro Manuel Sanginés Gutiérrez",  # Senador
    "pedro munoz": "Pedro Muñoz Abrines",  # PP
    "pedro munoz abrines": "Pedro Muñoz Abrines",  # PP
    "pedro navarro": "Pedro Navarro López",  # PP
    "pedro navarro lopez": "Pedro Navarro López",  # PP
    "pedro puy": "Pedro Puy Fraga",  # PP
    "pedro puy fraga": "Pedro Puy Fraga",  # PP
    "pedro samuel martin": "Pedro Samuel Martín García",  # PP
    "pedro samuel martin garcia": "Pedro Samuel Martín García",  # PP
    "pedro sanchez perez-castejon": "Pedro Sánchez Pérez-Castejón",  # PSOE
    "pena camarero": "Esther Peña Camarero",  # PSOE
    "pepe mercadal": "Pepe Mercadal Baquero",  # PSOE
    "pepe mercadal baquero": "Pepe Mercadal Baquero",  # PSOE
    "pere joan pons sampietro": "Pere Joan Pons Sampietro",  # Senador
    "perea i conillas": "María Mercè Perea i Conillas",  # PSOE
    "perez becerra": "María del Carmen Pérez Becerra",  # Senador
    "perez coronado": "Pablo Pérez Coronado",  # PP
    "perez esteban": "María Teresa Pérez Esteban",  # Senador
    "perez jimenez": "Israel Roberto Pérez Jiménez",  # Senador
    "perez lopez": "Álvaro Pérez López",  # PP
    "perez ortiz": "Isabel María Pérez Ortiz",  # PSOE
    "perez osma": "Daniel Pérez Osma",  # PP
    "perez recuerda": "Isabel Gema Pérez Recuerda",  # PP
    "piedad sanchez garcia": "Piedad Sánchez García",  # Senador
    "pilar calvo": "Pilar Calvo Gómez",  # Junts
    "pilar calvo gomez": "Pilar Calvo Gómez",  # Junts
    "pilar milagros rojo noguera": "Pilar Milagros Rojo Noguera",  # Senador
    "pilar vallugera": "Pilar Vallugera Balañà",  # ERC
    "pilar vallugera balana": "Pilar Vallugera Balañà",  # ERC
    "pio garcia-escudero marquez": "Pío García-Escudero Márquez",  # Senador
    "pisarello prados": "Gerardo Pisarello Prados",  # Sumar
    "planas puchades": "Luis Planas Puchades",  # Gobierno
    "plaza garcia": "Inés Plaza García",  # PSOE
    "poblador pacheco": "María Araceli Poblador Pacheco",  # PSOE
    "pobo sanchez": "María Carmen Isabel Pobo Sánchez",  # Senador
    "polanco rebolleda": "Carlos Alfonso Polanco Rebolleda",  # Senador
    "pomes lopez": "Jan Pomés López",  # Senador
    "pons sampietro": "Pere Joan Pons Sampietro",  # Senador
    "pose mesura": "Modesto Pose Mesura",  # PSOE
    "poveda zapata": "Antonio Poveda Zapata",  # Senador
    "pozueta fernandez": "Isabel Pozueta Fernández",  # EH Bildu
    "priego chacon": "Fernando Priego Chacón",  # Senador
    "prieto serrano": "María Isabel Prieto Serrano",  # PP
    "prieto valencia": "Benjamín Prieto Valencia",  # Senador
    "puente santiago": "Óscar Puente Santiago",  # PSOE
    "pujol bonell": "Eduard Pujol Bonell",  # Senador
    "puy fraga": "Pedro Puy Fraga",  # PP
    "queralt jimenez": "Joan Josep Queralt Jiménez",  # Senador
    "quintana carballo": "Rosa Quintana Carballo",  # PP
    "quintanilla navarro": "Miguel Ángel Quintanilla Navarro",  # PP
    "quintero hernandez": "Jonay Quintero Hernández",  # PSOE
    "rafael antonio hernando": "Rafael Antonio Hernando Fraile",  # PP
    "rafael antonio hernando fraile": "Rafael Antonio Hernando Fraile",  # PP
    "rafael benigno belmonte": "Rafael Benigno Belmonte Gómez",  # PP
    "rafael benigno belmonte gomez": "Rafael Benigno Belmonte Gómez",  # PP
    "rafael cofino": "Rafael Cofiño Fernández",  # Sumar
    "rafael cofino fernandez": "Rafael Cofiño Fernández",  # Sumar
    "rafael damian lemus rubiales": "Rafael Damián Lemus Rubiales",  # Senador
    "rafael granados ruiz": "Rafael Granados Ruiz",  # Senador
    "rafael guia marques": "Rafael Guía Marqués",  # Senador
    "rafael rodriguez villarino": "Rafael Rodríguez Villarino",  # Senador
    "rafael simancas": "Rafael Simancas Simancas",  # PSOE
    "rafael simancas simancas": "Rafael Simancas Simancas",  # PSOE
    "rafaela crespin": "Rafaela Crespín Rubio",  # PSOE
    "rafaela crespin rubio": "Rafaela Crespín Rubio",  # PSOE
    "rafaela romero": "Rafaela Romero Pozo",  # PSOE
    "rafaela romero pozo": "Rafaela Romero Pozo",  # PSOE
    "rallo lombarte": "Artemi Rallo Lombarte",  # PSOE
    "ramajo prada": "Óscar Ramajo Prada",  # PP
    "ramirez carner": "Arnau Ramírez Carner",  # PSOE
    "ramirez del rio": "José Ramírez del Río",  # Vox
    "ramirez martin": "María Lourdes Ramírez Martín",  # PP
    "ramirez moreno": "María de las Nieves Ramírez Moreno",  # PSOE
    "ramon morales quesada": "Ramón Morales Quesada",  # Senador
    "raquel clemente": "Raquel Clemente Muñoz",  # PP
    "raquel clemente munoz": "Raquel Clemente Muñoz",  # PP
    "raquel gonzalez benito": "Raquel González Benito",  # Senador
    "raul cuevas": "Raúl Cuevas Larrosa",  # PP
    "raul cuevas larrosa": "Raúl Cuevas Larrosa",  # PP
    "raul dalmacio valero mejia": "Raúl Dalmacio Valero Mejía",  # Senador
    "raul diaz": "Raúl Díaz Marín",  # PSOE
    "raul diaz marin": "Raúl Díaz Marín",  # PSOE
    "recas martin": "Alda Recas Martín",  # Sumar
    "redondo cardenas": "Gonzalo Redondo Cárdenas",  # PSOE
    "redondo gamero": "Eva María Redondo Gamero",  # Senador
    "redondo garcia": "Ana Redondo García",  # Gobierno
    "rego candamil": "Néstor Rego Candamil",  # Mixto
    "renteria lasanta": "Nerea Renteria Lasanta",  # PNV
    "requena ruiz": "Juan Diego Requena Ruiz",  # PP
    "revenga albacete": "José Ramón Díez de Revenga Albacete",  # Senador
    "rey de las heras": "Luis Alfonso Rey de las Heras",  # PSOE
    "rey varela": "José Manuel Rey Varela",  # Senador
    "reynal reillo": "Esperanza Reynal Reillo",  # PP
    "ricardo chamorro": "Ricardo Chamorro Delmo",  # Vox
    "ricardo chamorro delmo": "Ricardo Chamorro Delmo",  # Vox
    "ricardo tarno": "Ricardo Tarno Blanco",  # PP
    "ricardo tarno blanco": "Ricardo Tarno Blanco",  # PP
    "riolobos regadera": "María Carmen Riolobos Regadera",  # Senador
    "rivera arias": "Engracia Rivera Arias",  # Sumar
    "rives arcayna": "Caridad Rives Arcayna",  # PSOE
    "roberto garcia": "Roberto García Morís",  # PSOE
    "roberto garcia moris": "Roberto García Morís",  # PSOE
    "robles fernandez": "Margarita Robles Fernández",  # Gobierno
    "robles lopez": "Joaquín Robles López",  # Vox
    "robles orozco": "Gonzalo Jesús Robles Orozco",  # Senador
    "rocio aguirre": "Rocío Aguirre Gil de Biedma",  # Vox
    "rocio aguirre gil": "Rocío Aguirre Gil de Biedma",  # Vox
    "rocio aguirre gil de biedma": "Rocío Aguirre Gil de Biedma",  # Vox
    "rocio briones morales": "Rocío Briones Morales",  # Senador
    "rocio de": "Rocío De Meer Méndez",  # Vox
    "rocio de meer": "Rocío De Meer Méndez",  # Vox
    "rocio de meer mendez": "Rocío De Meer Méndez",  # Vox
    "rodriguez almeida": "Andrés Alberto Rodríguez Almeida",  # Vox
    "rodriguez calleja": "Patricia Rodríguez Calleja",  # PP
    "rodriguez de millan parro": "María José Rodríguez de Millán Parro",  # Vox
    "rodriguez garcia": "Isabel Rodríguez García",  # Gobierno
    "rodriguez gomez de celis": "Alfonso Rodríguez Gómez de Celis",  # PSOE
    "rodriguez gonzalez": "José Manuel Rodríguez González",  # Senador
    "rodriguez herrer": "María Elvira Rodríguez Herrer",  # PP
    "rodriguez palacios": "Javier Rodríguez Palacios",  # PSOE
    "rodriguez salas": "José Antonio Rodríguez Salas",  # PSOE
    "rodriguez serra": "Santi Rodríguez Serra",  # PP
    "rodriguez suarez": "Esther Rodríguez Suárez",  # PSOE
    "rodriguez villarino": "Rafael Rodríguez Villarino",  # Senador
    "rodriguez-comendador perez": "Luis Rogelio Rodríguez-Comendador Pérez",  # Senador
    "rojas garcia": "Carlos Rojas García",  # PP
    "rojas manrique": "Juan Antonio Rojas Manrique",  # PP
    "rojo blas": "Alberto Rojo Blas",  # PSOE
    "rojo noguera": "Pilar Milagros Rojo Noguera",  # Senador
    "rollan ojeda": "Pedro Manuel Rollán Ojeda",  # Senador
    "roman jasanada": "Antonio Román Jasanada",  # PP
    "romero garcia": "Pasión Gador Romero García",  # Senador
    "romero hernandez": "Carmelo Romero Hernández",  # Senador
    "romero sanchez": "Rosa María Romero Sánchez",  # Senador
    "romero vilches": "María de los Reyes Romero Vilches",  # Vox
    "ros martinez": "Susana Ros Martínez",  # PSOE
    "rosa faustina viera fernandez": "Rosa Faustina Viera Fernández",  # Senador
    "rosa maria aldea gomez": "Rosa María Aldea Gómez",  # Senador
    "rosa maria gallego neira": "Rosa María Gallego Neira",  # Senador
    "rosa maria romero sanchez": "Rosa María Romero Sánchez",  # Senador
    "rosa maria sanchez gandara": "Rosa María Sánchez Gándara",  # Senador
    "rosa martin": "Miguel Ángel de la Rosa Martín",  # Senador
    "rosa quintana": "Rosa Quintana Carballo",  # PP
    "rosa quintana carballo": "Rosa Quintana Carballo",  # PP
    "rovira costas": "Núria Rovira Costas",  # Senador
    "rubio blasco": "María Cristina Rubio Blasco",  # Senador
    "rudi ubeda": "Luisa Fernanda Rudi Úbeda",  # Senador
    "rueda perello": "Patricia Rueda Perelló",  # Vox
    "rufian romero": "Gabriel Rufián Romero",  # ERC
    "ruiz boix": "Juan Carlos Ruiz Boix",  # PSOE
    "ruiz de diego": "Víctor Javier Ruiz de Diego",  # PSOE
    "ruiz escudero": "Enrique Ruiz Escudero",  # Senador
    "ruiz rivas": "Manuel Santos Ruiz Rivas",  # Senador
    "ruiz solas": "María de la Cabeza Ruiz Solás",  # Vox
    "ruiz-sillero bernal": "María Teresa Ruiz-Sillero Bernal",  # Senador
    "saavedra domenech": "Marta Jorgina Saavedra Doménech",  # Senador
    "saez alonso-munumer": "Pablo Sáez Alonso-Muñumer",  # Vox
    "saez cruz": "Emilio Sáez Cruz",  # PSOE
    "sagastizabal unzetabarrenetxea": "Idoia Sagastizabal Unzetabarrenetxea",  # PNV
    "sagredo marco": "Juan Antonio Sagredo Marco",  # Senador
    "sahuquillo garcia": "Luis Carlos Sahuquillo García",  # PSOE
    "sainz martin": "María Sainz Martín",  # PSOE
    "saiz delgado": "Elma Saiz Delgado",  # Gobierno
    "salom coll": "María Salom Coll",  # Senador
    "salvador de foronda vaquero": "Salvador de Foronda Vaquero",  # Senador
    "salvador i duch": "Jordi Salvador i Duch",  # ERC
    "salvador vidal varela": "Salvador Vidal Varela",  # Senador
    "san juan": "Kilian Sánchez San Juan",  # Senador
    "sanchez diaz": "María Carmen Sánchez Díaz",  # PSOE
    "sanchez gandara": "Rosa María Sánchez Gándara",  # Senador
    "sanchez garcia": "José María Sánchez García",  # Vox
    "sanchez ojeda": "Carlos Alberto Sánchez Ojeda",  # PP
    "sanchez perez": "César Sánchez Pérez",  # PP
    "sanchez perez-castejon": "Pedro Sánchez Pérez-Castejón",  # PSOE
    "sanchez roca": "Inmaculada Sánchez Roca",  # Senador
    "sanchez serna": "Javier Sánchez Serna",  # Sumar
    "sanchez sierra": "María del Mar Sánchez Sierra",  # PP
    "sanchez torregrosa": "Maribel Sánchez Torregrosa",  # PP
    "sancho iniguez": "Herminio Rufino Sancho Íñiguez",  # PSOE
    "sandra pascual": "Sandra Pascual Rocamora",  # PP
    "sandra pascual rocamora": "Sandra Pascual Rocamora",  # PP
    "sangines gutierrez": "Pedro Manuel Sanginés Gutiérrez",  # Senador
    "sans pairuto": "Martín Sans Pairutó",  # Senador
    "santamaria ruiz": "Luis Javier Santamaría Ruiz",  # Senador
    "santana aguilera": "Ada Santana Aguilera",  # PSOE
    "santana perera": "Noemí Santana Perera",  # Sumar
    "santi rodriguez": "Santi Rodríguez Serra",  # PP
    "santi rodriguez serra": "Santi Rodríguez Serra",  # PP
    "santiago abascal": "Santiago Abascal Conde",  # Vox
    "santiago abascal conde": "Santiago Abascal Conde",  # Vox
    "santos maraver": "Agustín Santos Maraver",  # Sumar
    "sanz jeronimo": "Paloma Inés Sanz Jerónimo",  # Senador
    "sanz martinez": "Luisa Sanz Martínez",  # PSOE
    "sanz vitorio": "Juan José Sanz Vitorio",  # Senador
    "sara aagesen munoz": "Sara Aagesen Muñoz",  # Gobierno
    "sara bailac ardanuy": "Sara Bailac Ardanuy",  # Senador
    "sarria morell": "Vicent Manuel Sarrià Morell",  # PSOE
    "sastre uya": "Miguel Ángel Sastre Uyá",  # PP
    "sayas lopez": "Sergio Sayas López",  # PP
    "secundino caso roiz": "Secundino Caso Roiz",  # Senador
    "semper pascual": "Borja Sémper Pascual",  # PP
    "senderos oraa": "Daniel Senderos Oraá",  # PSOE
    "sergio barasoain rodrigo": "Sergio Barasoain Rodrigo",  # Senador
    "sergio gutierrez": "Sergio Gutiérrez Prieto",  # PSOE
    "sergio gutierrez prieto": "Sergio Gutiérrez Prieto",  # PSOE
    "sergio sayas": "Sergio Sayas López",  # PP
    "sergio sayas lopez": "Sergio Sayas López",  # PP
    "serrada pariente": "David Serrada Pariente",  # PSOE
    "serrano arguello": "María Arenales Serrano Argüello",  # Senador
    "serrano lopez": "Juan Carlos Serrano López",  # Senador
    "serrano martinez": "Juan Francisco Serrano Martínez",  # PSOE
    "serrano sanchez-capuchino": "Alfonso Carlos Serrano Sánchez-Capuchino",  # Senador
    "severiano angel cuesta alonso": "Severiano Ángel Cuesta Alonso",  # Senador
    "sierra caballero": "Francisco Sierra Caballero",  # Sumar
    "sierra gallardo": "Leopoldo Jerónimo Sierra Gallardo",  # Senador
    "silva mendez": "María Carmen da Silva Méndez",  # Senador
    "silva rego": "María del Carmen Silva Rego",  # Senador
    "silvan rodriguez": "Antonio Silván Rodríguez",  # Senador
    "silverio arguelles": "Silverio Argüelles García",  # PP
    "silverio arguelles garcia": "Silverio Argüelles García",  # PP
    "silvia franco": "Silvia Franco González",  # PP
    "silvia franco gonzalez": "Silvia Franco González",  # PP
    "simancas simancas": "Rafael Simancas Simancas",  # PSOE
    "simarro vicens": "Carlos Simarro Vicens",  # PP
    "simon valentin bueno vargas": "Simón Valentín Bueno Vargas",  # Senador
    "sira rego": "Sira Rego",  # Gobierno
    "sofia acedo": "Sofía Acedo Reyes",  # PP
    "sofia acedo reyes": "Sofía Acedo Reyes",  # PP
    "soldevilla novials": "Alba Soldevilla Novials",  # PSOE
    "soler mur": "Alejandro Soler Mur",  # PSOE
    "soler santos": "Mario Soler Santos",  # Senador
    "somalo garcia": "Paula Alicia Somalo García",  # Senador
    "suarez lamata": "Eloy Suárez Lamata",  # Senador
    "susana diaz pacheco": "Susana Díaz Pacheco",  # Senador
    "susana ros": "Susana Ros Martínez",  # PSOE
    "susana ros martinez": "Susana Ros Martínez",  # PSOE
    "taboadela alvarez": "Obdulia Taboadela Álvarez",  # PSOE
    "tarno blanco": "Ricardo Tarno Blanco",  # PP
    "tellado filgueira": "Miguel Tellado Filgueira",  # PP
    "ten costa": "Francesc Xavier Ten Costa",  # Senador
    "teniente sanchez": "Cristina Teniente Sánchez",  # PP
    "teresa jorda": "Teresa Jordà i Roura",  # ERC
    "teresa jorda i": "Teresa Jordà i Roura",  # ERC
    "teresa jorda i roura": "Teresa Jordà i Roura",  # ERC
    "teresa maria belmonte sanchez": "Teresa María Belmonte Sánchez",  # Senador
    "teslem andala": "Teslem Andala Ubbi",  # Sumar
    "teslem andala ubbi": "Teslem Andala Ubbi",  # Sumar
    "tirado ochoa": "Vicente Tirado Ochoa",  # Senador
    "tofino perez": "José Manuel Tofiño Pérez",  # Senador
    "tolon jaime": "Milagros Tolón Jaime",  # Gobierno
    "tomas cabezon": "Tomás Cabezón Casas",  # PP
    "tomas cabezon casas": "Tomás Cabezón Casas",  # PP
    "tomas fernandez": "Tomás Fernández Ríos",  # Vox
    "tomas fernandez rios": "Tomás Fernández Ríos",  # Vox
    "tomas olivares": "Violante Tomás Olivares",  # PP
    "torralba valiente": "Carmen Torralba Valiente",  # Senador
    "torres perez": "Ángel Víctor Torres Pérez",  # Gobierno
    "torres tejada": "María Torres Tejada",  # PP
    "torres valencoso": "Amparo Torres Valencoso",  # Senador
    "torres valls": "Martín Ángel Torres Valls",  # Senador
    "trenzano rubio": "Marta Trenzano Rubio",  # PSOE
    "trinidad carmen argota": "Trinidad Carmen Argota Castro",  # PSOE
    "trinidad carmen argota castro": "Trinidad Carmen Argota Castro",  # PSOE
    "tristana maria moraleja": "Tristana María Moraleja Gómez",  # PP
    "tristana maria moraleja gomez": "Tristana María Moraleja Gómez",  # PP
    "tudanca fernandez": "Luis Tudanca Fernández",  # Senador
    "txema guijarro": "Txema Guijarro García",  # Sumar
    "txema guijarro garcia": "Txema Guijarro García",  # Sumar
    "ucero perez": "Natalia Ucero Pérez",  # Senador
    "uriarte bengoechea": "Edurne Uriarte Bengoechea",  # PP
    "uribe-etxebarria apalategui": "Luis Jesús Uribe-Etxebarria Apalategui",  # Senador
    "urtasun domenech": "Ernest Urtasun Domènech",  # Gobierno
    "valbuena alonso": "José Antonio Valbuena Alonso",  # Senador
    "valero mejia": "Raúl Dalmacio Valero Mejía",  # Senador
    "valero morales": "Juan Antonio Valero Morales",  # Sumar
    "valle mellado": "Valle Mellado Sierra",  # PSOE
    "valle mellado sierra": "Valle Mellado Sierra",  # PSOE
    "vallugera balana": "Pilar Vallugera Balañà",  # ERC
    "varela pazos": "Marta Varela Pazos",  # PP
    "vazquez blanco": "Ana Belén Vázquez Blanco",  # PP
    "vazquez jimenez": "María del Mar Vázquez Jiménez",  # PP
    "vedrina conesa": "María Elisa Vedrina Conesa",  # PP
    "vega carrera": "José Manuel de la Vega Carrera",  # Senador
    "velarde gomez": "Martina Velarde Gómez",  # Sumar
    "velasco morillo": "Elvira Velasco Morillo",  # PP
    "velasco retamosa": "José Manuel Velasco Retamosa",  # PP
    "verano dominguez": "Bella Verano Domínguez",  # PP
    "verdejo vicente": "Ferran Verdejo Vicente",  # PSOE
    "vergara roman": "Laura Vergara Román",  # Sumar
    "veronica maria casal miguez": "Verónica María Casal Míguez",  # Senador
    "veronica martinez barbero": "Verónica Martínez Barbero",  # Sumar
    "vicenc vidal": "Vicenç Vidal Matas",  # Sumar
    "vicenc vidal matas": "Vicenç Vidal Matas",  # Sumar
    "vicent manuel sarria": "Vicent Manuel Sarrià Morell",  # PSOE
    "vicent manuel sarria morell": "Vicent Manuel Sarrià Morell",  # PSOE
    "vicente azpitarte perez": "Vicente Azpitarte Pérez",  # Senador
    "vicente montavez": "Vicente Montávez Aguillaume",  # PSOE
    "vicente montavez aguillaume": "Vicente Montávez Aguillaume",  # PSOE
    "vicente tirado ochoa": "Vicente Tirado Ochoa",  # Senador
    "victor camino": "Víctor Camino Miñana",  # PSOE
    "victor camino minana": "Víctor Camino Miñana",  # PSOE
    "victor gonzalez fernandez": "Víctor González Fernández",  # Senador
    "victor gutierrez": "Víctor Gutiérrez Santiago",  # PSOE
    "victor gutierrez santiago": "Víctor Gutiérrez Santiago",  # PSOE
    "victor javier ruiz": "Víctor Javier Ruiz de Diego",  # PSOE
    "victor javier ruiz de": "Víctor Javier Ruiz de Diego",  # PSOE
    "victor javier ruiz de diego": "Víctor Javier Ruiz de Diego",  # PSOE
    "victor torres perez": "Ángel Víctor Torres Pérez",  # Gobierno
    "vidal galicia jaramillo": "Vidal Galicia Jaramillo",  # Senador
    "vidal matas": "Vicenç Vidal Matas",  # Sumar
    "vidal saez": "Aina Vidal Sáez",  # Sumar
    "vidal varela": "Salvador Vidal Varela",  # Senador
    "viera fernandez": "Rosa Faustina Viera Fernández",  # Senador
    "vila gomez": "Elena Vila Gómez",  # Senador
    "villalba chavarria": "María José Villalba Chavarría",  # Senador
    "violante tomas": "Violante Tomás Olivares",  # PP
    "violante tomas olivares": "Violante Tomás Olivares",  # PP
    "viviane ogou": "Viviane Ogou i Corbi",  # Sumar
    "viviane ogou i": "Viviane Ogou i Corbi",  # Sumar
    "viviane ogou i corbi": "Viviane Ogou i Corbi",  # Sumar
    "yecora roca": "Carlos Yécora Roca",  # Senador
    "yeves leal": "Lucía Yeves Leal",  # Senador
    "yolanda diaz perez": "Yolanda Díaz Pérez",  # Sumar
    "zamora bastante": "María del Pilar Zamora Bastante",  # Senador
    "zaragoza alonso": "José Zaragoza Alonso",  # PSOE
    "zubiaga garate": "Mario Zubiaga Garate",  # Senador
}


# Aliases de apellido único — SOLO válidos dentro de presentación explícita.
# Lista REDUCIDA: solo apellidos que son genuinamente únicos en el Congreso XV.
# Nombres con homónimos conocidos (sanchez, montero, esteban, lopez) NO están aquí.
ALIASES_APELLIDO_UNICO = {
    "rufian": "Gabriel Rufián",
    "nogueras": "Miriam Nogueras",
    "belarra": "Ione Belarra",
    "urteaga": "Ione Belarra",
    "otegi": "Arnaldo Otegi",
    "aizpurua": "Mertxe Aizpurua Arzallus",
    "valido": "Cristina Valido García",
    "catalan": "Alberto Catalán Higueras",
    "madrera": "Noèlia Madrera Simil",
    "micomici": "Àgueda Micó Micó",
    "mico": "Àgueda Micó Micó",
    "galindo": "Enrique Santiago",
    "vaquero": "Maribel Vaquero Montero",
    "montoro": "Cristóbal Montoro",
    "figaredo": "José María Figaredo",
    "feijoo": "Alberto Núñez Feijóo",
    # Eliminados por ambigüedad: sanchez, montero, esteban, lopez, diaz, rego, pastor
}

# Conjunto para detección rápida de si un nombre viene de alias de apellido único
_NOMBRES_DE_APELLIDO_UNICO = set(ALIASES_APELLIDO_UNICO.values())

# Alias combinado (solo para retrocompatibilidad interna)
ALIASES_NOMBRES = {**ALIASES_NOMBRE_COMPLETO, **ALIASES_APELLIDO_UNICO}

# Nombres canónicos únicos del diccionario (~661), usados para sugerir candidatos
# acotados a LLM-DICT por similitud textual, en vez de mandar el diccionario entero
# (que serían ~11 500 tokens y reventaría el TPM del free tier de Groq).
_NOMBRES_CANONICOS_LOWER = {n.lower(): n for n in sorted(set(ALIASES_NOMBRES.values()))}


def _construir_nombre_a_partido() -> dict:
    """Extrae el partido (o rol: 'Senador'/'Gobierno') anotado como comentario
    junto a cada entrada de ALIASES_NOMBRE_COMPLETO. Esa información solo existe
    como comentario Python, así que no sobrevive en el dict ya cargado — hay que
    releer el propio fichero fuente. Se ejecuta una sola vez al importar el
    módulo. Si el fichero no se puede leer (p. ej. empaquetado sin fuente),
    devuelve un dict vacío y todo sigue funcionando, solo que sin esta pista."""
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            contenido = f.read()
        inicio = contenido.index("ALIASES_NOMBRE_COMPLETO = {")
        fin = contenido.index("\n}", inicio)
        bloque = contenido[inicio:fin]
        mapa = {}
        for linea in bloque.split("\n"):
            m = re.match(r'\s*"[^"]+"\s*:\s*"([^"]+)"\s*,?\s*#\s*(.*)$', linea)
            if m:
                nombre, partido = m.group(1).strip(), m.group(2).strip()
                if nombre not in mapa:
                    mapa[nombre] = partido
        return mapa
    except Exception:
        return {}


# Nombre canónico -> partido (o 'Senador'/'Gobierno' si no se anotó partido real).
# ~1130 de las 1703 entradas tienen partido real; ~570 (senadores y gobierno)
# solo tienen el rol, no el partido — se muestra igualmente porque orienta al LLM.
NOMBRE_A_PARTIDO = _construir_nombre_a_partido()


def _candidatos_diccionario(nombre_deformado: str, top_n: int = 6) -> list:
    """Devuelve hasta top_n nombres del diccionario más parecidos textualmente al
    nombre deformado, para dárselos a LLM-DICT como candidatos acotados en vez de
    pasarle el diccionario entero (inviable por tokens en el free tier de Groq)."""
    if not nombre_deformado:
        return []
    parecidos = difflib.get_close_matches(
        nombre_deformado.lower(), _NOMBRES_CANONICOS_LOWER.keys(), n=top_n, cutoff=0.5
    )
    return [_NOMBRES_CANONICOS_LOWER[k] for k in parecidos]


# Extracción de partido por palabra clave, independiente de PATRONES_GRUPO/
# grupo_a_partido (que falla con frases como "Grupo Parlamentario Vasco PNV" o
# "Grupo Parlamentario Vasco de la JPNV" porque compara claves con el prefijo
# "grupo parlamentario" contra texto capturado SIN ese prefijo). Se usa para
# sobreescribir en código lo que diga el LLM sobre el partido — más fiable que
# confiar en que el LLM lea bien el texto de la Mesa, que en la práctica hemos
# visto que a veces ignora y "ancla" en el partido más frecuente del contexto.
_PARTIDOS_POR_PALABRA_CLAVE = {
    "psoe": "PSOE", "socialista": "PSOE",
    "popular": "PP",
    "vox": "Vox",
    "republicano": "ERC", "erc": "ERC",
    "junts": "Junts",
    "sumar": "Sumar",
    "bildu": "EH Bildu",
    "vasco": "PNV", "pnv": "PNV",
    "navarro": "UPN", "upn": "UPN",
    "canaria": "CC",
    "podemos": "Podemos",
    "mixto": "Mixto",
}


def _partido_desde_texto_mesa(*textos) -> str:
    """Busca menciones explícitas de un grupo parlamentario conocido en el texto
    de la Mesa (presentación y/o cierre). Devuelve el partido si lo encuentra,
    o None si no hay ninguna mención reconocible."""
    for texto in textos:
        if not texto:
            continue
        texto_l = texto.lower()
        for clave, partido in _PARTIDOS_POR_PALABRA_CLAVE.items():
            if clave in texto_l:
                return partido
    return None

# FIX B: palabras de cargo que no deben tratarse como nombres
PALABRAS_CARGO = {
    "ministro", "ministra",
    "secretario", "secretaria",
    "subsecretario", "subsecretaria",
    "director", "directora",
    "presidente", "presidenta",
    "vicepresidente", "vicepresidenta",
    "comisario", "comisaria",
    "delegado", "delegada",
    "consejero", "consejera",
    "portavoz",
}

# FIX C: palabras de cortesía que el cierre retroactivo no interpreta como nombre
PALABRAS_CORTESIA_CIERRE = {
    "presidenta", "presidente", "señoría", "señorías",
    "diputada", "diputado", "ministra", "ministro",
    "secretaria", "secretario", "señora", "señor",
}

FORMULAS_INICIO = [
    r"muchas gracias",
    r"muchísimas gracias",
    r"señorías",
    r"buenos días",
    r"buenas tardes",
    r"buenas noches",
    r"señor presidente",
    r"señora presidenta",
    r"con la venia",
]

# =============================================================
# UTILIDADES
# =============================================================

def _quitar_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _nombre_es_apellido_unico(nombre: str) -> bool:
    """Devuelve True si el nombre normalizado proviene de ALIASES_APELLIDO_UNICO."""
    return nombre in _NOMBRES_DE_APELLIDO_UNICO


def corregir_nombre_whisper(nombre_raw: str) -> str | None:
    """
    Intenta corregir deformaciones de Whisper buscando en ALIASES_NOMBRE_COMPLETO
    con coincidencias de subcadena y por primer/último token.
    Solo usa aliases multi-token para evitar falsos positivos.
    """
    if not nombre_raw:
        return None
    clave = _quitar_tildes(nombre_raw.strip().lower())
    if clave in ALIASES_NOMBRE_COMPLETO:
        return ALIASES_NOMBRE_COMPLETO[clave]
    for alias, canonico in ALIASES_NOMBRE_COMPLETO.items():
        if len(alias.split()) > 1 and (alias in clave or clave in alias):
            return canonico
    tokens = clave.split()
    if tokens:
        for token in [tokens[0], tokens[-1]]:
            if token in ALIASES_NOMBRE_COMPLETO:
                return ALIASES_NOMBRE_COMPLETO[token]
    return None


def normalizar_nombre(
    nombre_raw: str,
    solo_completos: bool = False,
) -> tuple[str | None, bool]:
    """
    Normaliza un nombre capturado por heurística.
    Devuelve (nombre_normalizado, nombre_desde_apellido_unico).

    - nombre_desde_apellido_unico=True indica que el match fue por apellido
      único (último recurso), señal de posible ambigüedad → activar LLM-2.
    - solo_completos=True: ignora ALIASES_APELLIDO_UNICO completamente.
    - FIX B: devuelve (None, False) si el primer token es PALABRA_CARGO.

    ORDEN DE BÚSQUEDA (v2.4 — fix de raíz):
      1. Nombre completo exacto en ALIASES_NOMBRE_COMPLETO
      2. Subcadena multi-token (≥2 palabras) en ALIASES_NOMBRE_COMPLETO
      3. Corrección fuzzy Whisper (multi-token)
      4. Apellido único (solo si no solo_completos) → flag ambigüedad
      5. Fallback: devolver el nombre tal cual (title-case)
    """
    if not nombre_raw:
        return None, False

    tokens_raw = nombre_raw.strip().split()
    if not tokens_raw:
        return None, False

    # FIX B: descartar si empieza por cargo
    primer_token = _quitar_tildes(tokens_raw[0].lower())
    if primer_token in PALABRAS_CARGO:
        return None, False

    clave = _quitar_tildes(nombre_raw.strip().lower())

    # ── 1. Nombre completo exacto ────────────────────────────────────────
    if clave in ALIASES_NOMBRE_COMPLETO:
        return ALIASES_NOMBRE_COMPLETO[clave], False

    # ── 2. Subcadena multi-token en ALIASES_NOMBRE_COMPLETO ─────────────
    # Solo si la clave tiene más de un token (evita falsos positivos)
    if len(tokens_raw) > 1:
        for alias, canonico in ALIASES_NOMBRE_COMPLETO.items():
            if len(alias.split()) > 1 and (alias in clave or clave in alias):
                return canonico, False

    # ── 3. Corrección fuzzy Whisper (multi-token) ────────────────────────
    corregido = corregir_nombre_whisper(nombre_raw)
    if corregido:
        return corregido, False

    # ── 4. Apellido único — ÚLTIMO RECURSO, señal de ambigüedad ─────────
    if not solo_completos:
        ultimo_token = clave.split()[-1]
        if ultimo_token in ALIASES_APELLIDO_UNICO:
            return ALIASES_APELLIDO_UNICO[ultimo_token], True  # ← ambiguo

    # ── 5. Fallback: devolver tal cual ───────────────────────────────────
    return nombre_raw.strip().title(), False


def grupo_a_partido(texto_grupo: str) -> str | None:
    if not texto_grupo:
        return None
    clave = _quitar_tildes(texto_grupo.strip().lower())
    for nombre, sigla in GRUPOS_PARLAMENTARIOS.items():
        if nombre in clave or clave in nombre:
            return sigla
    return None


# =============================================================
# CARGA Y GUARDADO
# =============================================================

def cargar_chunks(ruta: Path) -> list:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_json(chunks: list, ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


# =============================================================
# CAPA 1: HEURÍSTICA
# =============================================================

PATRONES_PRESENTACION = [
    # Presentaciones directas con nombre
    (r"tiene.*?la palabra (?:el|la) se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    (r"tiene.*?la palabra.*?(?:el|la) se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    (r"interviene (?:el|la) se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    (r"para (?:su defensa|la defensa de la iniciativa).*?(?:el|la) se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    (r"(?:el|la) portavoz del grupo.*?(?:el|la) se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    (r"por el grupo parlamentario.*?(?:el|la) se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    (r"tiene.*?la palabra (?:el candidato|la candidata).*?(?:se[ñn]or[a]?\s+)?([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.95),
    # CAMBIO 1: cargos institucionales SOLO dentro de presentación explícita
    (r"tiene.*?la palabra el presidente del gobierno", 0.95),   # → CARGOS_FIJOS
    (r"tiene.*?la palabra la presidenta del congreso", 0.95),
    (r"tiene.*?la palabra el presidente del senado", 0.95),
]

# Regex auxiliar para extraer el cargo de los patrones de presentación con cargo
_REGEX_CARGO_EN_PRESENTACION = re.compile(
    r"tiene la palabra (?:el|la) (presidente del gobierno|presidenta del congreso|presidente del senado)",
    re.IGNORECASE,
)

PATRONES_CIERRE = [
    (r"(?:muchas|muchísimas) gracias,?\s+se[ñn]or[a]?\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})", 0.92),
]

PATRONES_NOMBRE = PATRONES_PRESENTACION  # alias retrocompatibilidad

PATRONES_GRUPO = [
    r"grupo parlamentario ([\w\s,]+?)(?:,|\.|tiene|$)",
    r"en nombre del grupo ([\w\s]+)",
    r"por el grupo ([\w\s]+)",
]

# CAMBIO 1: PATRONES_CARGO_INST eliminado. Los cargos institucionales ya no
# asignan nombre al speaker por el mero hecho de ser mencionados en el texto.
# Solo se resuelven si aparecen dentro de PATRONES_PRESENTACION arriba.

PATRONES_PRESIDENCIA_MESA = [
    r"si sus? se[ñn]or[ií]as?\s+(?:quieren\s+)?ocupan?\s+los\s+esca[ñn]os",
    r"vamos a (?:re)?anudar\s+la\s+sesi[oó]n",
    r"se (?:abre|levanta|reanuda)\s+la\s+sesi[oó]n",
    r"comenzamos\s+con\s+el\s+punto",
    r"pasamos\s+(?:ahora\s+)?a\s+votar",
    r"votamos?\s+(?:ahora\s+)?el\s+punto",
    r"votos?\s+emitidos",
    r"en\s+cumplimiento\s+de\s+esta\s+disposici[oó]n",
    r"se\s+(?:aprueba|rechaza|no\s+se\s+aprueba)\s+(?:el\s+punto|la\s+proposici[oó]n|la\s+moci[oó]n)",
    r"(?:abrimos|cerramos)\s+el\s+plazo\s+de\s+votaci[oó]n",
    r"por\s+favor,?\s+si\s+guardan\s+silencio",
    r"un\s+(?:momento\s+de\s+)?silencio,?\s+por\s+favor",
]


def detectar_patrones_heuristicos(chunks: list, idx: int) -> dict:
    textos = []
    if idx > 0:
        textos.append(chunks[idx - 1]["texto"])
    textos.append(chunks[idx]["texto"])
    # NOTA: No incluimos chunks[idx + 1] porque si el siguiente chunk es la Mesa 
    # dando la palabra a un TERCER ponente, este ponente actual heredaría 
    # erróneamente ese nombre de presentación.
    ventana = " ".join(textos)

    # FIX 1: presidencia de mesa (solo texto actual)
    texto_actual_solo = chunks[idx]["texto"]
    for patron in PATRONES_PRESIDENCIA_MESA:
        if re.search(patron, texto_actual_solo, re.IGNORECASE):
            return {
                "nombre": CARGOS_FIJOS.get("presidenta del congreso"),
                "partido": "Mesa",
                "confianza": 0.96,
                "fragmento_detectado": f"[presidencia_mesa] {patron}",
                "es_presentacion": False,
                "nombre_es_apellido_unico": False,
            }

    # CAMBIO 1: cargo institucional SOLO si aparece dentro de presentación
    m_cargo_pres = _REGEX_CARGO_EN_PRESENTACION.search(ventana)
    if m_cargo_pres:
        cargo_key = _quitar_tildes(m_cargo_pres.group(1).lower())
        nombre_cargo = CARGOS_FIJOS.get(cargo_key)
        return {
            "nombre": nombre_cargo,
            "nombre_raw": nombre_cargo,
            "partido": None,
            "confianza": 0.95,
            "fragmento_detectado": m_cargo_pres.group(0)[:120],
            "es_presentacion": True,
            "nombre_es_apellido_unico": False,
        }

    # Patrones de presentación con nombre
    for patron, confianza in PATRONES_PRESENTACION:
        if "presidente del" in patron or "presidenta del" in patron:
            continue
        m = re.search(patron, ventana, re.IGNORECASE)
        if m:
            nombre_raw_capturado = m.group(1).strip()
            nombre_encontrado, es_apellido_unico = normalizar_nombre(
                nombre_raw_capturado, solo_completos=False
            )
            if nombre_encontrado is None:
                continue  # FIX B: cargo detectado
            partido_encontrado = None
            for p_grupo in PATRONES_GRUPO:
                mg = re.search(p_grupo, ventana, re.IGNORECASE)
                if mg:
                    partido_encontrado = grupo_a_partido(mg.group(1))
                    if partido_encontrado:
                        break
            return {
                "nombre": nombre_encontrado,
                "nombre_raw": nombre_raw_capturado,   # sin normalizar
                "partido": partido_encontrado,
                "confianza": confianza,
                "fragmento_detectado": m.group(0)[:120],
                "es_presentacion": True,
                "nombre_es_apellido_unico": es_apellido_unico,
            }

    # Autopresentación
    texto_actual = chunks[idx]["texto"].strip()
    m_auto = re.match(
        r"(?:me llamo|mi nombre es|soy|represento a)\s+([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+){0,3})",
        texto_actual, re.IGNORECASE,
    )
    if m_auto:
        nombre_raw_capturado = m_auto.group(1)
        nombre, _ = normalizar_nombre(nombre_raw_capturado, solo_completos=True)
        return {
            "nombre": nombre,
            "nombre_raw": nombre_raw_capturado,
            "partido": None,
            "confianza": 0.50,
            "fragmento_detectado": m_auto.group(0)[:120],
            "es_presentacion": False,
            "nombre_es_apellido_unico": False,
        }

    # Solo partido
    for patron in PATRONES_GRUPO:
        m = re.search(patron, ventana, re.IGNORECASE)
        if m:
            partido = grupo_a_partido(m.group(1))
            if partido:
                return {
                    "nombre": None,
                    "nombre_raw": None,
                    "partido": partido,
                    "confianza": 0.70,
                    "fragmento_detectado": m.group(0)[:120],
                    "es_presentacion": False,
                    "nombre_es_apellido_unico": False,
                }

    return {
        "nombre": None,
        "nombre_raw": None,
        "partido": None,
        "confianza": 0.0,
        "fragmento_detectado": None,
        "es_presentacion": False,
        "nombre_es_apellido_unico": False,
    }


# =============================================================
# FIX 3 / FIX C: Cierre retroactivo mejorado
# =============================================================

def intentar_cierre_retroactivo(chunks: list, idx: int, fingerprints: dict) -> bool:
    if idx == 0:
        return False
    texto_actual = chunks[idx]["texto"]
    for patron, confianza in PATRONES_CIERRE:
        m = re.search(patron, texto_actual, re.IGNORECASE)
        if m:
            nombre_raw = m.group(1).strip()
            # FIX C: ignorar palabras de cortesía / cargo
            if _quitar_tildes(nombre_raw.lower()) in PALABRAS_CORTESIA_CIERRE:
                continue
            # FIX B + nueva firma de normalizar_nombre
            nombre, _ = normalizar_nombre(nombre_raw, solo_completos=False)
            if nombre is None:
                continue
            chunk_anterior = chunks[idx - 1]
            if chunk_anterior.get("estado_id") in ("DESCONOCIDO", "AMBIGUO"):
                ventana_ant = chunk_anterior["texto"]
                partido = None
                for p_grupo in PATRONES_GRUPO:
                    mg = re.search(p_grupo, ventana_ant, re.IGNORECASE)
                    if mg:
                        partido = grupo_a_partido(mg.group(1))
                        if partido:
                            break
                chunk_anterior.update({
                    "nombre": nombre,
                    "partido": partido,
                    "estado_id": "IDENTIFICADO",
                    "confianza_id": confianza,
                    "metodo_id": "heuristica_cierre",
                })
                actualizar_fingerprint(
                    fingerprints,
                    chunk_anterior["ponente"],
                    {"nombre": nombre, "partido": partido, "confianza": confianza},
                )
                return True
    return False


# =============================================================
# CAPA 2: CAMBIO DE TURNO
# =============================================================

def detectar_cambio_turno(chunks: list, idx: int) -> bool:
    textos = []
    if idx > 0:
        textos.append(chunks[idx - 1]["texto"])
    textos.append(chunks[idx]["texto"])
    ventana = " ".join(textos)

    for patron, _ in PATRONES_NOMBRE:
        if re.search(patron, ventana, re.IGNORECASE):
            return True

    texto_actual = chunks[idx]["texto"].strip()
    for formula in FORMULAS_INICIO:
        if re.match(formula, texto_actual, re.IGNORECASE):
            return True

    return False


# =============================================================
# CAPA 3: LLM (GEMINI)
# =============================================================

def _resumen_fingerprints(fingerprints: dict) -> str:
    """Genera resumen legible de fingerprints para el prompt del LLM (LLM-1)."""
    lineas = []
    for sp, fp in sorted(fingerprints.items()):
        if not fp.get("ambiguo") and fp.get("nombre"):
            partido = fp.get("partido") or "desconocido"
            lineas.append(f"  - {sp}: {fp['nombre']} ({partido})")
    return "\n".join(lineas) if lineas else "  (ninguno identificado aún)"




# =============================================================
# FINGERPRINTING
# =============================================================

def aplicar_fingerprint(chunk: dict, fingerprints: dict) -> dict | None:
    ponente = chunk.get("ponente")
    if not ponente or ponente not in fingerprints:
        return None
    fp = fingerprints[ponente]
    if fp.get("ambiguo"):
        return None
    return {"nombre": fp["nombre"], "partido": fp["partido"], "confianza": fp["confianza"]}


def actualizar_fingerprint(fingerprints: dict, ponente: str, resultado: dict):
    if resultado.get("confianza", 0) < 0.90 or not resultado.get("nombre"):
        return
    nombre = resultado["nombre"]
    if ponente not in fingerprints:
        fingerprints[ponente] = {
            "nombre": nombre,
            "partido": resultado.get("partido"),
            "confianza": resultado["confianza"],
            "ambiguo": False,
        }
    else:
        fp = fingerprints[ponente]
        if not fp["ambiguo"] and fp["nombre"] != nombre:
            fp["ambiguo"] = True


# =============================================================
# FIX 4: Fusión de fingerprints por nombre canónico
# =============================================================

def fusionar_fingerprints_por_nombre(chunks: list, fingerprints: dict) -> int:
    nombre_a_speakers = defaultdict(list)
    for sp, fp in fingerprints.items():
        if not fp.get("ambiguo") and fp.get("nombre"):
            nombre_a_speakers[fp["nombre"]].append(sp)

    actualizados = 0
    for nombre, speakers in nombre_a_speakers.items():
        partido_canonico = next(
            (fingerprints[sp].get("partido") for sp in speakers
             if fingerprints[sp].get("partido")),
            None,
        )
        confianza_canonica = max(fingerprints[sp]["confianza"] for sp in speakers)
        
        for sp in speakers:
            for chunk in chunks:
                if chunk["ponente"] == sp and chunk["estado_id"] in ("DESCONOCIDO", "AMBIGUO"):
                    metodo = "fingerprint_fusion" if len(speakers) > 1 else "fingerprint_retroactivo"
                    chunk.update({
                        "nombre": nombre,
                        "partido": partido_canonico,
                        "estado_id": "IDENTIFICADO",
                        "confianza_id": confianza_canonica,
                        "metodo_id": metodo,
                    })
                    actualizados += 1
    return actualizados


# =============================================================
# FUNCIÓN PRINCIPAL
# =============================================================

# CAMBIO 2: umbral superior de LLM-2 ampliado a 0.97 para cubrir
# heurísticas de confianza 0.95 con apellido único ambiguo
UMBRAL_LLM2_MAX = 0.97



# =============================================================
# NUEVAS FUNCIONES LLM (reemplazan LLM-1 y LLM-2)
# =============================================================

def _buscar_contexto_mesa(chunks: list, idx: int) -> str:
    """Chunks de Mesa consecutivos anteriores al chunk dado."""
    ponente = chunks[idx]["ponente"]
    textos = []
    for j in range(idx - 1, max(idx - 15, -1), -1):
        c = chunks[j]
        es_mesa = c.get("partido") == "Mesa" or c.get("nombre") == "Francina Armengol"
        if c["ponente"] == ponente:
            continue
        if not es_mesa:
            break
        textos.insert(0, c["texto"])
    return " ".join(textos).strip()


def _buscar_contexto_mesa_posterior(chunks: list, idx: int) -> str:
    """Chunks de Mesa consecutivos posteriores al chunk dado."""
    ponente = chunks[idx]["ponente"]
    textos = []
    for j in range(idx + 1, min(idx + 10, len(chunks))):
        c = chunks[j]
        es_mesa = c.get("partido") == "Mesa" or c.get("nombre") == "Francina Armengol"
        if c["ponente"] == ponente:
            continue
        if not es_mesa:
            break
        textos.append(c["texto"])
    return " ".join(textos).strip()


def _nombre_fuera_diccionario(nombre_raw: str) -> bool:
    """True si el nombre no tiene match en el diccionario."""
    if not nombre_raw:
        return False
    clave = _quitar_tildes(nombre_raw.strip().lower())
    tokens = clave.split()
    if clave in ALIASES_NOMBRE_COMPLETO:
        return False
    if len(tokens) > 1:
        for alias in ALIASES_NOMBRE_COMPLETO:
            if len(alias.split()) > 1 and (alias in clave or clave in alias):
                return False
    if corregir_nombre_whisper(nombre_raw):
        return False
    if tokens and tokens[-1] in ALIASES_APELLIDO_UNICO:
        return False
    return True


def _llamar_groq(api_key: str, prompt: str, modelo: str = "llama-3.1-8b-instant") -> str:
    """Wrapper para llamar a la API de Groq (OpenAI-compatible).

    Modelo por defecto: llama-3.1-8b-instant — es el modelo del free tier de Groq
    con más llamadas diarias permitidas (14 400 RPD / 30 RPM / 6 000 TPM a fecha
    de escritura), muy por encima de llama-3.3-70b-versatile (1 000 RPD) u otros
    modelos Llama del catálogo gratuito. Requiere `pip install groq`.
    """
    from groq import Groq as _Groq
    cliente = _Groq(api_key=api_key)
    respuesta = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    texto = respuesta.choices[0].message.content.strip() if respuesta.choices[0].message.content else ""
    if texto.startswith("```"):
        texto = texto.strip("```json").strip("```").strip()
    return texto


def identificar_desconocidos_con_llm(chunks, fingerprints, api_key, umbral=0.75):
    """LLM-DESC: identifica chunks DESCONOCIDO usando contexto de Mesa."""
    bloques = {}
    for i, c in enumerate(chunks):
        if c["estado_id"] in ("DESCONOCIDO", "AMBIGUO") and not c.get("nombre"):
            sp = c["ponente"]
            if sp not in bloques:
                bloques[sp] = []
            bloques[sp].append(i)

    if bloques:
        print(f"\n[LLM-DESC] {len(bloques)} bloque(s) DESCONOCIDO/AMBIGUO por resolver...")

    n = 0
    for ponente, indices in bloques.items():
        primer_idx = indices[0]
        texto_mesa = _buscar_contexto_mesa(chunks, primer_idx)
        # Se trunca CADA chunk por separado (no la concatenación entera) para que
        # un primer chunk largo no se coma el espacio de los siguientes — así no
        # se pierden frases de autoidentificación de partido que aparecen más
        # adelante en el bloque (caso real: "somos nosotros, Vox..." en el 2º chunk).
        # Se incluyen los 3 primeros chunks + el último del bloque (no solo los
        # 4 primeros): la autoidentificación de partido a menudo aparece en el
        # cierre del discurso, no al principio (casos reales: "el Partido
        # Socialista Obrero Español..." y "mi portavoz, Miriam Nogueiras..." en
        # el último chunk de bloques de 5-6 chunks, que antes se perdían siempre).
        indices_muestra = indices[:3] + [indices[-1]] if len(indices) > 3 else indices[:4]
        texto_speaker = " ".join(chunks[i]["texto"][:400] for i in indices_muestra)
        partido_mesa = _partido_desde_texto_mesa(texto_mesa)
        if not texto_mesa and not texto_speaker:
            print(f"[LLM-DESC] {ponente}: sin contexto de Mesa ni texto propio, se omite")
            continue

        print(f"[LLM-DESC] → Llamando a Groq: {ponente} ({len(indices)} chunk(s), "
              f"contexto Mesa={'sí' if texto_mesa else 'no'})")

        prompt = (
            "Eres un experto en política española y en el Congreso de los Diputados.\n\n"
            "TAREA: Identificar quién es el orador.\n\n"
            "EJEMPLOS:\n\n"
            "Ejemplo 1 — apertura estándar con grupo explícito:\n"
            'PRESENTACIÓN DE LA MESA: "Para defender la enmienda, tiene la palabra, '
            'por el Grupo Parlamentario Republicano, el señor Rufián."\n'
            'TEXTO DEL ORADOR: "Gracias, presidenta. Señorías, venimos hoy a hablar de..."\n'
            '→ {"nombre": "Rufián", "partido": "ERC", "confianza": 0.95, '
            '"razon": "Mesa nombra directamente al orador y su grupo"}\n\n'
            "Ejemplo 2 — cierre y apertura combinados en el mismo chunk:\n"
            'PRESENTACIÓN DE LA MESA: "Muchas gracias, señor Echániz. Tiene ahora la palabra, '
            'por el Grupo Parlamentario Vasco (EAJ-PNV), el señor Esteban."\n'
            'TEXTO DEL ORADOR: "Mila esker, presidenta. Como decíamos..."\n'
            '→ {"nombre": "Esteban", "partido": "PNV", "confianza": 0.95, '
            '"razon": "Cierre del anterior y apertura del nuevo orador en el mismo bloque"}\n\n'
            "Ejemplo 3 — la Mesa no menciona el partido, se deduce del propio discurso:\n"
            'PRESENTACIÓN DE LA MESA: "Tiene la palabra el señor Bel."\n'
            'TEXTO DEL ORADOR: "Gracias, presidenta. Desde Junts per Catalunya consideramos que..."\n'
            '→ {"nombre": "Bel", "partido": "Junts", "confianza": 0.8, '
            '"razon": "Partido deducido del propio texto del orador, no de la Mesa"}\n\n'
            "Ejemplo 4 — información insuficiente, NO inventar:\n"
            'PRESENTACIÓN DE LA MESA: "(no disponible)"\n'
            'TEXTO DEL ORADOR: "Un momento, por favor. Un segundo."\n'
            '→ {"nombre": null, "partido": null, "confianza": 0.0, '
            '"razon": "Sin presentación de Mesa ni contenido suficiente en el texto"}\n\n'
            "AHORA RESUELVE ESTE CASO:\n\n"
            f"PRESENTACIÓN DE LA MESA:\n{texto_mesa or '(no disponible)'}\n\n"
            f"TEXTO DEL ORADOR:\n{texto_speaker}\n\n"
            f"ORADORES YA IDENTIFICADOS:\n{_resumen_fingerprints(fingerprints)}\n\n"
            "INSTRUCCIONES: Lee la presentación de la Mesa — casi siempre tiene el nombre. "
            "Si no, deduce el partido del texto del orador. Si no hay información suficiente, "
            "sigue el patrón del Ejemplo 4: no inventes.\n\n"
            'Responde SOLO con JSON: {"nombre": "nombre o null", "partido": "partido o null", "confianza": 0.0, "razon": "max 100 chars"}'
        )
        try:
            texto_resp = _llamar_groq(api_key, prompt)
            if not texto_resp:
                print(f"[LLM-DESC]   ✗ {ponente}: respuesta vacía de Groq")
                continue
            datos = json.loads(texto_resp)
            nombre_llm = datos.get("nombre")
            partido_llm = partido_mesa or datos.get("partido")
            confianza = float(datos.get("confianza", 0.0))
            razon = datos.get("razon", "")
            if partido_mesa and partido_mesa != datos.get("partido"):
                razon += f" [partido corregido a {partido_mesa} por mención explícita de la Mesa]"
            logging.info(f"[LLM-DESC] {ponente} → {nombre_llm} ({partido_llm}) conf={confianza:.2f} | {razon}")
            if confianza < umbral or (not nombre_llm and not partido_llm):
                print(f"[LLM-DESC]   ✗ {ponente}: sin cambio (conf={confianza:.2f} < {umbral}) | {razon}")
                continue
            nombre_final = None
            if nombre_llm:
                nombre_norm, _ = normalizar_nombre(nombre_llm.strip(), solo_completos=False)
                nombre_final = nombre_norm or nombre_llm.strip()
            n_bloque = 0
            for i in indices:
                c = chunks[i]
                if not c.get("nombre"):
                    c["nombre"] = nombre_final
                    c["partido"] = partido_llm
                    c["estado_id"] = "IDENTIFICADO" if confianza >= 0.85 else "AMBIGUO"
                    c["confianza_id"] = confianza
                    c["metodo_id"] = "llm_desconocidos"
                    n += 1
                    n_bloque += 1
            print(f"[LLM-DESC]   ✓ {ponente} → {nombre_final} | {partido_llm} "
                  f"(conf={confianza:.2f}, {n_bloque} chunk(s) actualizados)")
            if nombre_final and confianza >= 0.85:
                actualizar_fingerprint(fingerprints, ponente, {"nombre": nombre_final, "partido": partido_llm, "confianza": confianza})
        except Exception as e:
            print(f"[LLM-DESC]   ⚠ {ponente}: error llamando a Groq — {e}")
            logging.error(f"[LLM-DESC] Error en {ponente}: {e}")
    return n


def verificar_nombres_fuera_diccionario_con_llm(chunks, fingerprints, api_key, umbral=0.78):
    """LLM-DICT: corrige nombres deformados por Whisper no presentes en el diccionario."""
    bloques = {}
    for i, c in enumerate(chunks):
        if (c["estado_id"] == "IDENTIFICADO"
                and c.get("nombre")
                and c.get("metodo_id") in ("heuristica", "fingerprint", "fingerprint_fusion")
                and _nombre_fuera_diccionario(c["nombre"])):
            sp = c["ponente"]
            if sp not in bloques:
                bloques[sp] = []
            bloques[sp].append(i)

    if not bloques:
        return 0

    print(f"\n[LLM-DICT] {len(bloques)} nombre(s) fuera de diccionario por verificar...")

    n = 0
    for ponente, indices in bloques.items():
        primer_idx = indices[0]
        ultimo_idx = indices[-1]
        nombre_actual = chunks[primer_idx].get("nombre", "")
        texto_mesa_prev = _buscar_contexto_mesa(chunks, primer_idx)
        texto_mesa_post = _buscar_contexto_mesa_posterior(chunks, ultimo_idx)
        # Se incluyen los 3 primeros chunks + el último del bloque (no solo los
        # 4 primeros): la autoidentificación de partido a menudo aparece en el
        # cierre del discurso, no al principio (caso real detectado: "el
        # Partido Socialista Obrero Español..." en el último chunk de un bloque
        # de 6, que antes se perdía siempre).
        indices_muestra = indices[:3] + [indices[-1]] if len(indices) > 3 else indices[:4]
        texto_speaker = " ".join(chunks[i]["texto"][:300] for i in indices_muestra)
        partido_mesa = _partido_desde_texto_mesa(texto_mesa_prev, texto_mesa_post)

        candidatos = _candidatos_diccionario(nombre_actual)
        texto_candidatos = (
            "\n".join(f"- {c} ({NOMBRE_A_PARTIDO.get(c, '?')})" for c in candidatos) if candidatos
            else "(ninguno con similitud razonable)"
        )

        print(f"[LLM-DICT] → Llamando a Groq: {ponente} (nombre actual: '{nombre_actual}', "
              f"{len(candidatos)} candidato(s) del diccionario)")

        prompt = (
            "Eres un experto en política española y en el Congreso de los Diputados.\n\n"
            "TAREA: El sistema de reconocimiento de voz capturó un nombre posiblemente deformado. "
            "Identifica el nombre real del orador. Puedes usar tu propio conocimiento, pero si "
            "alguno de los CANDIDATOS DEL DICCIONARIO encaja con el contexto, prefiérelo — son "
            "nombres reales ya verificados de diputados de esta legislatura, con su partido entre "
            "paréntesis (dato verificado). Si el texto de la Mesa da un partido más específico que "
            "el del candidato (p. ej. el candidato dice 'Mixto' pero la Mesa dice 'UPN'), usa el "
            "de la Mesa, que es más preciso.\n\n"
            "EJEMPLOS:\n\n"
            "Ejemplo 1 — deformación de dos apellidos, confirmada por el cierre:\n"
            "NOMBRE CAPTURADO: 'Bellugera Balañá'\n"
            "CANDIDATOS DEL DICCIONARIO: Vallugera Balañà (ERC), Villalonga Ferrer (PSOE), Ballugera Gómez (PP)\n"
            'CIERRE DE LA MESA (después): "Muchas gracias, señora Vallugera."\n'
            '→ {"nombre": "Pilar Vallugera Balañà", "partido": "ERC", "confianza": 0.95, '
            '"razon": "Cierre confirma apellido, coincide con candidato del diccionario"}\n\n'
            "Ejemplo 2 — deformación fonética, sin cierre pero con candidato claro:\n"
            "NOMBRE CAPTURADO: 'Baquiero Montero'\n"
            "CANDIDATOS DEL DICCIONARIO: Vaquero Montero (PNV), Ballestero Montoro (PP)\n"
            'TEXTO DEL ORADOR: "...como diputada del PNV, quiero trasladar..."\n'
            '→ {"nombre": "Maribel Vaquero Montero", "partido": "PNV", "confianza": 0.9, '
            '"razon": "Coincide fonéticamente con candidato del diccionario y el partido cuadra"}\n\n'
            "Ejemplo 3 — nombre muy deformado, candidato correcto entre varios, y la Mesa da un partido más específico que el del diccionario:\n"
            "NOMBRE CAPTURADO: 'Catalanes Hogueras'\n"
            "CANDIDATOS DEL DICCIONARIO: Alberto Catalán Higueras (Mixto), Carlos Flores Juberías (Vox), Miriam Nogueras (Junts)\n"
            'PRESENTACIÓN DE LA MESA (antes): "Tiene la palabra, por el Grupo Parlamentario Mixto, '
            'el señor Catalán, de Unión del Pueblo Navarro."\n'
            '→ {"nombre": "Alberto Catalán Higueras", "partido": "UPN", "confianza": 0.9, '
            '"razon": "Mesa confirma apellido y da partido más específico (UPN) que el genérico Mixto del diccionario"}\n\n'
            "Ejemplo 4 — ni el diccionario ni el conocimiento propio dan una respuesta segura:\n"
            "NOMBRE CAPTURADO: 'Xilonga Prat'\n"
            "CANDIDATOS DEL DICCIONARIO: (ninguno con similitud razonable)\n"
            'TEXTO DEL ORADOR: "Gracias."\n'
            '→ {"nombre": null, "partido": null, "confianza": 0.0, '
            '"razon": "Sin candidato plausible ni contexto suficiente para confirmar"}\n\n'
            "AHORA RESUELVE ESTE CASO:\n\n"
            f"NOMBRE CAPTURADO (posiblemente deformado): '{nombre_actual}'\n\n"
            f"CANDIDATOS DEL DICCIONARIO:\n{texto_candidatos}\n\n"
            f"PRESENTACIÓN DE LA MESA (antes):\n{texto_mesa_prev or '(no disponible)'}\n\n"
            f"CIERRE DE LA MESA (después):\n{texto_mesa_post or '(no disponible)'}\n\n"
            f"TEXTO DEL ORADOR:\n{texto_speaker}\n\n"
            f"ORADORES YA IDENTIFICADOS:\n{_resumen_fingerprints(fingerprints)}\n\n"
            "INSTRUCCIONES: Si el cierre dice 'muchas gracias, señora X', X es el nombre real. "
            "Usa presentación y texto para confirmar o precisar el partido del candidato. Si no hay "
            "ninguna pista fiable, sigue el patrón del Ejemplo 4: no inventes.\n\n"
            'Responde SOLO con JSON: {"nombre": "nombre correcto o null", "partido": "partido o null", "confianza": 0.0, "razon": "max 100 chars"}'
        )
        try:
            texto_resp = _llamar_groq(api_key, prompt)
            if not texto_resp:
                print(f"[LLM-DICT]   ✗ {ponente}: respuesta vacía de Groq")
                continue
            datos = json.loads(texto_resp)
            nombre_llm = datos.get("nombre")
            partido_llm = partido_mesa or datos.get("partido")
            confianza = float(datos.get("confianza", 0.0))
            razon = datos.get("razon", "")
            if partido_mesa and partido_mesa != datos.get("partido"):
                razon += f" [partido corregido a {partido_mesa} por mención explícita de la Mesa]"
            logging.info(f"[LLM-DICT] {ponente} '{nombre_actual}' → {nombre_llm} ({partido_llm}) conf={confianza:.2f} | {razon}")
            if confianza < umbral or not nombre_llm:
                print(f"[LLM-DICT]   ✗ {ponente}: sin cambio (conf={confianza:.2f} < {umbral}) | {razon}")
                continue
            nombre_norm, _ = normalizar_nombre(nombre_llm.strip(), solo_completos=False)
            nombre_final = nombre_norm or nombre_llm.strip()
            # Si la Mesa no dio partido explícito, usar el partido verificado del
            # diccionario para el nombre ya resuelto — más fiable que la
            # respuesta del LLM. "Senador"/"Gobierno" no son partidos reales
            # (son entradas de senadores/gobierno sin partido anotado), así que
            # se descartan como fuente de partido.
            if not partido_mesa:
                partido_dict = NOMBRE_A_PARTIDO.get(nombre_final)
                if partido_dict not in (None, "Senador", "Gobierno") and partido_dict != partido_llm:
                    razon += f" [partido corregido a {partido_dict} según diccionario verificado]"
                    partido_llm = partido_dict
            n_bloque = 0
            for i in indices:
                c = chunks[i]
                if c.get("nombre") == nombre_actual:
                    c["nombre"] = nombre_final
                    if partido_llm:
                        c["partido"] = partido_llm
                    c["confianza_id"] = confianza
                    c["metodo_id"] = "llm_correccion_dict"
                    n += 1
                    n_bloque += 1
            print(f"[LLM-DICT]   ✓ {ponente}: '{nombre_actual}' → '{nombre_final}' | {partido_llm} "
                  f"(conf={confianza:.2f}, {n_bloque} chunk(s) corregidos)")
            if confianza >= 0.85:
                actualizar_fingerprint(fingerprints, ponente, {"nombre": nombre_final, "partido": partido_llm or chunks[primer_idx].get("partido"), "confianza": confianza})
        except Exception as e:
            print(f"[LLM-DICT]   ⚠ {ponente}: error llamando a Groq — {e}")
            logging.error(f"[LLM-DICT] Error en {ponente}: {e}")
    return n


def identificar_video(
    ruta_json_entrada: Path,
    ruta_json_salida: Path = None,
    usar_llm: bool = True,
    umbral_llm: float = 0.90,
) -> Path:
    ruta_json_entrada = Path(ruta_json_entrada)
    if ruta_json_salida is None:
        ruta_json_salida = ruta_json_entrada.parent / (ruta_json_entrada.stem + "_identificado.json")
    ruta_json_salida = Path(ruta_json_salida)

    ruta_log = ruta_json_salida.parent / f"identificacion_{ruta_json_entrada.stem}.log"
    logging.basicConfig(
        filename=ruta_log, filemode="w", level=logging.INFO,
        format="%(asctime)s %(message)s", encoding="utf-8",
    )
    logging.info(f"Iniciando identificación v2.14: {ruta_json_entrada}")

    chunks = cargar_chunks(ruta_json_entrada)
    fingerprints: dict = {}

    n_heuristica = n_fingerprint = n_ambiguo = n_desconocido = 0
    n_cierre_retroactivo = 0

    for idx, chunk in enumerate(chunks):
        ponente = chunk.get("ponente", "DESCONOCIDO")
        chunk["nombre"] = None
        chunk["partido"] = None
        chunk["estado_id"] = "DESCONOCIDO"
        chunk["confianza_id"] = 0.0
        chunk["metodo_id"] = None

        # FIX 3/C: cierre retroactivo
        if intentar_cierre_retroactivo(chunks, idx, fingerprints):
            n_cierre_retroactivo += 1
            logging.info(f"[{idx}] CIERRE RETROACTIVO sobre chunk {idx-1}")

        # ── 1. Fingerprint ──────────────────────────────────────────────
        fp = aplicar_fingerprint(chunk, fingerprints)
        if fp:
            chunk.update({"nombre": fp["nombre"], "partido": fp["partido"],
                          "estado_id": "IDENTIFICADO", "confianza_id": fp["confianza"],
                          "metodo_id": "fingerprint"})
            n_fingerprint += 1
            logging.info(f"[{idx}] FINGERPRINT {ponente} → {fp['nombre']} ({fp['confianza']:.2f})")
            continue

        # ── 2. Heurística ───────────────────────────────────────────────
        h = detectar_patrones_heuristicos(chunks, idx)
        if h["nombre"] or h["partido"]:
            conf = h["confianza"]
            estado = "IDENTIFICADO" if conf >= 0.90 else "AMBIGUO"

            chunk.update({"nombre": h["nombre"], "partido": h["partido"],
                          "estado_id": estado, "confianza_id": h["confianza"],
                          "metodo_id": "heuristica"})
            if estado == "IDENTIFICADO":
                actualizar_fingerprint(fingerprints, ponente, h)
                n_heuristica += 1
            else:
                n_ambiguo += 1
            logging.info(
                f"[{idx}] HEURÍSTICA {ponente} → {h['nombre']} | {h['partido']} "
                f"({h['confianza']:.2f}) | '{h['fragmento_detectado']}'"
            )
            continue

        # ── 3. Análisis cambio de turno ─────────────────────────────────────
        cambio_acustico = (idx == 0 or chunks[idx]["ponente"] != chunks[idx - 1]["ponente"])
        cambio_turno = detectar_cambio_turno(chunks, idx)

        if not cambio_acustico and not cambio_turno:
            n_desconocido += 1
            logging.info(f"[{idx}] SKIP sin cambio acústico ni textual")
            continue

        # LLM-1 eliminado (v2.8): los chunks que llegan aquí quedan
        # DESCONOCIDO/AMBIGUO y se resuelven después con LLM-DESC.
        if cambio_acustico and not cambio_turno:
            chunk["estado_id"] = "AMBIGUO"
            n_ambiguo += 1
            logging.info(f"[{idx}] AMBIGUO — cambio acústico sin cambio textual (pendiente LLM-DESC)")
        else:
            n_desconocido += 1
            logging.info(f"[{idx}] DESCONOCIDO — cambio textual detectado, pendiente LLM-DESC")

    # ── FIX 4: fusión de fingerprints ────────────────────────────────────
    n_fusion = fusionar_fingerprints_por_nombre(chunks, fingerprints)
    if n_fusion:
        logging.info(f"FUSIÓN fingerprints: {n_fusion} chunks actualizados")

    # ── LLM-DESC: identificar desconocidos con contexto de Mesa ───────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if usar_llm and groq_key:
        n_llm_desc = identificar_desconocidos_con_llm(chunks, fingerprints, api_key=groq_key)
        if n_llm_desc:
            logging.info(f"LLM-DESC: {n_llm_desc} chunks identificados")
        n_llm_dict = verificar_nombres_fuera_diccionario_con_llm(chunks, fingerprints, api_key=groq_key)
        if n_llm_dict:
            logging.info(f"LLM-DICT: {n_llm_dict} nombres corregidos")
    elif usar_llm and not groq_key:
        print("[LLM-DESC/DICT] ⚠ GROQ_API_KEY no encontrada, se omite corrección por LLM")
        logging.warning("GROQ_API_KEY no encontrada")

    # ── PROPAGACIÓN GLOBAL por SPEAKER_ID ─────────────────────────────────
    nombre_por_speaker = {}
    for c in chunks:
        if c["estado_id"] == "IDENTIFICADO" and c.get("nombre") and c["confianza_id"] >= 0.90:
            sp = c["ponente"]
            if sp not in nombre_por_speaker:
                nombre_por_speaker[sp] = []
            entrada = (c["nombre"], c.get("partido"))
            if entrada not in nombre_por_speaker[sp]:
                nombre_por_speaker[sp].append(entrada)

    n_prop = 0
    for c in chunks:
        if c["estado_id"] in ("DESCONOCIDO", "AMBIGUO") and not c.get("nombre"):
            sp = c["ponente"]
            if sp in nombre_por_speaker and len(nombre_por_speaker[sp]) == 1:
                nombre_p, partido_p = nombre_por_speaker[sp][0]
                c["nombre"] = nombre_p
                c["partido"] = partido_p
                c["estado_id"] = "IDENTIFICADO"
                c["confianza_id"] = 0.88
                c["metodo_id"] = "propagacion_global"
                n_prop += 1
    if n_prop:
        logging.info(f"PROPAGACIÓN GLOBAL: {n_prop} chunks")

    guardar_json(chunks, ruta_json_salida)

    # ── Resumen final ────────────────────────────────────────────────────
    total = len(chunks)
    identificados = sum(1 for c in chunks if c["estado_id"] == "IDENTIFICADO")
    n_ambiguo_final = sum(1 for c in chunks if c["estado_id"] == "AMBIGUO")
    n_desconocido_final = sum(1 for c in chunks if c["estado_id"] == "DESCONOCIDO")

    print(f"\n--- RESUMEN IDENTIFICACIÓN v2.14 ---")
    print(f"Total chunks:               {total}")
    print(f"Identificados:              {identificados} ({identificados*100//total if total else 0}%)")
    print(f"  Por heurística:           {n_heuristica}")
    print(f"  Por heurística (cierre):  {n_cierre_retroactivo}")
    print(f"  Por fingerprint:          {n_fingerprint}")
    print(f"  Por fusión fp:            {n_fusion}")
    print(f"Ambiguos:                   {n_ambiguo_final} ({n_ambiguo_final*100//total if total else 0}%)")
    print(f"Desconocidos:               {n_desconocido_final} ({n_desconocido_final*100//total if total else 0}%)")
    n_llm_desc_tot = sum(1 for c in chunks if c.get("metodo_id") == "llm_desconocidos")
    n_llm_dict_tot = sum(1 for c in chunks if c.get("metodo_id") == "llm_correccion_dict")
    n_prop_tot     = sum(1 for c in chunks if c.get("metodo_id") == "propagacion_global")
    if n_llm_desc_tot: print(f"  Por LLM-DESC (desconocidos): {n_llm_desc_tot}")
    if n_llm_dict_tot: print(f"  Por LLM-DICT (corrección):   {n_llm_dict_tot}")
    if n_prop_tot:     print(f"  Por propagación global:      {n_prop_tot}")
    print(f"Speakers únicos:            {len(set(c['ponente'] for c in chunks))}")
    print(f"Fingerprints activos:       {sum(1 for v in fingerprints.values() if not v.get('ambiguo'))}")
    print(f"\nJSON guardado en:  {ruta_json_salida}")
    print(f"Log guardado en:   {ruta_log}")

    logging.info(
        f"FIN v2.14 — identificados={identificados}, cierre={n_cierre_retroactivo}, "
        f"fusion={n_fusion}, ambiguos={n_ambiguo_final}, "
        f"desconocidos={n_desconocido_final}"
    )
    return ruta_json_salida


# =============================================================
# ENTRYPOINT
# =============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python identificador_speakers_v2_14.py <ruta_json> [--sin-llm]")
        sys.exit(1)

    ruta = Path(sys.argv[1])
    if not ruta.exists():
        print(f"Error: no existe {ruta}")
        sys.exit(1)

    identificar_video(ruta, usar_llm="--sin-llm" not in sys.argv)