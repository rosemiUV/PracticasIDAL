import json
import urllib.request
import urllib.parse
from pathlib import Path
import pandas as pd
import ssl
import concurrent.futures
import re

DB_PATH = Path(__file__).parent / "wiki_diputados.json"
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_wiki_info(nombre_completo):
    # Convertimos 'Apellidos, Nombre' a 'Nombre Apellidos' para buscar mejor en Wikipedia
    partes = nombre_completo.split(',')
    if len(partes) == 2:
        nombre_buscar = f"{partes[1].strip()} {partes[0].strip()}"
    else:
        nombre_buscar = nombre_completo
        
    query = urllib.parse.quote(nombre_buscar)
    url = f"https://es.wikipedia.org/w/api.php?action=query&format=json&prop=extracts|pageimages&exintro=1&explaintext=1&pithumbsize=400&redirects=1&titles={query}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BPI-Bot/1.0'})
        response = urllib.request.urlopen(req, timeout=10).read()
        data = json.loads(response)
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return nombre_completo, "", ""
            extract = page_data.get("extract", "")
            img_url = page_data.get("thumbnail", {}).get("source", "")
            return nombre_completo, extract, img_url
    except Exception as e:
        pass
    return nombre_completo, "", ""

def create_real_db():
    print("Descargando tabla oficial de diputados de Wikipedia (XV Legislatura)...")
    url = 'https://es.wikipedia.org/wiki/Anexo:Diputados_de_la_XV_legislatura_de_Espa%C3%B1a'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read()

    tables = pd.read_html(html)
    # Tabla 1 contiene los diputados activos
    df = tables[1]
    
    diputados = []
    
    # Limpiar nombres de notas al pie como [11]
    nombres = df['Nombre y apellidos'].apply(lambda x: re.sub(r'\[\d+\]', '', str(x)).strip()).tolist()
    
    # Rellenar valores NaN en Candidatura.1 (celdas combinadas en Wikipedia)
    col_partido = 'Candidatura.1' if 'Candidatura.1' in df.columns else 'Grupo.1'
    partidos = df[col_partido].ffill().tolist()
    
    print(f"Extrayendo biografías y fotos de {len(nombres)} diputados mediante la API de Wikipedia. Por favor, espera unos segundos...")
    
    wiki_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_name = {executor.submit(fetch_wiki_info, n): n for n in nombres}
        for future in concurrent.futures.as_completed(future_to_name):
            n, extract, img = future.result()
            wiki_data[n] = (extract, img)
            
    for i in range(len(nombres)):
        nombre = nombres[i]
        partido = partidos[i]
        if isinstance(partido, pd.Series):
            partido = partido.iloc[0]
            
        extract, img = wiki_data.get(nombre, ("", ""))
        
        # Fallbacks si Wikipedia no tiene info exacta
        if not extract or len(extract) < 10:
            extract = f"Diputado/a de la XV Legislatura por el partido {partido}."
            
        if not img:
            img = f"https://ui-avatars.com/api/?name={urllib.parse.quote(nombre)}&background=random"
            
        diputados.append({
            "id": i,
            "nombre": nombre,
            "foto_url": img,
            "partido": str(partido),
            "biografia": extract
        })
        
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(diputados, f, ensure_ascii=False, indent=2)
        
    print(f"¡Éxito! Base de datos REAL generada en {DB_PATH} con {len(diputados)} diputados.")

if __name__ == "__main__":
    create_real_db()
