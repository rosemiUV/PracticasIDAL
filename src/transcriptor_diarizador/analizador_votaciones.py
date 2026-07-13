import re
import json
import pandas as pd
from pathlib import Path

def extraer_info_votacion(bloque):
    """Extrae resultados, calcula el total y detecta el objeto de forma robusta."""
    
    # 1. Extraer resultados
    patrones = {
        "favor": r"(?:a favor|¿a favor\?)[,:\s]*(\d+)",
        "contra": r"(?:en contra|¿en contra\?)[,:\s]*(\d+)",
        "abstenciones": r"(?:abstenciones|¿abstenciones\?)[,:\s]*(\d+)"
    }
    
    resultados = {k: int(re.search(p, bloque, re.IGNORECASE).group(1)) 
                  for k, p in patrones.items() if re.search(p, bloque, re.IGNORECASE)}
    
    total = sum(resultados.values())
    
    # 2. Extraer "QUÉ SE VOTA" de forma mejorada
    # Busca disparadores de inicio y captura hasta el siguiente anuncio de resultado o punto
    patron_que = r"(?:votamos|empezamos|se vota|vota la|vota el|votación de)\.?\s+(.*?)(?:\.|\?|\s+empezamos|votos emitidos|votamos|¿votamos|¿empezamos)"
    objeto_match = re.search(patron_que, bloque, re.IGNORECASE | re.DOTALL)
    
    if objeto_match:
        objeto = objeto_match.group(1).strip()
        # Limpieza: quitamos artículos iniciales sobrantes
        objeto = re.sub(r"^(ahora|el|la|los|las)\s+", "", objeto, flags=re.IGNORECASE)
    else:
        objeto = "Objeto no identificado"
    
    return resultados, total, objeto

def procesar_votaciones_definitivo(ruta_json):
    """Procesa un archivo JSON buscando múltiples votaciones por bloque."""
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        df = pd.DataFrame(datos)
        # Filtramos solo la Mesa
        mesa_df = df[df['partido'] == "Mesa"].sort_values('inicio')

        # Regex para separar múltiples votaciones dentro de un mismo texto largo
        regex_separador = r"(?i)(?=Votamos ahora|Empezamos votando|Votamos la|Votamos el|Votamos a favor)"

        for _, row in mesa_df.iterrows():
            texto_largo = row['texto']
            bloques = re.split(regex_separador, texto_largo)
            
            for bloque in bloques:
                if len(bloque.strip()) < 10: continue
                
                resultados, total, objeto = extraer_info_votacion(bloque)
                
                # Filtro de robustez: detectamos votación solo si hay al menos 2 tipos de resultados
                if len(resultados) >= 2:
                    print(f"--- Votación detectada en {ruta_json.name} (Min {int(row['inicio']/60)}) ---")
                    print(f"QUÉ: {objeto.capitalize()}")
                    print(f"RESULTADOS: {resultados} | TOTAL: {total}")
                    print("-" * 50)
    except Exception as e:
        print(f"Error procesando {ruta_json.name}: {e}")

if __name__ == "__main__":
    carpeta = Path(__file__).parent / "resultados_finales"
    for archivo in carpeta.glob("*.json"):
        procesar_votaciones_definitivo(archivo)