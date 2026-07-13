import json
import difflib
from pathlib import Path
from thefuzz import process

DB_PATH = Path(__file__).parent / "diputados_db.json"

class CongresoAPI:
    def __init__(self):
        self.diputados = []
        self._cargar_db()

    def _cargar_db(self):
        if DB_PATH.exists():
            try:
                with open(DB_PATH, "r", encoding="utf-8") as f:
                    self.diputados = json.load(f)
            except Exception as e:
                print(f"Error cargando diputados_db.json: {e}")

    def buscar_diputado(self, nombre_buscado):
        """
        Busca un diputado usando fuzzy matching.
        Retorna el diccionario del diputado si encuentra una coincidencia aceptable, o None.
        """
        if not self.diputados:
            return None
            
        nombres_oficiales = [d["nombre"] for d in self.diputados]
        
        # Intenta primero con difflib
        matches = difflib.get_close_matches(nombre_buscado, nombres_oficiales, n=1, cutoff=0.6)
        
        # Si no funciona, thefuzz suele ser mejor para nombres desordenados ("Pilar Vallugera" vs "Vallugera Balañà, Pilar")
        if not matches:
            resultado = process.extractOne(nombre_buscado, nombres_oficiales, score_cutoff=60)
            if resultado:
                mejor_nombre, puntuacion = resultado[:2] # extractOne puede devolver (match, score, index)
                # Si la puntuación es razonable, lo aceptamos
                if puntuacion >= 65:
                    matches = [mejor_nombre]

        if matches:
            mejor_nombre = matches[0]
            for d in self.diputados:
                if d["nombre"] == mejor_nombre:
                    return d
                    
        return None

congreso_api = CongresoAPI()
