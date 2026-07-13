import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Cargar variables de entorno
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
# Solución al proxy de la universidad: cambiamos a +ssc para aceptar certificados autofirmados
if NEO4J_URI and "+s://" in NEO4J_URI:
    NEO4J_URI = NEO4J_URI.replace("+s://", "+ssc://")

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
            self.__driver.verify_connectivity()
            logger.info("Conexión exitosa a Neo4j Aura.")
        except Exception as e:
            logger.error("Error conectando a Neo4j: %s", e)
            
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            logger.info("Conexión a Neo4j cerrada.")
            
    def execute_query(self, query, parameters=None, db=None):
        """Ejecuta cualquier query (lectura o escritura) usando la API moderna de Neo4j 5/6"""
        assert self.__driver is not None, "Driver not initialized!"
        target_db = db if db else NEO4J_DATABASE
        try:
            records, summary, keys = self.__driver.execute_query(
                query,
                parameters_ = parameters,
                database_=target_db
            )
            return [record.data() for record in records]
        except Exception as e:
            logger.error("Ejecución de query fallida: %s", e)
            return None
            
    # Mantenemos estos nombres por compatibilidad con el resto del script,
    # pero ahora llaman a la misma API moderna por debajo.
    def query(self, query, parameters=None, db=None):
        return self.execute_query(query, parameters, db)

    def execute_write(self, query, parameters=None, db=None):
        return self.execute_query(query, parameters, db)

# Instancia global para ser importada por otros módulos
db = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

def init_db():
    """Crea los índices y restricciones iniciales de la base de datos"""
    # Restricciones de unicidad (ID único para cada entidad)
    db.query("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE")
    db.query("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Partido) REQUIRE o.id IS UNIQUE")
    db.query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Ley) REQUIRE l.id IS UNIQUE")
    logger.info("Índices y restricciones inicializados en Neo4j.")

def seed_db():
    """Añade algunos datos de prueba iniciales si la base de datos está vacía"""
    # Crear partido
    db.execute_write('''
        MERGE (pp:Partido {id: "org_pp"})
        ON CREATE SET pp.nombre = "Partido Popular", pp.siglas = "PP", pp.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Marca_PP_2022.png/960px-Marca_PP_2022.png", pp.alias = "PP, Partido Popular, P.P."
    ''')
    db.execute_write('''
        MERGE (psoe:Partido {id: "org_psoe"})
        ON CREATE SET psoe.nombre = "Partido Socialista Obrero Español", psoe.siglas = "PSOE", psoe.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Logo_PSOE_41_Congreso.svg/250px-Logo_PSOE_41_Congreso.svg.png", psoe.alias = "PSOE, Partido Socialista, P.S.O.E., Partido Socialista Obrero Español"
    ''')
    db.execute_write('''
        MERGE (junts:Partido {id: "org_junts"})
        ON CREATE SET junts.nombre = "Junts per Catalunya", junts.siglas = "Junts", junts.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Logo_partit_Junts_per_Catalunya.png/250px-Logo_partit_Junts_per_Catalunya.png", junts.alias = "Junts, JxCat, Junts per Catalunya"
    ''')
    db.execute_write('''
        MERGE (podemos:Partido {id: "org_podemos"})
        ON CREATE SET podemos.nombre = "Podemos", podemos.siglas = "PODEMOS", podemos.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Podemos_logo_completo_%282022%29.svg/330px-Podemos_logo_completo_%282022%29.svg.png", podemos.alias = "Podemos, PODEMOS, Unidas Podemos"
    ''')
    db.execute_write('''
        MERGE (erc:Partido {id: "org_erc"})
        ON CREATE SET erc.nombre = "Esquerra Republicana de Catalunya", erc.siglas = "ERC", erc.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/ERC_logo_2025.svg/330px-ERC_logo_2025.svg.png", erc.alias = "ERC, Esquerra, Esquerra Republicana, Esquerra Republicana de Catalunya"
    ''')
    db.execute_write('''
        MERGE (bildu:Partido {id: "org_bildu"})
        ON CREATE SET bildu.nombre = "Euskal Herria Bildu", bildu.siglas = "EH Bildu", bildu.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Logo_de_EH_Bildu_%282023%29.svg/250px-Logo_de_EH_Bildu_%282023%29.svg.png", bildu.alias = "Bildu, EH Bildu, Euskal Herria Bildu"
    ''')
    db.execute_write('''
        MERGE (vox:Partido {id: "org_vox"})
        ON CREATE SET vox.nombre = "Vox", vox.siglas = "VOX", vox.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/VOX_logo.svg/250px-VOX_logo.svg.png", vox.alias = "Vox, VOX"
    ''')
    db.execute_write('''
        MERGE (sumar:Partido {id: "org_sumar"})
        ON CREATE SET sumar.nombre = "Sumar", sumar.siglas = "SUMAR", sumar.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Sumar_logo.svg/250px-Sumar_logo.svg.png", sumar.alias = "Sumar, SUMAR"
    ''')
    db.execute_write('''
        MERGE (pnv:Partido {id: "org_pnv"})
        ON CREATE SET pnv.nombre = "Partido Nacionalista Vasco", pnv.siglas = "PNV", pnv.logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Logo_PNV_2025.svg/250px-Logo_PNV_2025.svg.png", pnv.alias = "PNV, Partido Nacionalista Vasco, EAJ-PNV, EAJ"
    ''')
    
    # Crear personas
    db.execute_write('''
        MERGE (cuca:Persona {id: "per_cucagamarra"})
        ON CREATE SET cuca.nombre = "Cuca Gamarra", cuca.rol = "Portavoz del PP"
        MERGE (pp:Partido {id: "org_pp"})
        MERGE (cuca)-[:PERTENECE_A]->(pp)
    ''')
    db.execute_write('''
        MERGE (noelia:Persona {id: "per_noeliasimil"})
        ON CREATE SET noelia.nombre = "Noèlia Madrera Simil", noelia.rol = "Diputada"
        MERGE (junts:Partido {id: "org_junts"})
        MERGE (noelia)-[:PERTENECE_A]->(junts)
    ''')
    
    # Crear leyes
    db.execute_write('''
        MERGE (cp:Ley {id: "ley_codigopenal"})
        ON CREATE SET cp.nombre = "Código Penal", 
                      cp.resumen = "Conjunto de normas jurídicas punitivas de un Estado.",
                      cp.url = "https://www.boe.es/buscar/act.php?id=BOE-A-1995-25444"
    ''')
    
    logger.info("Base de datos de grafos inicializada con datos semilla.")

def get_all_entities_for_prompt():
    """Recupera las entidades (Personas, Partidos, Leyes) para inyectarlas en el prompt del LLM"""
    texto_entidades = "CONOCIMIENTO DE ENTIDADES RELEVANTES (Usa estas etiquetas en tu respuesta si mencionas a estas entidades):\n"
    
    # Personas
    personas = db.execute_query("MATCH (p:Persona) RETURN p.id AS id, p.nombre AS nombre")
    if personas:
        texto_entidades += "- Personas:\n"
        for p in personas:
            texto_entidades += f"  - {p['nombre']} -> <entidad id=\"{p['id']}\">{p['nombre']}</entidad>\n"
            
    # Partidos
    partidos = db.execute_query("MATCH (o:Partido) RETURN o.id AS id, o.nombre AS nombre, o.siglas AS siglas, o.alias AS alias")
    if partidos:
        texto_entidades += "- Partidos/Organizaciones:\n"
        for o in partidos:
            # Procesar los alias, separándolos por coma (si los hay)
            alias_list = [a.strip() for a in (o.get('alias') or '').split(',') if a.strip()]
            
            # Formatear cada alias como una entidad clickeable
            alias_str = " o ".join([f'<entidad id="{o["id"]}">{a}</entidad>' for a in alias_list])
            
            # Formatear también las siglas (legado) si existen y no están en los alias
            if o.get('siglas') and o['siglas'] not in alias_list and o['siglas'] != o['nombre']:
                extra_sigla = f' o <entidad id="{o["id"]}">{o["siglas"]}</entidad>'
            else:
                extra_sigla = ''
                
            if alias_str:
                texto_entidades += f"  - {o['nombre']} -> <entidad id=\"{o['id']}\">{o['nombre']}</entidad> o {alias_str}{extra_sigla}\n"
            else:
                texto_entidades += f"  - {o['nombre']} -> <entidad id=\"{o['id']}\">{o['nombre']}</entidad>{extra_sigla}\n"
            
    # Leyes
    leyes = db.execute_query("MATCH (l:Ley) RETURN l.id AS id, l.nombre AS nombre")
    if leyes:
        texto_entidades += "- Leyes:\n"
        for l in leyes:
            texto_entidades += f"  - {l['nombre']} -> <entidad id=\"{l['id']}\">{l['nombre']}</entidad>\n"
            
    # Eventos
    eventos = db.execute_query("MATCH (ev:Evento) RETURN ev.id AS id, ev.nombre AS nombre")
    if eventos:
        texto_entidades += "- Eventos/Sucesos:\n"
        for ev in eventos:
            texto_entidades += f"  - {ev['nombre']} -> <entidad id=\"{ev['id']}\">{ev['nombre']}</entidad>\n"
            
    # Conceptos
    conceptos = db.execute_query("MATCH (c:Concepto) RETURN c.id AS id, c.nombre AS nombre")
    if conceptos:
        texto_entidades += "- Programas/Conceptos:\n"
        for c in conceptos:
            texto_entidades += f"  - {c['nombre']} -> <entidad id=\"{c['id']}\">{c['nombre']}</entidad>\n"
            
    return texto_entidades

def obtener_entidad_existente(nombre: str, tipo: str) -> dict:
    """
    Busca si una entidad ya existe en Neo4j por su nombre aproximado.
    Devuelve un diccionario con 'explicacion' y 'foto_url' si existe, o None si no.
    """
    nombre_limpio = ''.join(c for c in nombre if c.isalnum()).lower()
    if not nombre_limpio:
        return None
        
    query = ""
    if tipo == "persona":
        query = "MATCH (n:Persona) WHERE toLower(n.nombre) = toLower($nombre) OR n.id = $id RETURN n.rol AS explicacion, n.foto_url AS foto_url, n.fuente AS fuente LIMIT 1"
    elif tipo == "ley":
        query = "MATCH (n:Ley) WHERE toLower(n.nombre) = toLower($nombre) OR n.id = $id RETURN n.resumen AS explicacion, n.foto_url AS foto_url, n.fuente AS fuente LIMIT 1"
    elif tipo in ["institucion", "partido"]:
        query = "MATCH (n:Partido) WHERE toLower(n.nombre) = toLower($nombre) OR n.id = $id OR toLower(n.siglas) = toLower($nombre) RETURN n.descripcion AS explicacion, n.logo_url AS foto_url, n.fuente AS fuente LIMIT 1"
    elif tipo == "lugar":
        query = "MATCH (n:Lugar) WHERE toLower(n.nombre) = toLower($nombre) OR n.id = $id RETURN n.descripcion AS explicacion, n.foto_url AS foto_url, n.fuente AS fuente LIMIT 1"
    elif tipo == "evento":
        query = "MATCH (n:Evento) WHERE toLower(n.nombre) = toLower($nombre) OR n.id = $id RETURN n.descripcion AS explicacion, n.foto_url AS foto_url, n.fuente AS fuente LIMIT 1"
    elif tipo == "concepto":
        query = "MATCH (n:Concepto) WHERE toLower(n.nombre) = toLower($nombre) OR n.id = $id RETURN n.descripcion AS explicacion, n.foto_url AS foto_url, n.fuente AS fuente LIMIT 1"
        
    if query:
        res = db.execute_query(query, {"nombre": nombre, "id": f"{tipo[:3]}_{nombre_limpio}"})
        if res:
            return res[0]
    return None

def guardar_entidad_dinamica(nombre: str, tipo: str, explicacion: str, partido: str = None, video_id: str = None, alias: str = None, foto_url: str = None, fuente: str = None) -> str:
    """
    Guarda o actualiza una entidad dinámica extraída por el LLM en Neo4j.
    Crea las relaciones `(Video)-[:MENCIONA]->(Entidad)` necesarias.
    """
    nombre_limpio = ''.join(c for c in nombre if c.isalnum()).lower()
    
    query_video = '''
        WITH e
        MATCH (v:Video {id: $video_id})
        MERGE (v)-[:MENCIONA]->(e)
    '''

    if tipo == "persona":
        id_entidad = f"per_{nombre_limpio}"
        query = '''
            MERGE (p:Persona {id: $id})
            ON CREATE SET p.nombre = $nombre, p.rol = $explicacion
        '''
        if foto_url:
            query += ', p.foto_url = $foto_url'
        if fuente:
            query += ', p.fuente = $fuente'
            
        if video_id:
            query += query_video.replace('WITH e', 'WITH p AS e')
        db.execute_write(query, {"id": id_entidad, "nombre": nombre, "explicacion": explicacion, "video_id": video_id, "foto_url": foto_url, "fuente": fuente})
        
        if partido:
            partido_id = f"org_{''.join(c for c in partido if c.isalnum()).lower()}"
            db.execute_write('''
                MATCH (p:Persona {id: $per_id})
                MERGE (o:Partido {id: $org_id})
                ON CREATE SET o.nombre = $org_nombre, o.siglas = $org_nombre
                MERGE (p)-[:PERTENECE_A]->(o)
            ''', {"per_id": id_entidad, "org_id": partido_id, "org_nombre": partido})
            
    elif tipo == "ley":
        id_entidad = f"ley_{nombre_limpio}"
        url_boe = f"https://www.boe.es/buscar/boe.php?q={urllib.parse.quote(nombre)}"
        query = '''
            MERGE (l:Ley {id: $id})
            ON CREATE SET l.nombre = $nombre, l.resumen = $explicacion, l.url = $url
        '''
        if foto_url:
            query += ', l.foto_url = $foto_url'
        if fuente:
            query += ', l.fuente = $fuente'
            
        if video_id:
            query += query_video.replace('WITH e', 'WITH l AS e')
        db.execute_write(query, {"id": id_entidad, "nombre": nombre, "explicacion": explicacion, "url": url_boe, "video_id": video_id, "foto_url": foto_url, "fuente": fuente})
        
    elif tipo == "institucion" or tipo == "partido":
        id_entidad = f"org_{nombre_limpio}" # ID por defecto
        
        # 1. Buscar si ya existe un partido con este nombre o en sus alias
        buscar_id_query = """
            MATCH (o:Partido)
            WHERE toLower(o.nombre) = toLower($nombre)
               OR toLower(o.siglas) = toLower($nombre)
               OR toLower(o.alias) CONTAINS toLower($nombre)
            RETURN o.id AS id LIMIT 1
        """
        resultado = db.execute_query(buscar_id_query, {"nombre": nombre})
        
        # 2. Si no, buscar si alguno de los alias que nos dio el LLM coincide con algo existente
        if not resultado and alias:
            for a in [x.strip() for x in alias.split(',') if x.strip()]:
                res_alias = db.execute_query(buscar_id_query, {"nombre": a})
                if res_alias:
                    resultado = res_alias
                    break
                    
        # 3. Si lo hemos encontrado, usamos su ID existente para evitar duplicados
        if resultado:
            id_entidad = resultado[0]['id']
            
        query = '''
            MERGE (o:Partido {id: $id})
            ON CREATE SET o.nombre = $nombre, o.siglas = $nombre, o.descripcion = $explicacion
        '''
        if alias:
            # Solo lo ponemos al crear para no machacar alias previos más completos
            query += ', o.alias = $alias'
            
        if fuente:
            query += ', o.fuente = $fuente'
            
        query += '''
            ON MATCH SET o.descripcion = CASE WHEN o.descripcion IS NULL THEN $explicacion ELSE o.descripcion END
        '''
        
        if foto_url:
            query += '\n SET o.logo_url = $foto_url'
            
        if video_id:
            query += query_video.replace('WITH e', 'WITH o AS e')
            
        params = {"id": id_entidad, "nombre": nombre, "explicacion": explicacion, "video_id": video_id, "foto_url": foto_url, "fuente": fuente}
        if alias:
            params["alias"] = alias
        db.execute_write(query, params)
        
    elif tipo == "lugar":
        id_entidad = f"lug_{nombre_limpio}"
        query = '''
            MERGE (l:Lugar {id: $id})
            ON CREATE SET l.nombre = $nombre, l.descripcion = $explicacion
        '''
        if foto_url:
            query += ', l.foto_url = $foto_url'
        if fuente:
            query += ', l.fuente = $fuente'
            
        if video_id:
            query += query_video.replace('WITH e', 'WITH l AS e')
        db.execute_write(query, {"id": id_entidad, "nombre": nombre, "explicacion": explicacion, "video_id": video_id, "foto_url": foto_url, "fuente": fuente})

    elif tipo == "evento":
        id_entidad = f"evt_{nombre_limpio}"
        query = '''
            MERGE (ev:Evento {id: $id})
            ON CREATE SET ev.nombre = $nombre, ev.descripcion = $explicacion
        '''
        if foto_url:
            query += ', ev.foto_url = $foto_url'
        if fuente:
            query += ', ev.fuente = $fuente'
            
        if video_id:
            query += query_video.replace('WITH e', 'WITH ev AS e')
        db.execute_write(query, {"id": id_entidad, "nombre": nombre, "explicacion": explicacion, "video_id": video_id, "foto_url": foto_url, "fuente": fuente})

    elif tipo == "concepto":
        id_entidad = f"con_{nombre_limpio}"
        query = '''
            MERGE (c:Concepto {id: $id})
            ON CREATE SET c.nombre = $nombre, c.descripcion = $explicacion
        '''
        if foto_url:
            query += ', c.foto_url = $foto_url'
        if fuente:
            query += ', c.fuente = $fuente'
            
        if video_id:
            query += query_video.replace('WITH e', 'WITH c AS e')
        db.execute_write(query, {"id": id_entidad, "nombre": nombre, "explicacion": explicacion, "video_id": video_id, "foto_url": foto_url, "fuente": fuente})
        
    else:
        logger.warning(f"Tipo de entidad desconocido: {tipo}")
        return None
        
    logger.info(f"Entidad guardada en Neo4j: {nombre} ({tipo})")
    return id_entidad

if __name__ == "__main__":
    init_db()
    seed_db()
    print("Neo4j Setup completado con éxito.")
