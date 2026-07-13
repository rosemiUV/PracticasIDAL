import os
import sys
from pathlib import Path

# Añadir la raíz del proyecto al PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.motor_busqueda.db_neo4j import db

print('Borrando TODA la base de datos Neo4j (nodos y relaciones)...')

query1 = "MATCH (n) DETACH DELETE n"
db.execute_write(query1)

from src.motor_busqueda.db_neo4j import seed_db
print('Plantando semillas básicas de nuevo...')
seed_db()

print('Caché limpiada COMPLETAMENTE. Base de datos reseteada.')
