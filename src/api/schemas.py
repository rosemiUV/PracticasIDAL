'''
Modelos de datos (Pydantic) para validación estricta
'''

from pydantic import BaseModel
from typing import List, Optional

#1. Lo que recibimos del Frontend (Input)
class SearchRequest(BaseModel):
    pregunta: str
    id_sesion: Optional[str] = None
    is_global: bool = False

class Fuente(BaseModel):
    ponente: str
    texto: str
    enlace_video: str
    nombre: Optional[str] = None
    partido: Optional[str] = None
    inicio: Optional[str] = None
    fin: Optional[str] = None
    inicio_segundos: Optional[float] = None
    fin_segundos: Optional[float] = None
    
    class Config:
        extra = "allow"

class SearchResponse(BaseModel):
    pregunta: str
    respuesta_llm: str
    fuentes_top_k: List[Fuente]

class UrlVideoRequest(BaseModel):
    url: Optional[str] = None
    urls: Optional[List[str]] = None
    client_id: Optional[str] = None

class ContextRequest(BaseModel):
    video_id: str
    ponente: str
    inicio: float
    fin: float

class SummaryRequest(BaseModel):
    video_id: str

class EntitiesRequest(BaseModel):
    video_id: str
    pregunta: Optional[str] = None
    force_refresh: bool = False

class StatsRequest(BaseModel):
    video_id: str
