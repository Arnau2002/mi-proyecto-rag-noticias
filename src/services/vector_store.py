import os
import hashlib
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import google.generativeai as genai
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    google_api_key: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    rss_url: str = "https://feeds.weblogssl.com/xataka2" # Default value just in case

    class Config:
        env_file = ".env"

settings = Settings()

# Configurar API Key de Gemini
genai.configure(api_key=settings.google_api_key)

COLLECTION_NAME = "ai_news_collection"
VECTOR_SIZE = 3072  # Gemini gemini-embedding-001 dimension size

# Inicialización asíncrona de Qdrant client
qdrant_client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

async def init_db():
    """Inicializa la colección en Qdrant. Si existe con otra dimensión, la recrea."""
    collections = await qdrant_client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    recreate = False
    if COLLECTION_NAME in collection_names:
        info = await qdrant_client.get_collection(COLLECTION_NAME)
        current_size = info.config.params.vectors.size
        if current_size != VECTOR_SIZE:
            print(f"Dimensión detectada ({current_size}) no coincide con la esperada ({VECTOR_SIZE}). Recreando...")
            await qdrant_client.delete_collection(COLLECTION_NAME)
            recreate = True
    else:
        recreate = True

    if recreate:
        print(f"Creando colección {COLLECTION_NAME} en Qdrant con dim={VECTOR_SIZE}...")
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"La colección {COLLECTION_NAME} ya existe con la dimensión correcta.")

def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    """Genera embeddings usando Gemini Embedding 001."""
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task_type
    )
    return response['embedding']

async def upsert_documents(documents: list[dict]):
    """
    Inserta o actualiza documentos en Qdrant.
    Usa un hash de la URL (incluyendo el ID de chunk) como ID del punto
    para sobrescribir noticias si ya existen y evitar duplicados (Upsert).
    """
    points = []
    for doc in documents:
        # Generar hash UUID/Integer desde la url para que actúe como Point ID
        url_hash = hashlib.md5(doc['url'].encode('utf-8')).hexdigest()
        point_id = str(url_hash) # Qdrant acepta UUID str formato hex-like (e.g. 32 chars convertidos a guid válido o int). 
        # Es preferible usar UUID format usando python uuid o hash truncado. Vamos a usar un uint64
        point_id = int(url_hash[:15], 16)

        try:
            vector = get_embedding(doc['text'], task_type="retrieval_document")
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": doc.get('text', ''),
                        "url": doc.get('url', ''),
                        "title": doc.get('title', ''),
                        "date": doc.get('date', ''),
                        "source": doc.get('source', 'unknown'),
                        "category": doc.get('category', 'AI'),
                    }
                )
            )
        except Exception as e:
            print(f"Error generando embedding para la url {doc['url']}: {e}")
            continue
    
    if points:
        operation_info = await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        return operation_info
    
    return None

async def search_vectors(query: str, top_k: int = 5):
    """Busca en Qdrant los vectores más similares a la consulta del usuario."""
    try:
        query_embedding = get_embedding(query, task_type="retrieval_query")
        
        # Intentar con la API moderna query_points
        try:
                response = await qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True
                )
                return response.points
        except Exception as e:
            # Fallback a la API de búsqueda clásica
            print(f"Aviso: Falló query_points, intentando search... error={e}")
            search_result = await qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            return search_result
    except Exception as e:
        print(f"Error crítico en búsqueda Qdrant: {e}")
        return []
