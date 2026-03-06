import sys
import os

# Ajustar classpath para reconocer el module processes y services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from api.routers import search
from services.vector_store import init_db

app = FastAPI(
    title="API RAG Noticias IA",
    description="Sistema End-to-End para Retrieval-Augmented Generation sobre Noticias Tecnológicas.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Inicia la recolección inicial o comprueba la base de datos de vectores en el arranque del servidor."""
    print("Iniciando aplicación. Conectando con la Base de Datos Vectorial...")
    try:
        await init_db()
    except Exception as e:
        print(f"Advertencia al inicializar BD: ¿Levantaste el Qdrant en docker-compose? error={e}")

# Registrar Endpoints
app.include_router(search.router)

@app.get("/")
async def root():
    return {"message": "Bienvenido al Sistema RAG de Noticias IA. Haz llamadas POST a /search para consultar."}

if __name__ == "__main__":
    import uvicorn
    # Inicialización local para desarrollo 
    uvicorn.run("api.main:app", host="0.0.0.0", port=8088, reload=True)
