import sys
import os

# Ajustar PYTHONPATH para asegurar importaciones relativas dentro del paquete api
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from processes.routing import generate_rag_response

router = APIRouter(prefix="/search", tags=["RAG Search"])

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@router.post("/", response_model=QueryResponse)
async def search_endpoint(request: QueryRequest):
    """
    Endpoint RAG: 
    Recibe una pregunta, convierte a embeddings, busca contexto top 5 en BD, 
    genera respuesta LLM en base a ese contexto y retorna la respuesta final al usuario.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
        
    try:
        result = await generate_rag_response(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
