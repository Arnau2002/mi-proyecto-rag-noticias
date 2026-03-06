import google.generativeai as genai
from services.vector_store import search_vectors

def format_context(results) -> str:
    """Extrae y formatea el texto de los resultados de Qdrant."""
    context_text = ""
    for i, res in enumerate(results):
        payload = res.payload
        context_text += f"\n--- Fuente {i+1} ---\n"
        context_text += f"Título: {payload.get('title')}\n"
        context_text += f"Fecha: {payload.get('date')}\n"
        context_text += f"URL: {payload.get('url')}\n"
        context_text += f"Contenido: {payload.get('text')}\n"
    return context_text

async def generate_rag_response(query: str) -> dict:
    """
    Orquesta el flujo RAG:
    1. Busca en Qdrant.
    2. Formatea el contexto.
    3. Construye el prompt estricto.
    4. Llama a Gemini.
    """
    # 1. Recuperación de contexto Top 5 de la BD Vectorial
    search_results = await search_vectors(query, top_k=5)
    
    if not search_results:
         return {
             "answer": "No encontré noticias recientes relevantes en la base de datos para responder a esa pregunta.",
             "sources": []
         }
         
    # 2. Formateo
    context = format_context(search_results)
    
    # 3. Prompt estricto
    prompt = f"""
Eres un asistente experto en noticias de tecnología y AI. Tu tarea es responder a la pregunta del usuario utilizando ÚNICAMENTE la información proporcionada en el siguiente contexto.
Si la respuesta no está en el contexto, responde "No tengo suficiente información en las noticias recientes para responder a esto" y no inventes respuestas.

Pregunta del usuario: {query}

Contexto (Noticias recuperadas):
{context}

Respuesta:
"""

    # 4. Generar Respuesta Final
    # Asumimos que Gemini 2.5 Flash L está mapeado como gemini-2.5-flash / gemini-1.5-flash
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # Usando versión genérica 2.5
    except:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
    try:
        response = model.generate_content(prompt)
        final_answer = response.text
    except Exception as e:
        final_answer = f"Ocurrió un error general consultando al modelo de lenguaje: {e}"

    # Extraer URLs fuente de las notas que se retornaron
    sources = [res.payload.get('url') for res in search_results]
    
    # Eliminar duplicados si hay chunks de la misma noticia devueltos
    base_sources = list(set([s.split("#")[0] for s in sources if s]))

    return {
        "answer": final_answer,
        "sources": base_sources
    }
