import google.generativeai as genai
from services.vector_store import search_vectors, Filter, FieldCondition, MatchValue
import datetime
import json

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

async def classify_query(query: str) -> dict:
    """Usa Gemini para clasificar la categoría y extraer la intención."""
    prompt = f"""
    Eres un clasificador de consultas para un sistema RAG de noticias de tecnología.
    Analiza la pregunta del usuario y devuelve un objeto JSON con dos campos:
    - 'category': Una palabra clave (ej. 'OpenAI', 'Apple', 'NVIDIA', 'IA', 'General') si la pregunta es específica sobre una empresa o tecnología. Si no, usa 'Tecnología'.
    - 'is_news_request': Booleano, true si el usuario busca noticias recientes.

    Pregunta: {query}

    Formato de salida (JSON puro):
    {{
        "category": "...",
        "is_news_request": true/false
    }}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await model.generate_content_async(prompt)
        # Limpiar posible markdown del JSON
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {"category": "Tecnología", "is_news_request": True}

async def generate_rag_response(query: str) -> dict:
    """
    Orquesta el flujo RAG:
    1. Busca en Qdrant.
    2. Formatea el contexto.
    3. Construye el prompt estricto.
    4. Llama a Gemini.
    """
    # 0. Clasificación y Enrutado Semántico
    classification = await classify_query(query)
    category = classification.get("category", "Tecnología")
    is_news = classification.get("is_news_request", True)
    
    print(f"\n--- [LOG RAG] ---")
    print(f"Pregunta: {query}")
    print(f"Categoría detectada: {category}")
    print(f"Es noticia reciente: {is_news}")
    
    # 1. Construcción de Filtros (Obsolescencia 30 días + Categoría)
    now = datetime.datetime.now()
    thirty_days_ago = (now - datetime.timedelta(days=30)).isoformat()
    
    print(f"Filtro temporal activo (Noticias > {thirty_days_ago})")

    conditions = [
        FieldCondition(key="date", range={"gte": thirty_days_ago})
    ]
    
    # Si la categoría no es genérica, enrutamos/filtramos
    if category != "Tecnología":
        print(f"Aplicando filtro de enrutamiento por categoría: {category}")
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    
    query_filter = Filter(must=conditions)

    # 2. Recuperación de contexto Top 5 de la BD Vectorial con Filtros
    search_results = await search_vectors(query, top_k=5, query_filter=query_filter)
    print(f"Resultados encontrados en BD: {len(search_results)}")
    
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
