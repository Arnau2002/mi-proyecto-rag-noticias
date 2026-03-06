import asyncio
from src.services.vector_store import search_vectors
from dotenv import load_dotenv

load_dotenv()

async def test_search():
    query = "¿Qué hay de nuevo sobre OpenAI y modelos de lenguaje?"
    print(f"Buscando: {query}")
    results = await search_vectors(query, top_k=5)
    
    print(f"\nResultados encontrados: {len(results)}")
    for i, res in enumerate(results):
        print(f"[{i+1}] Score: {res.score:.4f} | Title: {res.payload.get('title')[:100]}...")

if __name__ == "__main__":
    import os
    import sys
    # Add src to sys.path
    # sys.path.append(os.path.join(os.getcwd(), 'src'))
    asyncio.run(test_search())
