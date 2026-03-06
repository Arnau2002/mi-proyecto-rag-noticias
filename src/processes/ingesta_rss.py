import asyncio
import feedparser
from bs4 import BeautifulSoup
import sys
import os

# Ajustar PYTHONPATH para permitir importaciones relativas desde la raíz del src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import init_db, upsert_documents, settings

def chunk_text(text: str, max_words: int = 200) -> list[str]:
    """Divide un texto largo en fragmentos (chunks) más manejables de n palabras."""
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extract_text_from_html(html_content: str) -> str:
    """Usa BeautifulSoup para extraer texto plano del HTML omitiendo etiquetas."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=' ', strip=True)

async def process_rss():
    """Lee el feed RSS público, extrae texto, hace chunking y envía a Base de Datos."""
    print("Iniciando procesamiento del flujo RSS (Ingesta de datos)...")
    await init_db()
    
    rss_url = settings.rss_url
    print(f"Descargando feed desde: {rss_url}")
    feed = feedparser.parse(rss_url)
    
    documents = []
    for entry in feed.entries:
        title = entry.get("title", "Sin título")
        link = entry.get("link", "")
        # Algunos RSS utilizan 'published' y otros 'updated' para la fecha
        date = entry.get("published", entry.get("updated", ""))
        
        # El contenido principal puede venir bajo diferentes claves
        html_content = ""
        if "content" in entry:
            html_content = entry.content[0].value
        elif "summary" in entry:
            html_content = entry.summary
            
        text = extract_text_from_html(html_content)
        
        if not text:
            # Si tras la limpieza está vacío, lo omitimos
            continue
            
        print(f"Procesando noticia: {title}")
        
        # Aplicar el Chunking Básico
        chunks = chunk_text(text)
        
        # Para cada parte de la noticia generamos un documento independiente a indexar
        for i, chunk in enumerate(chunks):
            # Para evitar sobreescribir partes diferentes de una misma URL, añadimos un sufijo hash/index al source link
            # Esto cumple el requisito de "usar hash de la url como ID" permitiendo múltiples partes
            chunk_url = f"{link}#chunk{i}"
            
            documents.append({
                "text": chunk,
                "url": chunk_url,
                "title": title,
                "date": date,
                "source": link,
                "category": "Inteligencia Artificial"
            })
            
    print(f"Total chunks generados: {len(documents)}. Enviando a Qdrant y calculando embeddings...")
    
    if documents:
        try:
            # Inserción en lotes (batch) para no colapsar / hacer rate-limiting de la API de Gemini
            batch_size = 20
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                await upsert_documents(batch)
                print(f"Lote indexado satisfactoriamente ({len(batch)} chunks).")
                await asyncio.sleep(1) # Pequeña pausa de cortesía para la API
            print("El proceso de Ingesta y Upsert de noticias ha finalizado. Base de datos actualizada.")
        except Exception as e:
            print(f"Ocurrió un error al insertar documentos en Qdrant: {e}")
    else:
        print("No se encontraron documentos válidos en el RSS.")

if __name__ == "__main__":
    try:
        asyncio.run(process_rss())
    except KeyboardInterrupt:
        print("Operación cancelada por el usuario.")
