# Mi Proyecto RAG de Noticias 📰🤖

Sistema **Retrieval-Augmented Generation (RAG) End-to-End** sobre noticias de Inteligencia Artificial utilizando FastAPI, Qdrant y Gemini (Embeddings & LLM).

## Origen de las Noticias (Fuentes)
El sistema utiliza **Feeds RSS** públicos para obtener información actualizada. Por defecto, el proyecto está configurado para leer de **Xataka** (Tecnología), pero puedes cambiar la URL en el archivo `.env` por cualquier otro RSS compatible (ej. TechCrunch, Wired, etc.).

## Arquitectura y Componentes
Este desarrollo cumple con los requisitos del diseño de Bases de Datos Vectoriales:

1. **Ingesta de Datos (`src/processes/ingesta_rss.py`)**: Conecta al feed RSS de Xataka, extrae el texto puro, realiza chunking y genera vectores mediante `models/gemini-embedding-001` (3072 dimensiones).
2. **Base de Datos Vectorial (`src/services/vector_store.py`)**: Cliente asíncrono para Qdrant. Gestiona duplicados mediante el hash de la URL como ID de punto.
3. **Flujo RAG y Routing (`src/processes/routing.py`)**: Recupera el Top 5 de noticias más semánticamente cercanas a la duda del usuario y redacta una respuesta final usando **Gemini 2.5 Flash** basándose *estrictamente* en ese contexto.
4. **Backend API (`src/api/main.py`)**: Servidor FastAPI asíncrono configurado en el puerto **8088**.

## Instalación Paso a Paso

### 1. Prerrequisitos
- Tener instalado **Docker Desktop** (y abierto).
- Tener **Python 3.10+** instalado.
- Una **API Key** de Google AI Studio (Gemini).

### 2. Preparar el Entorno
Clona el repositorio o entra en la carpeta y ejecuta:

```powershell
# Crear entorno virtual
python -m venv venv

# ACTIVAR (en Windows PowerShell)
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Credenciales
Crea un archivo `.env` en la raíz del proyecto (copia el `.env.example`) y pega tu API Key:
```env
GOOGLE_API_KEY=AIzaSy... (tu clave aquí)
QDRANT_HOST=localhost
QDRANT_PORT=6333
RSS_URL=https://feeds.weblogssl.com/xataka2
```

### 4. Levantar la Base de Datos (Docker)
Asegúrate de que Docker Desktop está corriendo y ejecuta:
```bash
docker-compose up -d
```

---

## Flujo de Ejecución del Proyecto

Sigue este orden exacto para ver el sistema en funcionamiento:

### Paso 1: Ingesta de Noticias (Poblar la BD)
Este script descarga las noticias, las rompe en fragmentos, genera sus embeddings y los guarda en Docker.
```bash
.\venv\Scripts\python src/processes/ingesta_rss.py
```

### Paso 2: Iniciar el Servidor (API)
Lanza el servidor backend que procesará las preguntas.
```bash
.\venv\Scripts\python src/api/main.py
```

### Paso 3: Probar el RAG (Búsqueda)
Una vez el servidor diga `Application startup complete`, abre tu navegador en:
🔗 **[http://localhost:8088/docs](http://localhost:8088/docs)**

1. Haz clic en `POST /search/` -> **Try it out**.
2. Escribe una pregunta en el JSON, por ejemplo sobre noticias de Madrid o tecnología.
3. El sistema buscará en la base de datos local y te responderá usando el contexto real.

---

## Ejemplo de Consulta (CURL)
```bash
curl -X 'POST' \
  'http://localhost:8088/search/' \
  -H 'Content-Type: application/json' \
  -d '{"query": "¿Qué está pasando con los cantones de basura en Madrid?"}'
```
