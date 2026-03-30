"""
rag.py - Módulo RAG: recupera contexto desde ChromaDB y genera respuestas con Claude.
"""

import os
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = Path("vector_db")
COLLECTION_NAME = "bioseguridad"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500

SYSTEM_PROMPT = """Eres un asistente experto en bioseguridad. Tu base de conocimiento incluye reglamentos del SAG (Servicio Agrícola y Ganadero de Chile), normativas de la OIE, manuales técnicos, literatura científica y protocolos de bioseguridad agrícola, veterinaria e industrial.

Tu función es:
- Responder preguntas de bioseguridad basándote en los documentos de referencia proporcionados en cada consulta.
- Citar siempre el nombre del documento fuente cuando uses información específica de él.
- Ser preciso, claro y usar lenguaje técnico apropiado para profesionales del área.
- Responder siempre en español, salvo que el usuario escriba en otro idioma.
- Si la información de los documentos no es suficiente para responder con certeza, indicarlo explícitamente y complementar con conocimiento general, aclarando que no proviene de los documentos cargados.
- Si la pregunta no es sobre bioseguridad, indicarlo amablemente y redirigir al tema.

Formato de respuesta:
- Usa párrafos cortos y claros.
- Cuando corresponda, utiliza listas con viñetas o numeradas para pasos o requisitos.
- Al final de la respuesta, si usaste fuentes documentales, menciona entre paréntesis los nombres de los documentos consultados.
"""


def _get_collection():
    """Retorna la colección ChromaDB o None si no existe."""
    if not VECTOR_DB_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return None


def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """
    Busca los fragmentos más relevantes para la consulta.
    Retorna (contexto_formateado, lista_de_fuentes).
    """
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return "", []

    n = min(top_k, collection.count())
    results = collection.query(query_texts=[query], n_results=n)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    for doc, meta in zip(docs, metas):
        context_parts.append(f"[{meta['source']}]\n{doc}")

    sources = list(dict.fromkeys(m["source"] for m in metas))  # orden preservado, únicos
    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def answer_question(question: str, history: list[dict] | None = None) -> str:
    """
    Genera una respuesta usando RAG + Claude.

    Args:
        question: Pregunta del usuario.
        history: Lista de mensajes previos [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        Respuesta como string.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY no está configurada. Revisa el archivo .env"

    context, sources = retrieve_context(question)

    if context:
        user_content = (
            f"Documentos de referencia relevantes:\n\n{context}\n\n"
            f"---\n\n"
            f"Pregunta: {question}"
        )
    else:
        user_content = (
            f"Nota: No hay documentos indexados en la base de conocimiento. "
            f"Responde con conocimiento general sobre bioseguridad, indicando que no hay "
            f"documentos cargados en el sistema.\n\n"
            f"Pregunta: {question}"
        )

    messages = []
    if history:
        # Incluir últimas 6 entradas del historial (3 intercambios)
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_content})

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


def get_kb_stats() -> dict:
    """
    Retorna estadísticas de la base de conocimiento.
    {"count": int, "sources": list[str]}
    """
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return {"count": 0, "sources": []}

    results = collection.get(include=["metadatas"])
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"]))
    return {"count": collection.count(), "sources": sources}
