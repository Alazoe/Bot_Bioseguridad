"""
rag.py - Módulo RAG: recupera contexto con BM25 y genera respuestas con Claude.
"""

import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
import anthropic
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_DIR = Path("vector_db")
CHUNKS_FILE = VECTOR_DB_DIR / "chunks.json"
INDEX_FILE = VECTOR_DB_DIR / "bm25_index.pkl"
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


def tokenize(text: str) -> list[str]:
    """Tokeniza texto en palabras (minúsculas), soporta español."""
    return re.findall(r'\b\w+\b', text.lower())


def _load_store() -> tuple:
    """Carga el índice BM25 y los chunks desde disco. Retorna (bm25, chunks) o (None, [])."""
    if not CHUNKS_FILE.exists() or not INDEX_FILE.exists():
        return None, []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(INDEX_FILE, "rb") as f:
        bm25 = pickle.load(f)
    return bm25, chunks


def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """
    Busca los fragmentos más relevantes para la consulta usando BM25.
    Retorna (contexto_formateado, lista_de_fuentes).
    """
    bm25, chunks = _load_store()
    if bm25 is None or not chunks:
        return "", []

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_indices = [int(i) for i in top_indices if scores[i] > 0]

    if not top_indices:
        return "", []

    context_parts = []
    sources: list[str] = []
    for i in top_indices:
        chunk = chunks[i]
        context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        if chunk["source"] not in sources:
            sources.append(chunk["source"])

    return "\n\n---\n\n".join(context_parts), sources


def answer_question(question: str, history: list[dict] | None = None) -> str:
    """
    Genera una respuesta usando RAG (BM25) + Claude.

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
            f"Nota: No hay documentos indexados o no se encontraron fragmentos relevantes. "
            f"Responde con conocimiento general sobre bioseguridad, indicando que no hay "
            f"documentos cargados en el sistema.\n\n"
            f"Pregunta: {question}"
        )

    messages: list[dict] = []
    if history:
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
    """Retorna estadísticas de la base de conocimiento."""
    if not CHUNKS_FILE.exists():
        return {"count": 0, "sources": []}
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        return {"count": 0, "sources": []}
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    return {"count": len(chunks), "sources": sources}
