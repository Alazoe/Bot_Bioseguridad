"""
ingest.py - Indexa documentos PDF en la base de conocimiento vectorial.

Uso:
    python ingest.py                    # Indexa todos los PDFs en documents/
    python ingest.py ruta/a/carpeta     # Indexa PDFs desde una carpeta específica
    python ingest.py --limpiar          # Elimina todos los documentos indexados
    python ingest.py --listar           # Lista los documentos actualmente indexados
"""

import sys
import re
import json
import pickle
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from rank_bm25 import BM25Okapi

DOCUMENTS_DIR = Path("documents")
VECTOR_DB_DIR = Path("vector_db")
CHUNKS_FILE = VECTOR_DB_DIR / "chunks.json"
INDEX_FILE = VECTOR_DB_DIR / "bm25_index.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def tokenize(text: str) -> list[str]:
    """Tokeniza texto en palabras (minúsculas), soporta español."""
    return re.findall(r'\b\w+\b', text.lower())


def chunk_text(text: str) -> list[str]:
    """Divide texto en fragmentos con overlap."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_text_from_pdf(filepath: Path) -> str:
    """Extrae texto de un PDF página por página con PyMuPDF."""
    doc = fitz.open(filepath)
    pages_text = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text and text.strip():
            pages_text.append(f"[Página {i + 1}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages_text)


def load_chunks() -> list[dict]:
    """Carga fragmentos existentes desde disco."""
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_and_rebuild_index(all_chunks: list[dict]):
    """Guarda fragmentos y reconstruye el índice BM25."""
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    corpus = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(corpus)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)


def ingest_documents(documents_dir: Path = DOCUMENTS_DIR):
    """Indexa todos los PDFs de la carpeta indicada."""
    documents_dir.mkdir(exist_ok=True)
    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en '{documents_dir}/'")
        print("Coloca tus documentos PDF en esa carpeta y vuelve a ejecutar.")
        return

    print(f"Encontrados {len(pdf_files)} PDF(s) en '{documents_dir}/'")

    existing_chunks = load_chunks()
    existing_sources = {c["source"] for c in existing_chunks}
    all_chunks = list(existing_chunks)
    total_new = 0

    for pdf_path in pdf_files:
        if pdf_path.name in existing_sources:
            print(f"  Ya indexado: {pdf_path.name} (omitido)")
            continue

        print(f"  Procesando: {pdf_path.name} ...", end="", flush=True)
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(" (sin texto extraíble, omitido)")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{pdf_path.stem}_chunk_{i}",
                "text": chunk,
                "source": pdf_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        total_new += len(chunks)
        print(f" {len(chunks)} fragmentos indexados.")

    if total_new > 0:
        print("\nReconstruyendo índice BM25...", end="", flush=True)
        save_and_rebuild_index(all_chunks)
        print(" listo.")

    print(f"\nIndexación completa.")
    print(f"  Fragmentos nuevos:              {total_new}")
    print(f"  Total en base de conocimiento:  {len(all_chunks)}")


def list_documents():
    """Lista los documentos actualmente indexados."""
    chunks = load_chunks()
    if not chunks:
        print("La base de conocimiento está vacía.")
        return

    sources: dict[str, int] = {}
    for c in chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1

    print(f"Documentos indexados ({len(chunks)} fragmentos totales):")
    for src, count in sorted(sources.items()):
        print(f"  - {src}: {count} fragmentos")


def clear_collection():
    """Elimina todos los documentos indexados."""
    removed = 0
    for path in [CHUNKS_FILE, INDEX_FILE]:
        if path.exists():
            path.unlink()
            removed += 1
    if removed:
        print("Base de conocimiento limpiada.")
    else:
        print("La base de conocimiento ya estaba vacía.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--limpiar" in args:
        clear_collection()
    elif "--listar" in args:
        list_documents()
    else:
        folder = Path(args[0]) if args else DOCUMENTS_DIR
        ingest_documents(folder)
