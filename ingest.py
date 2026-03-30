"""
ingest.py - Indexa documentos PDF en la base de conocimiento vectorial.

Uso:
    python ingest.py                    # Indexa todos los PDFs en documents/
    python ingest.py ruta/a/carpeta     # Indexa PDFs desde una carpeta específica
    python ingest.py --limpiar          # Elimina todos los documentos indexados
    python ingest.py --listar           # Lista los documentos actualmente indexados
"""

import sys
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

DOCUMENTS_DIR = Path("documents")
VECTOR_DB_DIR = Path("vector_db")
COLLECTION_NAME = "bioseguridad"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_collection(create: bool = True):
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    ef = get_embedding_function()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        return None


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
    """Extrae texto de un PDF página por página."""
    doc = fitz.open(filepath)
    pages_text = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text and text.strip():
            pages_text.append(f"[Página {i + 1}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages_text)


def ingest_file(filepath: Path, collection) -> int:
    """Indexa un único PDF. Retorna el número de fragmentos añadidos."""
    print(f"  Procesando: {filepath.name} ...", end="", flush=True)

    text = extract_text_from_pdf(filepath)
    if not text.strip():
        print(" (sin texto extraíble, omitido)")
        return 0

    chunks = chunk_text(text)
    ids = [f"{filepath.stem}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": filepath.name, "chunk_index": i, "total_chunks": len(chunks)}
        for i in range(len(chunks))
    ]

    # Upsert en lotes para evitar errores de memoria
    batch_size = 50
    added = 0
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=chunks[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        added += len(chunks[i : i + batch_size])

    print(f" {len(chunks)} fragmentos indexados.")
    return len(chunks)


def ingest_documents(documents_dir: Path = DOCUMENTS_DIR):
    """Indexa todos los PDFs de la carpeta indicada."""
    documents_dir.mkdir(exist_ok=True)
    pdf_files = sorted(documents_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en '{documents_dir}/'")
        print("Coloca tus documentos PDF en esa carpeta y vuelve a ejecutar.")
        return

    print(f"Encontrados {len(pdf_files)} PDF(s) en '{documents_dir}/'")
    print("Cargando modelo de embeddings (primera vez puede tardar)...\n")

    collection = get_collection(create=True)
    total_chunks = 0

    for pdf_path in pdf_files:
        total_chunks += ingest_file(pdf_path, collection)

    print(f"\nIndexación completa.")
    print(f"  Fragmentos añadidos/actualizados: {total_chunks}")
    print(f"  Total en base de conocimiento:    {collection.count()}")


def list_documents():
    """Lista los documentos actualmente indexados."""
    collection = get_collection(create=False)
    if collection is None or collection.count() == 0:
        print("La base de conocimiento está vacía.")
        return

    results = collection.get(include=["metadatas"])
    sources = {}
    for meta in results["metadatas"]:
        src = meta["source"]
        sources[src] = sources.get(src, 0) + 1

    print(f"Documentos indexados ({collection.count()} fragmentos totales):")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count} fragmentos")


def clear_collection():
    """Elimina todos los documentos indexados."""
    collection = get_collection(create=False)
    if collection is None or collection.count() == 0:
        print("La base de conocimiento ya está vacía.")
        return

    count = collection.count()
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Base de conocimiento limpiada ({count} fragmentos eliminados).")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--limpiar" in args:
        clear_collection()
    elif "--listar" in args:
        list_documents()
    else:
        folder = Path(args[0]) if args else DOCUMENTS_DIR
        ingest_documents(folder)
