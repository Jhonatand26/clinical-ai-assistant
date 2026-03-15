"""
embedder.py

Convierte chunks de texto en embeddings y los persiste
en ChromaDB. Soporta OpenAI y Ollama como proveedores
según la variable de entorno LLM_PROVIDER.
"""

import logging
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
COLLECTION_NAME = "clinical_docs"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


def get_embedding_function():
    """
    Retorna la función de embeddings según el proveedor
    configurado en LLM_PROVIDER.

    Returns:
        Instancia del modelo de embeddings.

    Raises:
        ValueError: Si el proveedor no es soportado.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        logger.info(f"Using OpenAI embeddings: {model}")
        return OpenAIEmbeddings(model=model)

    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info(f"Using Ollama embeddings: {model}")
        return OllamaEmbeddings(model=model)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. " "Use 'openai' or 'ollama'."
        )


def build_vectorstore(
    chunks: list[Document],
) -> Chroma:
    """
    Crea o actualiza el vectorstore ChromaDB con los
    chunks proporcionados.

    Args:
        chunks: Lista de chunks con contenido y metadatos.

    Returns:
        Instancia de Chroma lista para consultas.
    """
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    embedding_fn = get_embedding_function()

    logger.info(f"Building vectorstore with {len(chunks)} chunks...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    logger.info(f"Vectorstore persisted at {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Carga un vectorstore ChromaDB existente desde disco.

    Returns:
        Instancia de Chroma lista para consultas.

    Raises:
        FileNotFoundError: Si el vectorstore no existe.
    """
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at {CHROMA_PERSIST_DIR}. "
            "Run build_vectorstore() first."
        )

    embedding_fn = get_embedding_function()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    logger.info(f"Vectorstore loaded from {CHROMA_PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.rag.chunker import load_all_documents

    docs = load_all_documents()
    vectorstore = build_vectorstore(docs)
    print(f"Vectorstore listo con {vectorstore._collection.count()} documentos/chunks")
