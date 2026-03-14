"""
chunker.py

Carga documentos PDF y los divide en chunks con overlap
para alimentar el pipeline RAG.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs/sample_docs")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Carga un PDF y retorna sus páginas como documentos.

    Args:
        pdf_path: Ruta al archivo PDF.

    Returns:
        Lista de documentos LangChain, uno por página.

    Raises:
        FileNotFoundError: Si el PDF no existe.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")
    return docs


def clean_text(text: str) -> str:
    """
    Limpia el texto de un chunk eliminando espacios
    excesivos, líneas vacías y caracteres no imprimibles.

    Args:
        text: Texto crudo del chunk.

    Returns:
        Texto limpio.
    """
    import re

    # Colapsa espacios múltiples en uno
    text = re.sub(r" {2,}", " ", text)
    # Colapsa saltos de línea múltiples en uno
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Elimina líneas que solo tienen espacios o símbolos solos
    lines = [line for line in text.splitlines() if len(line.strip()) > 10]
    return "\n".join(lines).strip()


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Divide documentos en chunks con overlap usando
    RecursiveCharacterTextSplitter y limpia el texto
    de cada chunk resultante.

    Args:
        docs: Lista de documentos LangChain.

    Returns:
        Lista de chunks limpios listos para embedir.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Limpiar y filtrar chunks con poco contenido útil
    clean_chunks = []
    for chunk in chunks:
        cleaned = clean_text(chunk.page_content)
        if len(cleaned) > 50:
            chunk.page_content = cleaned
            clean_chunks.append(chunk)

    logger.info(
        f"After cleaning: {len(clean_chunks)} chunks were generated"
        f"(removed {len(chunks) - len(clean_chunks)})"
    )
    return clean_chunks


def load_all_pdfs() -> list[Document]:
    """
    Carga y divide todos los PDFs en docs/sample_docs/.

    Returns:
        Lista combinada de chunks de todos los PDFs.

    Raises:
        ValueError: Si no hay PDFs en el directorio.
    """
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in {DOCS_DIR}")

    logger.info(f"Found {len(pdf_files)} PDFs")

    all_chunks = []
    for pdf_path in pdf_files:
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_all_pdfs()
    print(f"Total chunks generados: {len(chunks)}")
    print(f"\nEjemplo chunk 1:\n{chunks[0].page_content[:200]}")
