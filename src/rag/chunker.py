"""
chunker.py

Carga documentos de MÚLTIPLES fuentes (PDF, DB, Web) y los
prepara para el pipeline RAG. Los PDFs se dividen en chunks.
Los datos de DB/Web se convierten en documentos autocontenidos.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -- Setup --
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar los nuevos loaders
from src.extraction.legacy_loader import LegacyLoader
from src.extraction.web_scraper import WebScraper

DOCS_DIR = Path("docs/sample_docs")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


# --- Helper Functions (unchanged or slightly modified) ---

def load_pdf(pdf_path: Path) -> list[Document]:
    """Carga un PDF y retorna sus páginas como documentos."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from {pdf_path.name}")
    return docs

def clean_text(text: str) -> str:
    """Limpia el texto de un chunk eliminando espacios/líneas excesivas."""
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line for line in text.splitlines() if len(line.strip()) > 10]
    return "\n".join(lines).strip()

def split_documents(docs: list[Document]) -> list[Document]:
    """Divide documentos largos (de PDFs) en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    clean_chunks = []
    for chunk in chunks:
        cleaned = clean_text(chunk.page_content)
        if len(cleaned) > 50:
            chunk.page_content = cleaned
            clean_chunks.append(chunk)
    logger.info(
        f"Split into {len(clean_chunks)} chunks from {len(docs)} pages "
        f"(removed {len(chunks) - len(clean_chunks)} small chunks)."
    )
    return clean_chunks


# --- NEW: Data Integration Functions ---

def _format_dict_content(data: Dict) -> str:
    """Convierte un diccionario en un string de texto formateado."""
    # Excluir claves que no aportan valor como contenido
    excluded_keys = ["id", "patient_id"]
    return "\n".join(f"- {key.replace('_', ' ').capitalize()}: {value}"
                     for key, value in data.items() if key not in excluded_keys and value)

def _dicts_to_documents(
    data_list: List[Dict], source_name: str, content_key: str = None
) -> List[Document]:
    """
    Convierte una lista de diccionarios en una lista de Documentos LangChain.
    Cada diccionario es un documento.
    """
    documents = []
    for item in data_list:
        # Usar una clave específica como 'name' o 'medication' para el título si es posible
        title = item.get(content_key, "Registro") if content_key else "Registro"
        page_content = f"Fuente: {source_name}\n"
        page_content += f"Tipo de Registro: {title}\n"
        page_content += _format_dict_content(item)

        metadata = item.copy()
        metadata["source"] = source_name
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    logger.info(f"Converted {len(documents)} records from '{source_name}' to Documents.")
    return documents


# --- Main Orchestrator Function ---

def load_all_documents() -> list[Document]:
    """
    Carga y procesa documentos de TODAS las fuentes configuradas (PDF, DB, Web).

    - PDFs: Se cargan y se dividen en chunks.
    - DB/Web: Cada registro se convierte en un Documento autocontenido.

    Returns:
        Una lista combinada de todos los documentos y chunks.
    """
    all_docs = []

    # 1. Cargar y procesar PDFs
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {DOCS_DIR}")
    else:
        logger.info(f"Found {len(pdf_files)} PDFs to process.")
        for pdf_path in pdf_files:
            pdf_docs = load_pdf(pdf_path)
            pdf_chunks = split_documents(pdf_docs)
            all_docs.extend(pdf_chunks)

    # 2. Cargar datos de la base de datos legado
    try:
        legacy_loader = LegacyLoader()
        patient_data = legacy_loader.load()
        patient_docs = _dicts_to_documents(patient_data, "Pacientes Legado", content_key="name")
        all_docs.extend(patient_docs)
    except Exception as e:
        logger.error(f"Failed to load data from Legacy DB: {e}")

    # 3. Cargar datos desde el web scraper
    try:
        web_scraper = WebScraper()
        medicine_data = web_scraper.load()
        medicine_docs = _dicts_to_documents(medicine_data, "Medicamentos OMS", content_key="medicine")
        all_docs.extend(medicine_docs)
    except Exception as e:
        logger.error(f"Failed to load data from Web Scraper: {e}")

    logger.info(f"Total documents/chunks from all sources: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    final_documents = load_all_documents()
    print(f"\nTotal de documentos y chunks generados: {len(final_documents)}")
    
    # Imprimir un ejemplo de cada tipo de fuente
    pdf_chunk_example = next((doc for doc in final_documents if doc.metadata.get('page') is not None), None)
    patient_example = next((doc for doc in final_documents if doc.metadata.get('source') == "Pacientes Legado"), None)
    medicine_example = next((doc for doc in final_documents if doc.metadata.get('source') == "Medicamentos OMS"), None)

    if pdf_chunk_example:
        print("\n--- Ejemplo de Chunk de PDF ---")
        print(f"Metadata: {pdf_chunk_example.metadata}")
        print(f"Content: {pdf_chunk_example.page_content[:200]}...")

    if patient_example:
        print("\n--- Ejemplo de Documento de Paciente ---")
        print(f"Metadata: {patient_example.metadata}")
        print(f"Content:\n{patient_example.page_content}")

    if medicine_example:
        print("\n--- Ejemplo de Documento de Medicamento ---")
        print(f"Metadata: {medicine_example.metadata}")
        print(f"Content:\n{medicine_example.page_content}")
