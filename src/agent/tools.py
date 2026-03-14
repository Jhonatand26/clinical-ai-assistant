"""
tools.py

Define las tools disponibles para el agente clínico.
Cada tool encapsula una capacidad específica — el agente
decide cuándo invocar cada una.
"""

import logging
import sys
from pathlib import Path

from langchain.tools import tool

sys.path.append(str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def search_clinical_docs(query: str) -> str:
    """
    Busca información relevante en la documentación
    clínica indexada usando búsqueda híbrida semántica
    y por palabras clave (BM25 + ChromaDB).

    Úsala cuando el usuario pregunte sobre enfermedades,
    tratamientos, medicamentos o protocolos clínicos.

    Args:
        query: Pregunta o término clínico a buscar.

    Returns:
        Fragmentos relevantes con sus fuentes documentales.
    """
    from src.rag.chunker import load_all_pdfs
    from src.rag.embedder import load_vectorstore
    from src.rag.retriever import hybrid_search

    vectorstore = load_vectorstore()
    all_chunks = load_all_pdfs()
    chunks = hybrid_search(vectorstore, all_chunks, query)

    if not chunks:
        return "No se encontró información relevante."

    results = []
    for doc in chunks:
        source = Path(doc.metadata.get("source", "N/A")).name
        page = doc.metadata.get("page", "N/A")
        results.append(f"[{source} — Página {page}]\n" f"{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def search_patients(query: str) -> str:
    """
    Consulta la base de datos de pacientes del sistema
    legado clínico. Úsala cuando el usuario pregunte
    sobre pacientes específicos, diagnósticos registrados
    o medicamentos activos en el sistema.

    Args:
        query: Término de búsqueda — nombre, diagnóstico
               o medicamento.

    Returns:
        Registros de pacientes que coinciden con la búsqueda.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path("data/raw/legacy_clinic.db")

    if not db_path.exists():
        return "Base de datos de pacientes no disponible."

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name, age, diagnosis, medication, last_visit
        FROM patients
        WHERE name LIKE ?
           OR diagnosis LIKE ?
           OR medication LIKE ?
    """,
        (f"%{query}%", f"%{query}%", f"%{query}%"),
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"No se encontraron registros para: {query}"

    results = []
    for row in rows:
        name, age, diagnosis, medication, last_visit = row
        results.append(
            f"Paciente: {name} ({age} años)\n"
            f"Diagnóstico: {diagnosis}\n"
            f"Medicamento: {medication}\n"
            f"Última visita: {last_visit}"
        )

    return "\n\n---\n\n".join(results)
