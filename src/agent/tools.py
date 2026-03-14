"""
tools.py

Define la tool unificada disponible para el agente clínico.
Esta herramienta es un wrapper sobre el pipeline RAG completo.
"""

import logging
import sys
from pathlib import Path

from langchain.tools import tool

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Importar el pipeline RAG principal
from src.rag.pipeline import ask as ask_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def answer_clinical_question(query: str) -> str:
    """
    Responde a cualquier pregunta sobre el ámbito clínico utilizando
    el sistema RAG unificado. Este sistema tiene acceso a guías
    clínicas, protocolos, registros de pacientes del sistema legado
    y listados de medicamentos.

    Úsala para CUALQUIER pregunta del usuario, ya sea sobre
    tratamientos, enfermedades, o para buscar información de un
    paciente específico.

    Args:
        query: La pregunta completa del usuario en lenguaje natural.

    Returns:
        Una respuesta completa que incluye la contestación directa
        y las fuentes de información utilizadas.
    """
    logger.info(f"Executing RAG pipeline for query: '{query}'")

    # Invocar el pipeline RAG centralizado
    try:
        rag_response = ask_rag_pipeline(query)
    except Exception as e:
        logger.error(f"Error executing RAG pipeline: {e}")
        return "Hubo un error al procesar la solicitud. Por favor, revisa los logs."


    if not rag_response or not rag_response.answer:
        return "No se encontró información relevante para responder a la pregunta."

    # Formatear la respuesta para el agente
    formatted_sources = []
    for doc in rag_response.sources:
        source = doc.metadata.get("source", "N/A")
        # Para los PDFs, mostrar la página. Para otros, el nombre.
        if 'page' in doc.metadata:
            source_ref = f"{Path(source).name} (Página {doc.metadata.get('page', 'N/A')})"
        elif 'name' in doc.metadata:
            source_ref = f"{source} (Paciente: {doc.metadata.get('name', 'N/A')})"
        elif 'medicine' in doc.metadata:
            source_ref = f"{source} (Medicamento: {doc.metadata.get('medicine', 'N/A')})"
        else:
            source_ref = source
        
        # Evitar duplicados en las fuentes mostradas
        if source_ref not in formatted_sources:
            formatted_sources.append(source_ref)
    
    formatted_answer = (
        f"Respuesta: {rag_response.answer}\n\n"
        f"Fuentes Consultadas: {'; '.join(formatted_sources)}"
    )
    
    return formatted_answer
