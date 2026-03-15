"""
pipeline.py

Orquesta el pipeline RAG completo. Expone una interfaz
única para que componentes externos puedan hacer
consultas sin conocer los detalles internos del RAG.
"""

import logging
from src.config import get_llm
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Contrato de respuesta del pipeline RAG.

    Attributes:
        answer: Respuesta generada por el LLM.
        sources: Lista de chunks usados como contexto.
    """

    answer: str
    sources: list[Document]


def build_prompt(
    query: str,
    context_chunks: list[Document],
    chat_history: list[dict] | None = None,
) -> str:
    """
    Construye el prompt para el LLM combinando el
    contexto recuperado y la pregunta del usuario.

    Args:
        query: Pregunta del usuario.
        context_chunks: Chunks recuperados por el retriever.
        chat_history: Historial de conversacion previo.

    Returns:
        Prompt completo listo para enviar al LLM.
    """
    context = "\n\n".join(
        [
            f"[Fuente: {doc.metadata.get('source', 'N/A')}, "
            f"Página: {doc.metadata.get('page', 'N/A')}]\n"
            f"{doc.page_content}"
            for doc in context_chunks
        ]
    )

    history_text = ""
    if chat_history:
        history_text = "\n\nHistorial de conversación:\n"
        for msg in chat_history[-4:]:
            role = "Usuario" if msg["role"] == "user" else "Asistente"
            history_text += f"{role}: {msg['content']}\n"

    return f"""Eres un asistente médico especializado.
Responde de forma directa y clara, sin frases introductorias
como "según el contexto" o "la respuesta es".
Usa ÚNICAMENTE la información del contexto proporcionado.
Si la información no está en el contexto, di explícitamente
que no tienes suficiente información para responder.
{history_text}
Contexto documental:
{context}

Pregunta: {query}

Respuesta:"""


def ask(
    query: str,
    chat_history: list[dict] | None = None,
) -> RAGResponse:
    """
    Ejecuta el pipeline RAG completo para una consulta.

    Carga el vectorstore, ejecuta búsqueda híbrida,
    construye el prompt y genera la respuesta.

    Args:
        query: Pregunta del usuario en lenguaje natural.

    Returns:
        RAGResponse con la respuesta y las fuentes usadas.
    """
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.rag.chunker import load_all_documents
    from src.rag.embedder import load_vectorstore
    from src.rag.retriever import hybrid_search

    vectorstore = load_vectorstore()
    all_docs = load_all_documents()

    chunks = hybrid_search(vectorstore, all_docs, query)
    prompt = build_prompt(query, chunks)

    llm = get_llm()
    response = llm.invoke(prompt)

    return RAGResponse(
        answer=response.content,
        sources=chunks,
    )


if __name__ == "__main__":
    query = "¿Cuál es el tratamiento recomendado para la hipertensión?"
    print(f"Query: {query}\n")

    result = ask(query)

    print(f"Respuesta:\n{result.answer}\n")
    print("Fuentes:")
    for doc in result.sources:
        print(
            f"  - {doc.metadata.get('source', 'N/A')} "
            f"p.{doc.metadata.get('page', 'N/A')}"
        )
