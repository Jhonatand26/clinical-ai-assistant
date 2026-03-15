"""
retriever.py

Implementa búsqueda híbrida combinando ChromaDB
(semántica) y BM25 (keyword) con Reciprocal Rank
Fusion para mejorar la calidad del retrieval.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))


def get_semantic_results(
    vectorstore,
    query: str,
    k: int,
) -> list[Document]:
    """
    Recupera los k chunks más similares semánticamente
    a la consulta usando ChromaDB.

    Args:
        vectorstore: Instancia de Chroma cargada.
        query: Pregunta del usuario.
        k: Número de chunks a recuperar.

    Returns:
        Lista de Documents ordenados por similitud.
    """
    results = vectorstore.similarity_search(query, k=k)
    logger.info(f"Semantic search returned {len(results)} chunks")
    return results


def get_bm25_results(
    chunks: list[Document],
    query: str,
    k: int,
) -> list[Document]:
    """
    Recupera los k chunks más relevantes usando BM25
    (búsqueda por palabras clave).

    Args:
        chunks: Todos los chunks del corpus.
        query: Pregunta del usuario.
        k: Número de chunks a recuperar.

    Returns:
        Lista de Documents ordenados por score BM25.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Obtener índices de los top-k scores
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = [chunks[i] for i in top_indices]
    logger.info(f"BM25 search returned {len(results)} chunks")
    return results


def reciprocal_rank_fusion(
    semantic_results: list[Document],
    bm25_results: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Fusiona dos listas de resultados usando Reciprocal
    Rank Fusion (RRF).

    RRF asigna score = 1/(k + rank) a cada documento
    en cada lista y suma los scores. Documentos que
    aparecen en ambas listas reciben score más alto.

    Args:
        semantic_results: Chunks del retrieval semántico.
        bm25_results: Chunks del retrieval BM25.
        k: Constante de suavizado RRF (default 60).

    Returns:
        Lista de Documents ordenados por score RRF fusionado.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True,
    )

    fused = [doc_map[key] for key in sorted_keys]
    logger.info(f"RRF fusion returned {len(fused)} chunks")
    return fused


def hybrid_search(
    vectorstore,
    all_chunks: list[Document],
    query: str,
    top_k: int = TOP_K,
) -> list[Document]:
    """
    Ejecuta búsqueda híbrida combinando semántica y BM25
    con fusión RRF.

    Args:
        vectorstore: Instancia de Chroma cargada.
        all_chunks: Todos los chunks del corpus para BM25.
        query: Pregunta del usuario.
        top_k: Número final de chunks a retornar.

    Returns:
        Lista de los top_k chunks más relevantes.
    """
    semantic = get_semantic_results(vectorstore, query, k=top_k)
    bm25 = get_bm25_results(all_chunks, query, k=top_k)
    fused = reciprocal_rank_fusion(semantic, bm25)
    return fused[:top_k]


if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    from src.rag.chunker import load_all_pdfs
    from src.rag.embedder import load_vectorstore

    query = "¿Cuál es el tratamiento para la hipertensión?"

    print(f"Query: {query}\n")

    vectorstore = load_vectorstore()
    all_chunks = load_all_pdfs()
    results = hybrid_search(vectorstore, all_chunks, query)

    print(f"\nTop {len(results)} chunks recuperados:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"Fuente: {doc.metadata.get('source', 'N/A')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(f"Contenido: {doc.page_content[:150]}\n")
