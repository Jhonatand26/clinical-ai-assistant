"""
config.py

Configuración centralizada del LLM. Única fuente de verdad
para la creación del modelo de lenguaje, evitando duplicación
entre el agente y el pipeline RAG.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano")


def get_llm():
    """
    Retorna el LLM configurado según LLM_PROVIDER.

    Returns:
        Instancia del LLM lista para usar.

    Raises:
        ValueError: Si el proveedor no es soportado.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        logger.info(f"Using OpenAI LLM: {OPENAI_MODEL}")
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        logger.info(f"Using Ollama LLM: {OLLAMA_MODEL}")
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
