"""
agent.py

Define y expone el agente clínico conversacional.
Combina LangChain, memoria nativa, structured output
y trazabilidad con LangSmith.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

sys.path.append(str(Path(__file__).resolve().parents[2]))

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """
Eres un asistente médico especializado con acceso a
documentación clínica y registros de pacientes.

Tienes dos herramientas disponibles:
- search_clinical_docs: busca en guías clínicas y
  documentación médica indexada.
- search_patients: consulta registros de pacientes
  en el sistema legado.

Reglas:
1. Responde ÚNICAMENTE con información de tus herramientas.
2. Si no encuentras información suficiente, indícalo
   explícitamente con confidence_level 'baja'.
3. Siempre cita las fuentes documentales usadas.
4. Sugiere una pregunta de seguimiento relevante.
5. Nunca inventes información clínica.
"""


def get_llm():
    """
    Retorna el LLM configurado según LLM_PROVIDER.

    Returns:
        Instancia del LLM lista para usar.

    Raises:
        ValueError: Si el proveedor no es soportado.
    """
    if LLM_PROVIDER == "openai":
        logger.info(f"Using OpenAI LLM: {OPENAI_MODEL}")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
        )
    elif LLM_PROVIDER == "ollama":
        logger.info(f"Using Ollama LLM: {OLLAMA_MODEL}")
        return ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def build_agent():
    """
    Construye el agente clínico con memoria, tools y
    structured output.

    Returns:
        Instancia del agente lista para invocar.
    """
    from src.agent.schemas import ClinicalResponse
    from src.agent.tools import (
        search_clinical_docs,
        search_patients,
    )

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=[search_clinical_docs, search_patients],
        checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT,
        response_format=ClinicalResponse,
    )

    logger.info("Clinical agent built successfully")
    return agent


# Instancia global — se crea una sola vez
clinical_agent = build_agent()
