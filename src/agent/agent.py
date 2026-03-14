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
Eres un asistente médico especializado. Tu única función es
responder a las preguntas del usuario sobre temas clínicos.

Para responder, tienes una única y potente herramienta:
- answer_clinical_question: Esta herramienta utiliza un sistema
  de búsqueda avanzado (RAG) que tiene acceso a guías clínicas,
  protocolos, una base de datos de pacientes y listados de
  medicamentos.

Reglas de oro:
1. Para CUALQUIER pregunta del usuario, invoca SIEMPRE la
   herramienta `answer_clinical_question`.
2. Pasa la pregunta del usuario a la herramienta de la forma
   más directa y completa posible.
3. La respuesta de la herramienta ya está formateada para el
   usuario. Tu trabajo es simplemente devolver esa respuesta
   directamente.
4. El `response_format` que se te ha asignado es estricto.
   La respuesta de la herramienta `answer_clinical_question`
   contiene la información que necesitas. Debes usar esa
   información para rellenar los campos de `ClinicalResponse`.
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


def build_agent(llm, tools: list):
    """
    Construye un agente genérico dadas sus dependencias (Inyección de Dependencias).

    Args:
        llm: El modelo de lenguaje a utilizar.
        tools: Una lista de herramientas para el agente.

    Returns:
        Instancia del agente lista para invocar.
    """
    from src.agent.schemas import ClinicalResponse

    agent = create_agent(
        model=llm,
        tools=tools, # Corrected: use the 'tools' parameter
        checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT,
        response_format=ClinicalResponse,
    )

    logger.info("Clinical agent built successfully")
    return agent


# --- Creación de la instancia del agente ---
# 1. Recolectar dependencias
from src.agent.tools import answer_clinical_question # <- Nueva herramienta

llm = get_llm()
clinical_tools = [answer_clinical_question] # <- Nueva lista de herramientas

# 2. Inyectar dependencias en el constructor
clinical_agent = build_agent(llm=llm, tools=clinical_tools)
