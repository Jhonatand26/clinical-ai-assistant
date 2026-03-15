"""
schemas.py

Contratos de entrada y salida del agente clínico.
Define la estructura de respuesta usando Pydantic.
"""

from pydantic import BaseModel, Field


class ClinicalResponse(BaseModel):
    """
    Respuesta estructurada del agente clínico.

    Attributes:
        answer: Respuesta generada por el agente.
        confidence_level: Nivel de confianza estimado.
        sources: Fragmentos documentales usados.
        suggested_followup: Pregunta de seguimiento sugerida.
    """

    answer: str = Field(description="Respuesta clara y directa a la pregunta clínica.")
    confidence_level: str = Field(
        description=("Nivel de confianza en la respuesta: " "'alta', 'media' o 'baja'.")
    )
    sources: list[str] = Field(
        description=(
            "Lista de fuentes documentales usadas. "
            "Formato: 'nombre_archivo.pdf — Página N'."
        )
    )
    suggested_followup: str = Field(
        description=(
            "Una pregunta de seguimiento relevante "
            "que el usuario podría querer hacer."
        )
    )
