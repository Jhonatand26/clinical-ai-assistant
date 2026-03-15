"""
app.py

Interfaz conversacional con Streamlit para el agente
clínico. Usa el agente LangChain con memoria nativa,
structured output y trazabilidad LangSmith.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import streamlit as st
from langchain.messages import HumanMessage

from src.agent.agent import clinical_agent
from src.agent.schemas import ClinicalResponse

# ── Configuración de página ──────────────────────────
st.set_page_config(
    page_title="Asistente Clínico IA",
    page_icon="🏥",
    layout="centered",
)

# ── Estilos ──────────────────────────────────────────
st.markdown(
    """
    <style>
    .source-card {
        background-color: #f0f4f8;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #1e3a5f;
    }
    .confidence-alta {
        color: #16a34a;
        font-weight: bold;
    }
    .confidence-media {
        color: #d97706;
        font-weight: bold;
    }
    .confidence-baja {
        color: #dc2626;
        font-weight: bold;
    }
    .followup {
        background-color: #f8fafc;
        border: 1px dashed #94a3b8;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────
st.title("🏥 Asistente Clínico IA")
st.caption("Powered by LangChain · Trazabilidad con LangSmith")
st.divider()

# ── Session state ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid

    st.session_state.thread_id = str(uuid.uuid4())

if "responses" not in st.session_state:
    st.session_state.responses = []


def parse_response(content: str) -> ClinicalResponse | None:
    """
    Intenta parsear el contenido del agente como
    ClinicalResponse. Retorna None si no es posible.

    Args:
        content: Contenido del mensaje del agente.

    Returns:
        ClinicalResponse parseado o None.
    """
    try:
        # Intentar extraer JSON del contenido
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            data = json.loads(json_str)
            return ClinicalResponse(**data)
    except Exception:
        pass
    return None


def render_structured_response(
    structured: ClinicalResponse,
) -> None:
    """
    Renderiza una ClinicalResponse en Streamlit con
    formato visual enriquecido.

    Args:
        structured: Respuesta estructurada del agente.
    """
    st.markdown(structured.answer)

    # Confianza
    level = structured.confidence_level.lower()
    emoji = {"alta": "🟢", "media": "🟡", "baja": "🔴"}.get(level, "⚪")
    st.markdown(
        f'<span class="confidence-{level}">'
        f"{emoji} Confianza: {structured.confidence_level}"
        f"</span>",
        unsafe_allow_html=True,
    )

    # Fuentes
    if structured.sources:
        with st.expander("📄 Ver fuentes"):
            for source in structured.sources:
                st.markdown(
                    f'<div class="source-card">' f"📎 {source}" f"</div>",
                    unsafe_allow_html=True,
                )

    # Pregunta sugerida
    if structured.suggested_followup:
        st.markdown(
            f'<div class="followup">'
            f"💡 <strong>Pregunta sugerida:</strong> "
            f"{structured.suggested_followup}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Historial ────────────────────────────────────────
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            resp_index = sum(
                1 for m in st.session_state.messages[:i] if m["role"] == "assistant"
            )
            if resp_index < len(st.session_state.responses):
                stored = st.session_state.responses[resp_index]
                if isinstance(stored, ClinicalResponse):
                    render_structured_response(stored)
                else:
                    st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# ── Input ────────────────────────────────────────────
if query := st.chat_input("Haz una pregunta sobre la documentación clínica..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consultando documentación..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                response = clinical_agent.invoke(
                    {"messages": [HumanMessage(content=query)]},
                    config,
                )

                content = response["messages"][-1].content

                # Intentar structured, fallback a texto
                structured = parse_response(content)

                if structured:
                    render_structured_response(structured)
                    st.session_state.responses.append(structured)
                else:
                    st.markdown(content)
                    st.session_state.responses.append(content)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.responses.append(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )
