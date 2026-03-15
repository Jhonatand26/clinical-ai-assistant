# 🏥 Clinical AI Assistant

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![UV](https://img.shields.io/badge/UV-package%20manager-DE5FE9?style=flat)
![LangChain](https://img.shields.io/badge/LangChain-1.2-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-FF6B35?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorstore-FF6B35?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=flat&logo=streamlit&logoColor=white)

**Prueba Técnica — AI Engineer**

Asistente conversacional clínico con pipeline de extracción multi-fuente, recuperación híbrida de documentos y agente LangGraph con memoria persistente de conversación.

---

## Descripción

Sistema de consulta documental clínica que integra tres fuentes de datos heterogéneas en un único pipeline RAG (Retrieval-Augmented Generation):

1. El usuario escribe una pregunta clínica en la interfaz (Streamlit)
2. El **agente LangGraph** decide invocar la herramienta de consulta RAG
3. El **pipeline RAG híbrido** recupera chunks relevantes combinando búsqueda semántica (ChromaDB) y léxica (BM25) mediante Reciprocal Rank Fusion
4. El LLM genera una respuesta estructurada con nivel de confianza, fuentes citadas y pregunta sugerida de seguimiento
5. El agente mantiene **memoria de la conversación** entre turnos (LangGraph `InMemorySaver`)

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (:8501)                   │
│           Chat conversacional · ClinicalResponse         │
└───────────────────────┬─────────────────────────────────┘
                        │ HumanMessage
                        ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph Agent (ReAct)                    │
│   LLM + InMemorySaver + Tool: answer_clinical_question   │
└───────────────────────┬─────────────────────────────────┘
                        │ tool call
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG Pipeline                            │
│                                                          │
│  ┌─────────────────┐    ┌──────────────────────────┐    │
│  │ ChromaDB         │    │ BM25 (rank-bm25)          │    │
│  │ Semantic Search  │    │ Keyword Search            │    │
│  │ top-k chunks     │    │ top-k chunks              │    │
│  └────────┬────────┘    └────────────┬─────────────┘    │
│           └──────────────────────────┘                   │
│                         │ Reciprocal Rank Fusion          │
│                         ▼                                │
│              top-k chunks fusionados                     │
│                         │                                │
│                         ▼                                │
│              LLM → RAGResponse(answer, sources)          │
└─────────────────────────────────────────────────────────┘

Fuentes de datos (ingesta):
  PDFs clínicos ──────────────────────────────┐
  SQLite legacy (legacy_clinic.db) ────────────┼──→ ChromaDB + BM25 corpus
  Web scraping (Wikipedia OMS) ───────────────┘
```

**Componentes principales:**

| Módulo | Responsabilidad |
|--------|----------------|
| `src/agent/agent.py` | Agente LangGraph con memoria y herramienta RAG |
| `src/agent/tools.py` | Tool `answer_clinical_question` — wrapper del pipeline RAG |
| `src/agent/schemas.py` | `ClinicalResponse` — contrato Pydantic de respuesta estructurada |
| `src/rag/pipeline.py` | Orquestador RAG con caché de módulo |
| `src/rag/retriever.py` | Búsqueda híbrida: BM25 + ChromaDB + RRF |
| `src/rag/chunker.py` | Carga y chunking multi-fuente |
| `src/rag/embedder.py` | Gestión de embeddings y vectorstore ChromaDB |
| `src/extraction/base.py` | Interfaz `BaseLoader` — contrato para fuentes de datos |
| `src/extraction/legacy_loader.py` | Fuente 1: sistema legado SQLite |
| `src/extraction/web_scraper.py` | Fuente 2: scraping Wikipedia (Medicamentos OMS) |
| `src/config.py` | Configuración centralizada del LLM |
| `src/ui/app.py` | Interfaz Streamlit con renderizado estructurado |

---

## Decisiones Técnicas

Esta sección documenta el razonamiento detrás de las elecciones de diseño más relevantes.

### 1. Recuperación Híbrida: BM25 + ChromaDB + Reciprocal Rank Fusion

La búsqueda semántica pura falla en consultas de entidades específicas. Si el usuario pregunta por *"Carlos Ruiz"* o *"Metformina 850mg"*, el embedding de esa cadena puede no encontrar el registro exacto porque la similitud coseno favorece contexto semántico sobre coincidencia léxica. BM25 resuelve esto: premia la frecuencia inversa de tokens en el corpus.

El problema de combinar dos rankings distintos es que sus scores no son comparables (BM25 retorna valores arbitrarios, ChromaDB retorna distancias coseno). **Reciprocal Rank Fusion** (RRF) resuelve esto sin necesidad de normalización: solo usa las posiciones de cada documento en cada ranking, no los scores. La fórmula es `1 / (k + rank)` donde k=60 actúa como factor de suavizado.

```
RRF(d) = Σ  1 / (k + rankᵢ(d))
```

Resultado: documentos que aparecen en ambas listas suben al tope; los que solo aparecen en una lista reciben puntuación parcial.

### 2. LangGraph sobre LangChain Expression Language (LCEL)

LangGraph permite definir el ciclo agente-herramienta como un grafo de estados explícito. Las ventajas sobre una chain LCEL:

- **Memoria nativa**: `InMemorySaver` mantiene el historial de conversación sin código adicional
- **Estado tipado**: cada nodo recibe y emite un `AgentState` bien definido
- **Control de flujo**: es trivial agregar nodos de validación, re-intentos o routing condicional en el futuro

Para esta prueba, el grafo tiene un único ciclo: `agent → tools → agent`. La extensibilidad justifica la elección.

### 3. Abstracción `BaseLoader` para fuentes de datos

Todas las fuentes de datos implementan la misma interfaz:

```python
class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[dict]: ...
```

Esto permite agregar nuevas fuentes (FHIR API, base de datos relacional, RSS feed) sin modificar el pipeline RAG. El `chunker.py` solo sabe que recibe `List[Dict]` — no le importa el origen.

### 4. Caché a nivel de módulo para documentos y vectorstore

El pipeline carga ~1.400 chunks en cada consulta incluyendo 3 PDFs, una base de datos SQLite y una petición HTTP a Wikipedia. En producción esto es inaceptable. La solución implementada usa variables de módulo con inicialización lazy:

```python
_vectorstore_cache: Chroma | None = None
_documents_cache: list[Document] | None = None
```

La primera consulta paga el costo de carga (~5 segundos). Las siguientes van directamente a embeddings + búsqueda. El vectorstore no se recarga aunque el proceso siga corriendo.

### 5. Soporte dual de LLM: Ollama (local) y OpenAI (nube)

Una sola variable de entorno (`LLM_PROVIDER`) determina qué proveedor se usa. La función `get_llm()` en `src/config.py` es la única fuente de verdad — tanto el agente como el pipeline RAG la consumen.

Esto permite:
- **Desarrollo local**: Ollama + llama3.2 — cero costo, privacidad total
- **Producción / demo con mayor calidad**: OpenAI gpt-4o-mini

### 6. ChromaDB con persistencia en disco

ChromaDB persiste el índice vectorial en `data/chroma/` como SQLite. No requiere servidor externo, no tiene costo, funciona offline. Para un PoC con ~1.400 documentos, el rendimiento es más que suficiente.

**Nota importante sobre reconstrucción del vectorstore**: `Chroma.from_documents()` hace *append* a una colección existente, no la reemplaza. El script `src/rag/embedder.py` elimina la colección antes de reconstruir para evitar duplicados.

---

## Supuestos

| Supuesto | Justificación |
|----------|---------------|
| `legacy_clinic.db` (SQLite) simula un sistema clínico legado de escritorio | La prueba pide extracción desde "aplicación de escritorio". Un sistema legado real tendría un motor de base de datos local (Access, SQLite, Firebird). Se usa SQLite como representación fiel y portable de ese escenario. |
| Wikipedia (WHO Model List of Essential Medicines) representa una fuente web externa | En producción sería un formulario farmacéutico hospitalario con API propia. Wikipedia permite demostrar scraping real sin credenciales. |
| Los PDFs son guías clínicas de dominio público | Se usan GPC de hipertensión (Ministerio de Salud), guía de inmunogenicidad OMS y lista de medicamentos esenciales OMS. No contienen datos de pacientes reales. |
| El corpus es estático durante una sesión | Los documentos no cambian mientras el servidor está corriendo. Si se agregan PDFs, hay que reconstruir el vectorstore y reiniciar. |
| Un único espacio vectorial para todas las fuentes | Pacientes, guías clínicas y medicamentos comparten la misma colección ChromaDB. En producción se separarían por control de acceso. |

---

## Trade-offs y Limitaciones

### Lo que funciona bien
- El retrieval híbrido supera claramente a la búsqueda semántica pura para consultas de entidades específicas (nombres de pacientes, medicamentos exactos)
- La abstracción `BaseLoader` hace trivial agregar fuentes nuevas
- El caché de módulo elimina la latencia de carga después de la primera consulta
- El sistema funciona completamente offline con Ollama

### Limitaciones conocidas

**`response_format=ClinicalResponse` con modelos Ollama pequeños**

La salida estructurada (JSON schema enforcement) funciona de forma nativa con OpenAI via `with_structured_output()`. Con llama3.2 (3B parámetros), el modelo no siempre respeta el schema JSON y puede emitir una representación textual en lugar de JSON válido. El UI tiene un fallback a texto plano para este caso.

*Con OpenAI como proveedor, la salida estructurada funciona correctamente.*

**BM25 en memoria, reconstruido en cada arranque**

El índice BM25 se reconstruye desde los documentos en memoria cada vez que se inicia el servidor (~2 segundos para 1.400 docs). No está persistido en disco.

**Web scraping en arranque**

`WebScraper` hace una petición HTTP a Wikipedia cuando se carga el pipeline por primera vez. Si Wikipedia no está disponible, el sistema degrada gracefully (retorna lista vacía y loguea el error) pero pierde esa fuente de datos para esa sesión.

**Sin autenticación ni control de acceso**

La interfaz Streamlit no tiene login. En un sistema clínico real, los datos de pacientes requieren autenticación y control de acceso por rol (HIPAA / Ley 1581 en Colombia).

---

## ¿Qué haría diferente con más tiempo?

**1. Indexado BM25 persistente**
Serializar el índice BM25 con `pickle` para no reconstruirlo en cada arranque. Con corpus grandes, la reconstrucción sería el cuello de botella principal.

**2. Carga asíncrona de fuentes**
Usar `asyncio.gather()` para cargar PDFs, SQLite y Wikipedia en paralelo en lugar de secuencialmente. El web scraping es el paso más lento y bloquea al resto.

**3. Suite de tests completa**
Los archivos `tests/test_extraction.py` y `tests/test_rag.py` existen como stubs pero sin contenido. Con más tiempo implementaría:
- Tests unitarios para `LegacyLoader` y `WebScraper` con mocks de BD y HTTP
- Tests de integración para el retriever con un ChromaDB en memoria
- Tests del agente con un LLM mockeado

**4. Evaluación del pipeline RAG**
Usar [RAGAS](https://github.com/explodinggradients/ragas) o un gold standard Q&A propio para medir métricas de retrieval (Precision@k, MRR, Recall@k) y de generación (faithfulness, answer relevancy). Actualmente la calidad se evalúa manualmente.

**5. Manejo robusto de structured output para modelos locales**
Implementar un parser de fallback que extraiga los campos de `ClinicalResponse` desde texto libre cuando el modelo no genera JSON válido. Esto requeriría una segunda pasada de extracción con un prompt específico.

**6. Contenerización con Docker Compose**
Empaquetar la app Streamlit y el servidor Ollama en un `docker-compose.yml` para despliegue reproducible con un solo comando.

**7. Estrategia de chunking optimizada**
El tamaño de chunk actual (800 chars, 100 overlap) fue elegido empíricamente. Con más tiempo, haría un barrido sistemático evaluando el impacto en métricas de retrieval para el dominio clínico.

---

## Manejo de Errores

| Escenario | Comportamiento |
|-----------|---------------|
| Wikipedia no disponible | `WebScraper.load()` captura la excepción, loguea el error y retorna lista vacía. El pipeline continúa con las fuentes restantes. |
| PDF corrupto o ilegible | `chunker.py` captura excepciones por archivo. Los PDFs fallidos se omiten y los demás se procesan. |
| SQLite inexistente | `LegacyLoader` loguea el error y retorna lista vacía. El sistema no crashea. |
| ChromaDB no inicializado | `pipeline.py` llama `build_vectorstore()` si `load_vectorstore()` falla. |
| LLM_PROVIDER inválido | `get_llm()` lanza `ValueError` con mensaje explícito en lugar de fallar silenciosamente. |
| Query sin contexto relevante | El prompt instruye al LLM a declarar explícitamente cuando no hay información suficiente, en lugar de alucinar. |

---

## Tecnologías

| Tecnología | Versión | Rol |
|-----------|---------|-----|
| **Python** | 3.11 | Lenguaje principal |
| **UV** | latest | Gestión de paquetes y entornos virtuales |
| **LangChain** | ≥1.2.12 | Framework de componentes LLM |
| **LangGraph** | ≥1.1.2 | Orquestación de agentes con estado y memoria |
| **LangChain-OpenAI** | ≥1.1.11 | Integración con modelos GPT |
| **LangChain-Ollama** | ≥1.0.1 | Integración con modelos locales (llama3.2) |
| **ChromaDB** | ≥1.5.5 | Vectorstore con persistencia local |
| **rank-bm25** | ≥0.2.2 | Búsqueda léxica BM25Okapi |
| **PyPDF** | ≥6.8.0 | Extracción de texto desde PDFs |
| **BeautifulSoup4** | ≥4.14.3 | Web scraping estructurado |
| **Pydantic** | ≥2.12.5 | Contratos de datos y validación |
| **Streamlit** | ≥1.55.0 | Interfaz web conversacional |
| **python-dotenv** | ≥1.2.2 | Gestión de variables de entorno |

---

## Estructura del Proyecto

```
clinical-ai-assistant/
├── src/
│   ├── config.py                   # Configuración centralizada del LLM
│   ├── agent/
│   │   ├── agent.py                # Agente LangGraph con memoria (InMemorySaver)
│   │   ├── tools.py                # Tool: answer_clinical_question
│   │   └── schemas.py              # ClinicalResponse (Pydantic)
│   ├── rag/
│   │   ├── pipeline.py             # Orquestador RAG con caché de módulo
│   │   ├── retriever.py            # Búsqueda híbrida BM25 + ChromaDB + RRF
│   │   ├── chunker.py              # Carga multi-fuente y chunking de documentos
│   │   └── embedder.py             # Embeddings y gestión de ChromaDB
│   ├── extraction/
│   │   ├── base.py                 # Interfaz BaseLoader (ABC)
│   │   ├── legacy_loader.py        # Fuente 1: SQLite legacy (pacientes)
│   │   └── web_scraper.py          # Fuente 2: Wikipedia (medicamentos OMS)
│   └── ui/
│       └── app.py                  # Interfaz Streamlit con chat y renderizado
├── docs/
│   └── sample_docs/                # PDFs: guías clínicas y lista OMS
├── data/
│   ├── raw/
│   │   └── legacy_clinic.db        # Base de datos SQLite (sistema legado simulado)
│   ├── processed/                  # CSVs exportados de las fuentes
│   └── chroma/                     # Vectorstore ChromaDB (generado, no versionado)
├── tests/
│   ├── test_extraction.py
│   └── test_rag.py
├── main.py                         # Punto de entrada — construye el vectorstore
├── pyproject.toml                  # Dependencias y metadata del proyecto (UV)
├── .env.example                    # Plantilla de variables de entorno
└── makefile                        # Comandos de desarrollo
```

---

## Prerrequisitos

### Ejecución local con Ollama (sin costo)

- **Python 3.11** — [python.org](https://www.python.org/downloads/)
- **UV** — [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **Ollama** — [ollama.com](https://ollama.com/) con el modelo `llama3.2` descargado:
  ```bash
  ollama pull llama3.2
  ```

### Ejecución con OpenAI (mayor calidad de respuesta)

- Los mismos prerrequisitos anteriores, excepto Ollama
- **API Key de OpenAI** — [platform.openai.com](https://platform.openai.com/api-keys)

---

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/Jhonatand26/clinical-ai-assistant.git
cd clinical-ai-assistant
```

### 2. Instalar dependencias

```bash
uv sync
```

Esto instala todas las dependencias de `pyproject.toml` y registra el paquete en modo editable, lo que hace que los imports `from src.xxx import yyy` funcionen sin manipulación de `sys.path`.

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` según el proveedor deseado:

**Opción A — Ollama (local, sin costo):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

**Opción B — OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 4. Construir el vectorstore

Este paso procesa los PDFs, la base de datos legado y el scraping web, genera los embeddings y persiste el índice ChromaDB en `data/chroma/`. Solo es necesario ejecutarlo una vez (o cuando se agreguen nuevos documentos).

```bash
uv run python main.py
```

El log indicará el número de chunks procesados por fuente:
```
INFO:src.rag.chunker:Found 3 PDFs to process.
INFO:src.rag.chunker:Total documents/chunks from all sources: 1395
INFO:src.rag.embedder:Building vectorstore with 1395 chunks...
INFO:src.rag.embedder:Vectorstore persisted at data\chroma
```

### 5. Iniciar la aplicación

```bash
uv run streamlit run src/ui/app.py
```

Abrir en el navegador: [http://localhost:8501](http://localhost:8501)

---

## Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Proveedor del LLM: `openai` o `ollama` | `ollama` |
| `OPENAI_API_KEY` | API key de OpenAI (**requerida si `LLM_PROVIDER=openai`**) | — |
| `OPENAI_CHAT_MODEL` | Modelo de chat de OpenAI | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings de OpenAI | `text-embedding-3-small` |
| `OLLAMA_BASE_URL` | URL del servidor Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modelo a usar con Ollama | `llama3.2` |
| `CHROMA_PERSIST_DIR` | Directorio de persistencia del vectorstore | `./data/chroma` |
| `APP_ENV` | Entorno de la aplicación | `development` |

---

## Tests

```bash
uv run pytest
```

> Los stubs de test existen en `tests/` pero están pendientes de implementación completa. Ver sección *¿Qué haría diferente con más tiempo?* para el plan de testing.

---

## Troubleshooting

### El agente tarda mucho en la primera pregunta

La primera consulta carga ~1.400 chunks desde todas las fuentes (PDFs, SQLite, Wikipedia) y los cachea en memoria. Las consultas siguientes son significativamente más rápidas. Este comportamiento es visible en los logs con el mensaje `"Loading all documents (first time)..."`.

---

### La búsqueda semántica retorna muy pocos resultados

El vectorstore ChromaDB puede estar desactualizado. Reconstruirlo desde cero:

```bash
rm -rf data/chroma
uv run python main.py
```

Reiniciar la aplicación después de reconstruir.

---

### El agente dice que no tiene información sobre un paciente que sí existe

Mismo síntoma que el anterior: el vectorstore fue construido sin los registros del sistema legado. Reconstruir como se indica arriba.

---

### `ValueError: Unsupported LLM_PROVIDER`

Verificar que `.env` contiene `LLM_PROVIDER=openai` o `LLM_PROVIDER=ollama` (en minúsculas, sin espacios).

---

### Ollama no responde (`Connection refused`)

Verificar que el servidor Ollama está corriendo y el modelo está descargado:

```bash
ollama serve          # iniciar el servidor
ollama list           # verificar modelos disponibles
ollama pull llama3.2  # descargar si no aparece
```

---

### La respuesta muestra `ClinicalResponse id: ... type: Answer text: ...`

Comportamiento conocido con modelos Ollama pequeños (llama3.2, 3B parámetros). El modelo no siempre respeta el schema JSON de `ClinicalResponse`. La interfaz hace fallback a texto plano — la información sigue siendo correcta, solo pierde el formato estructurado.

**Solución**: usar `LLM_PROVIDER=openai` en `.env` para obtener salida estructurada consistente.

---

## Licencia

Este proyecto fue desarrollado como prueba técnica. Todo el código es de autoría propia del candidato.

© 2025 Jhonatan David Rengifo
