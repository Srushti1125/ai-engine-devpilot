# DevPilot AI Engine — Team Handoff & Integration Guide
**Owner:** Person 3 (GenAI + RAG + Prompt Engineering)  
**Service:** AI Engine (`ai-engine`)  
**Status:** Tested & Verified ✅ (11/11 unit tests passed, all 6 artifact pipelines operational)

---

## Architecture Overview

```
[Person 1: Frontend UI]
       │
       ▼ (HTTP / JWT)
[Person 2: Backend & Platform APIs]
       │
       ├──────────────────────────────────────┐
       ▼ (HTTP / JSON)                        ▼ (SQL Queries)
[Person 3: AI Engine (Port 8001)]     [PostgreSQL + pgvector]
  ├── RAG Context Retrieval ◄─────────────────┘
  ├── Prompt Engine (v1 templates)
  ├── Multi-Provider LLM Layer (Groq / Gemini / Mistral)
  ├── JSON Schema Validation (Person 4 Schemas)
  └── Quality Checker (Person 5 Metrics)
```

---

## 🎨 To Person 1: Frontend & Dashboard Team

### 1. Artifact Generation UI
* **Dropdown Selection:** Populate the artifact selection dropdown dynamically via `GET /artifact-types` through Person 2's backend.
* **Available Types:**
  * `user_story`
  * `acceptance_criteria`
  * `test_case`
  * `api_spec`
  * `architecture`
  * `sprint_plan`

### 2. Rendering Artifact Outputs
The `output` field in the response from generation is always a structured JSON object:
* **User Stories (`user_story`):** Array of stories with `title`, `story` (*"As a... I want... so that..."*), `priority` (`High` | `Medium` | `Low`), `category`, and `dependencies`.
* **Acceptance Criteria (`acceptance_criteria`):** Array of BDD criteria with `given`, `when`, `then`, and `case_type` (`positive` | `negative` | `edge`).
* **Test Cases (`test_case`):** Array with `id`, `title`, `category` (`positive` | `negative` | `boundary` | `security`), `steps` (array of strings), `expected_result`, and `traceability`.
* **API Specs (`api_spec`):** Array of endpoints with `method`, `path`, `description`, `request_schema`, `response_schema`, and `error_codes`.
* **Architecture (`architecture`):** Object containing `pattern`, `rationale`, `technology_recommendations`, `components` (list), `data_flow`, and `security_considerations`.
* **Sprint Plan (`sprint_plan`):** Object with `assumed_velocity_points_per_sprint`, `critical_path` array, and `items` array (`story_title`, `story_points`, `effort_estimate_hours`, `sprint`, `dependencies`).

### 3. Displaying Metadata
Display the following metadata on the artifact view cards / dashboard:
* **Latency:** `latency_ms` (e.g. `2,203 ms`)
* **Tokens Used:** `usage.total_tokens`
* **Prompt Version:** `prompt_version` (e.g. `user_story_v1`)

---

## ⚙️ To Person 2: Backend & Platform APIs Team

### 1. Database & Vector Store Configuration
* **Enable pgvector extension (Run once on PostgreSQL RDS/local):**
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
* **Embedding Dimension:** Embeddings use `gemini-embedding-001` with **768 dimensions**. Ensure any direct vector columns or indexes are created as `vector(768)`.

### 2. API Contract & Integration Endpoints
Base URL: `http://localhost:8001` (or internal Docker container network)

#### `GET /health`
* Readiness and liveness probe for health checks.
* **Response:** `{"status": "ok", "default_llm": "..."}`

#### `GET /artifact-types`
* Returns array of supported artifact identifiers.
* **Response:** `{"artifact_types": ["acceptance_criteria", "api_spec", "architecture", "sprint_plan", "test_case", "user_story"]}`

#### `POST /ingest`
Call this after Person 1 uploads a file (.txt, .docx, .pdf) and your backend saves it to a reachable filesystem path or volume.
* **Request Body:**
  ```json
  {
    "file_path": "/path/to/uploaded/requirements.txt",
    "project_id": "proj_123",
    "doc_type": "requirement"
  }
  ```
* **Response:**
  ```json
  {
    "chunks_indexed": 42,
    "project_id": "proj_123",
    "doc_type": "requirement"
  }
  ```

#### `POST /generate`
Call this when the user triggers artifact generation.
* **Request Body:**
  ```json
  {
    "project_id": "proj_123",
    "artifact_type": "user_story",
    "requirement_text": "The system shall allow users to reset their password via email."
  }
  ```
* **Response:**
  ```json
  {
    "request_id": "c68ee94f-a10d-4c11-bab3-e22833fc5dfa",
    "artifact_type": "user_story",
    "project_id": "proj_123",
    "prompt_version": "user_story_v1",
    "output": { ... },
    "usage": {
      "prompt_tokens": 485,
      "completion_tokens": 308,
      "total_tokens": 793,
      "total_cost_usd": 0.0
    },
    "latency_ms": 2203
  }
  ```

### 3. Required Database Persisted Fields
Store the following in your database tables:
1. **`artifacts` table:** Store the `output` JSON along with `project_id`, `artifact_type`, and timestamps.
2. **`generation_history` / `audit_logs` table (Critical for Person 5):**
   * `request_id` (UUID)
   * `project_id`
   * `artifact_type`
   * `prompt_version`
   * `tokens_used` (`usage.total_tokens`)
   * `latency_ms`

---

## 🏗️ To Person 4: SDLC Artifact Generation Engine Team

### 1. Schema Ownership & Customization
* All Pydantic parsing schemas live in `app/schemas/`:
  * `user_story.py`
  * `acceptance_criteria.py`
  * `test_case.py`
  * `api_spec.py`
  * `architecture.py`
  * `sprint_plan.py`
* To modify or add fields to an artifact structure, update the Pydantic schema in `app/schemas/` and the corresponding prompt template in `app/prompts/templates/`.
* The AI engine will automatically enforce the schema and retry with structured repair if the LLM output is non-compliant.

### 2. Adding New Artifact Types
1. Create a schema in `app/schemas/<new_type>.py`
2. Register it in `app/schemas/registry.py` (`SCHEMA_REGISTRY`)
3. Add prompt template in `app/prompts/templates/<new_type>_v1.txt`
4. The pipeline, validation, and `/generate` endpoint will support it automatically with zero changes to `chain.py` or `main.py`.

---

## 📊 To Person 5: Evaluation, AI Assistant & Case Study Team

### 1. Quality & Format Compliance Evaluation
* The module `app/pipeline/quality_check.py` provides programmatic evaluators:
  * `format_compliance_score()`: Validates Pydantic adherence and non-empty responses.
  * `consistency_score()`: Evaluates structural determinism across repeated runs.
* Each `/generate` response includes:
  * `prompt_version`: Use this to compare evaluation metrics across prompt iterations (e.g. `user_story_v1` vs `user_story_v2`).
  * `usage`: Full token accounting (`prompt_tokens`, `completion_tokens`, `total_tokens`, `total_cost_usd`).
  * `latency_ms`: Response latency benchmarking.

### 2. Conversational AI Assistant (UC-14)
* The vector store collection `devpilot_requirements` indexes chunks with metadata (`project_id`, `doc_type`, `chunk_index`).
* The assistant can query the vector store using `get_vectorstore().as_retriever(search_kwargs={"filter": {"project_id": "..."}})` to retrieve exact requirements for Q&A.

---

## 🔐 To Person 6: Testing, Security & DevOps Team

### 1. Running Automated Tests (Secret-free for CI/CD)
The unit and smoke test suite runs without requiring live database connections or external API keys:
```bash
pip install pytest
pytest tests/ -q
```
*Current test status: `11 passed in 1.11s`*

### 2. Docker Containerization
Build and run the containerized AI Engine:
```bash
docker build -t devpilot-ai-engine .
docker run -p 8001:8001 --env-file .env devpilot-ai-engine
```

### 3. Environment Variables Configuration
Inject the following environment variables in runtime / AWS Secrets Manager:
```env
DATABASE_URL=postgresql+psycopg://<user>:<password>@<rds_host>:5432/<db_name>
GOOGLE_API_KEY=<google_gemini_api_key_for_embeddings>
GROQ_API_KEY=<groq_api_key_for_fast_inference>
DEFAULT_LLM=groq-openai/gpt-oss-120b
FALLBACK_LLM=gemini-2.5-flash
AI_ENGINE_PORT=8001
LOG_LEVEL=INFO
```
