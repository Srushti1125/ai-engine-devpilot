# DevPilot AI Engine — Person 3 (GenAI + RAG + Prompt Engineering)

Implements: **Requirement → RAG → Prompt → LLM → Structured Response → Quality Check**

Deployed as its own containerized HTTP service. Person 2's backend calls it
over HTTP (`/generate`, `/ingest`) rather than importing it as a Python
package — this keeps deploys decoupled and gives Person 6 a clean container
boundary for CI/CD.

## Why this stack (not locally hosted)

| Concern | Choice | Why |
|---|---|---|
| LLM (generation) | **Google Gemini** free tier, `gemini-2.5-flash` | Hosted API (`generativelanguage.googleapis.com`), $0/token, no billing account needed. Gemini's up-to-1M-token context window means retrieved RAG context + requirement text is never token-starved. |
| Embeddings | **Gemini `text-embedding-004`** (768-dim) | Same `GOOGLE_API_KEY` as generation — one provider account for the whole team. |
| Vector store | **pgvector** extension inside the **same shared AWS RDS Postgres** instance Person 2 owns | No separate vector-DB infra; inherits Person 2's backups/migrations for free. |
| Fallback LLM | `gemini-2.5-pro` (same provider), or **Mistral AI free "Experiment" tier** if Gemini's daily cap gets tight | 1B tokens/month ceiling, no local install — still hosted. |
| Ingestion | `unstructured` (hosted parsing library, runs in our own container) + AWS Textract fallback for scanned PDFs | Handles the mixed file types coming from Person 1's upload screen. |

Nothing here runs as a local/on-device model — every inference call leaves
the container over HTTPS to a hosted API.

## Quickstart

```bash
cp .env.example .env
# fill in GOOGLE_API_KEY (free, from https://aistudio.google.com/app/apikey)
# fill in DATABASE_URL (shared RDS instance — ask Person 2)

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Person 2 must first run once on the shared RDS instance:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Run tests (no live API/DB needed)

```bash
pip install pytest
pytest tests/ -q
```

These are wiring smoke tests (prompt templates load, schemas validate,
chunker works, retry/fallback logic works) — they don't call Gemini or
Postgres, so they run in CI without secrets. Add live integration tests
separately once Person 6 wires a test RDS/API-key into CI.

### Run with Docker

```bash
docker build -t devpilot-ai-engine .
docker run -p 8001:8001 --env-file .env devpilot-ai-engine
```

## API contract (for Person 2)

### `POST /generate`
```json
{
  "project_id": "proj_123",
  "artifact_type": "user_story",
  "requirement_text": "The system shall allow users to reset their password via email."
}
```
`artifact_type` ∈ `{user_story, acceptance_criteria, test_case, api_spec, architecture, sprint_plan}`
(see `GET /artifact-types`).

Response:
```json
{
  "request_id": "uuid",
  "artifact_type": "user_story",
  "project_id": "proj_123",
  "prompt_version": "user_story_v1",
  "output": { "stories": [ ... ] },
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost_usd": 0.0 },
  "latency_ms": 1234
}
```
Store `prompt_version` + `usage` in Person 2's `generation_history` table on
every call — Person 5's evaluation framework depends on this.

### `POST /ingest`
```json
{ "file_path": "/tmp/downloaded/requirements.docx", "project_id": "proj_123", "doc_type": "requirement" }
```
Response: `{ "chunks_indexed": 42, "project_id": "proj_123", "doc_type": "requirement" }`

Call this after Person 1's upload screen delivers a file and Person 2's
backend has it on a locally-reachable path (e.g. downloaded from S3 into
the container's filesystem, or a shared volume).

### `GET /health`
Readiness probe for Person 6's orchestrator / CI smoke tests.

### `GET /artifact-types`
Returns the list of valid `artifact_type` values, for Person 1's frontend
dropdowns.

## Handoff checklist

**To Person 2 (Backend):**
- [ ] `pgvector` extension enabled on shared RDS instance (`CREATE EXTENSION IF NOT EXISTS vector;`)
- [ ] Embedding dimension confirmed: **768** for `text-embedding-004` — size the `vector` column accordingly if you ever query it directly
- [ ] `/generate` and `/ingest` API contract agreed (see above)
- [ ] Env vars provisioned: `DATABASE_URL`, `GOOGLE_API_KEY`, `DEFAULT_LLM`, `FALLBACK_LLM`
- [ ] Team aware Gemini free tier is rate/quota-limited and subject to change without notice — `usage` block on every `/generate` response should be logged so we can monitor

**To Person 4 (Artifact engine):**
- [ ] Pydantic schema per artifact type owned jointly (`app/schemas/*.py`) — Person 3 parses against it, Person 4 defines what "valid" means; add new fields here as the artifact spec evolves
- [ ] Metadata fields (`project_id`, `doc_type`, `chunk_index`) available on every retrieved chunk for traceability

**To Person 5 (Evaluation):**
- [ ] `prompt_version` and token usage logged on every `/generate` call
- [ ] `app/pipeline/quality_check.py` exposes `consistency_score()` and `format_compliance_score()` helpers for prompt-performance comparison
- [ ] Retry count is implicit in whether the primary chain or `fallback_chain` produced the final result — consider adding an explicit `used_fallback: bool` to the response if you need it split out

**To Person 6 (Testing/DevOps):**
- [ ] Service containerized (`Dockerfile`)
- [ ] `/health` endpoint for readiness checks
- [ ] Secrets (`GOOGLE_API_KEY`, `DATABASE_URL`, `MISTRAL_API_KEY`) must be injected via AWS Secrets Manager / ECS task env vars at runtime — never baked into the image (`.dockerignore` excludes `.env`)
- [ ] `tests/test_smoke.py` runs secret-free in CI

## Scope note

This is a working end-to-end skeleton for one artifact type's worth of
plumbing, generalized across all six artifact types (`user_story`,
`acceptance_criteria`, `test_case`, `api_spec`, `architecture`,
`sprint_plan`) — prompts, schemas, and the full RAG→LLM→QC chain exist for
each. Still to tune with real data:
- Chunk size/overlap against real requirement docs (currently 700/100 chars)
- Prompt wording per artifact type, once Person 4 finalizes exact schema requirements
- Self-query filtering (auto-deriving `project_id`/`doc_type` filters from
  natural language) once the conversational assistant (UC-14, Person 5)
  needs it
- Quota/backoff handling if Gemini's free-tier daily cap becomes a real
  constraint during heavy demo/testing days — `Mistral` fallback branch is
  wired into `app/llm/provider.py` for this, just needs `MISTRAL_API_KEY`
