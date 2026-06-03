# Regulatory Compliance Assistant for Banking

RAG over regulatory PDFs/HTML: ingest → chunk → embed → Qdrant → cited answers (OpenAI embeddings + DeepSeek generation).

## Setup

```bash
uv sync
cp .env.example .env   # OPENAI_API_KEY, DEEPSEEK_API_KEY
```

Put documents in `data/raw/` (paths in `config.yaml`).

## Run locally

```bash
# 1) Vector DB
docker run -d --name qdrant -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant

# 2) Ingest + index (batch; run again when you add PDFs)
uv run python -m src.cli run

# 3) Query
uv run python -m src.cli ask "What is customer due diligence?"

# Optional: one-shot ingest + index + ask
uv run python -m src.cli run -q "What are KYC requirements for corporate clients?"

# Check env + Qdrant
uv run python -m src.cli doctor
```

## API (local)

```bash
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is KYC?"}'
```

## Docker

**Split:** indexing on the host (CLI), **API in Docker**, vectors in **Qdrant** (not baked into the app image).

```bash
cp .env.example .env   # fill keys; QDRANT_URL=http://localhost:6333 for host CLI

# Qdrant + API
docker compose up -d --build

# Index from host (uses localhost:6333 → published Qdrant port)
uv run python -m src.cli run

# API inside compose uses QDRANT_URL=http://qdrant:6333 (set in docker-compose.yml)
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is KYC?"}'
```

| Step | Where | `QDRANT_URL` |
|------|--------|----------------|
| `cli run` | host | `http://localhost:6333` (`.env`) |
| `api` service | container | `http://qdrant:6333` (compose override) |

Rebuild the app image only when **code** changes, not when you add PDFs.

```bash
docker compose up -d --build api
docker compose logs -f api
docker compose down          # stop
docker compose down -v       # also deletes Qdrant volume (re-index required)
```

## Architecture

```text
data/raw → src/pipeline.py (ingest + index) → Qdrant
       → rag (retrieve + DeepSeek) → answer + sources
```

Orchestration: `src/pipeline.py` (`run_ingestion`, `run_indexing`, `run_all`).  
Health checks: `src/health.py` (CLI `doctor` and `GET /health`).

## Config

- `config.yaml` — paths, chunking, Qdrant, models
- `.env` — API keys; optional `QDRANT_URL`, `DEEPSEEK_BASE_URL`
