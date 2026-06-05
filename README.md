# Regulatory Compliance Assistant for Banking

RAG over regulatory PDFs/HTML: ingest → chunk → embed → Qdrant → cited answers (OpenAI embeddings + DeepSeek generation).

## Setup

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY, DEEPSEEK_API_KEY
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Put documents in `data/raw/` (paths configurable in `config.yaml`).

## Run (one command)

```bash
# Ingest + index
uv run python -m src.cli run

# Ingest + index + ask
uv run python -m src.cli run -q "What are KYC requirements for corporate clients?"

# Query only (index must exist)
uv run python -m src.cli ask "What is customer due diligence?"

# Check env
uv run python -m src.cli doctor
```

## API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
# POST http://127.0.0.1:8000/ask  {"question": "..."}
```

## Architecture

```text
data/raw → src/pipeline.py (ingest + index) → Qdrant
       → rag (retrieve + DeepSeek) → answer + sources
```

All orchestration lives in `src/pipeline.py` (`run_ingestion`, `run_indexing`, `run_all`).

## Config

- `config.yaml` — paths, chunking, Qdrant, models
- `.env` — API keys (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, optional `QDRANT_URL`)
