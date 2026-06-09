import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    from src.observability.langfuse_tracing import flush_traces

    flush_traces()


app = FastAPI(title="Regulatory Compliance RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer from indexed documents")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    source_filter: Optional[str] = Field(
        None,
        description="Optional: restrict retrieval to one document source path (must match payload 'source')",
    )
    return_chunks: bool = Field(False, description="If true, include retrieved chunks in response")

class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []
    chunks: Optional[list[dict[str, Any]]] = None

class HealthResponse(BaseModel):
    ok: bool
    env: dict[str, bool]
    status: dict[str, Any] = {}

class SourcesResponse(BaseModel):
    sources: list[str]

def _doctor_env() -> dict[str, bool]:
    from src.observability.langfuse_tracing import is_enabled, langfuse_installed

    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "QDRANT_URL": bool(os.getenv("QDRANT_URL")) or True,
        "LANGFUSE_CONFIGURED": langfuse_installed() and is_enabled(),
    }

@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    env_ok = _doctor_env()
    return HealthResponse(
        ok=env_ok["OPENAI_API_KEY"] and env_ok["DEEPSEEK_API_KEY"],
        env=env_ok,
        status={
            "message": "Service is up. Keys are checked locally only."
        },
    )

@app.get("/sources", response_model=SourcesResponse)
def sources() -> SourcesResponse:
    try:
        from src.indexing.store import list_sources

        return SourcesResponse(sources=list_sources())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list sources: {e}") from e

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    try:
        from src.rag.chain import answer

        session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())
        result = answer(
            req.question,
            top_k=req.top_k,
            source_filter=req.source_filter,
            trace_tags=["api"],
            session_id=session_id,
        )
        chunks = result.get("chunks")
        if req.return_chunks and chunks is not None:
            serializable_chunks = []
            for c in chunks:
                if hasattr(c, "__dict__"):
                    serializable_chunks.append(c.__dict__)
                else:
                    serializable_chunks.append(str(c))
        else:
            serializable_chunks = None
        return AskResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []) or [],
            chunks=serializable_chunks,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="src.rag.chain.answer could not be imported. Implement rag/chain.py first."
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG failed: {e}\n{traceback.format_exc()}",
        ) from e

if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")
