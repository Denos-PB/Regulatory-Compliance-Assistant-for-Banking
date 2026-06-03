import logging
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.health import doctor_report
from src.rag.chain import answer

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Regulatory Compliance RAG API", version="0.1.0")

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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    report = doctor_report(ping_qdrant=True)
    status = dict(report.get("status") or {})
    status["message"] = "Required API keys set; Qdrant reachable." if report["ok"] else (
        "Check env vars and Qdrant connectivity."
    )
    return HealthResponse(
        ok=report["ok"],
        env=report["env"],
        status=status,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = answer(
            req.question,
            top_k=req.top_k,
            source_filter=req.source_filter,
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("RAG request failed for question=%r", req.question[:80])
        raise HTTPException(
            status_code=500,
            detail="RAG request failed. Check server logs, Qdrant, and API keys.",
        ) from None
