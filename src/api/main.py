import os
import traceback
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

def _doctor_env() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "QDRANT_URL": bool(os.getenv("QDRANT_URL")) or True,
    }

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

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        from src.rag.chain import answer
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
