import logging
import os
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

from ..config import load_indexing_config, load_rag_config
from ..indexing.embedder import _client as _embedding_client
from ..indexing.store import search as qdrant_search

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    text: str
    score: float
    chunk_id: str
    source: str
    page_number: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int | None = None
    file_type: str | None = None

    def citation(self) -> str:
        parts=[self.source]
        if self.page_number is not None:
            parts.append(f"p.{self.page_number}")
        elif self.page_start is not None:
            end=self.page_end or self.page_start
            parts.append(f"pp.{self.page_start}-{end}")
        parts.append(self.chunk_id)
        return " | ".join(parts)

def _embed_query(query: str, cfg: dict | None = None) -> list[float]:
    indexing_cfg = {**load_indexing_config(), **(cfg or {})}
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")
    embedder = _embedding_client(cfg=indexing_cfg)
    return embedder.embed_query(query.strip())


def _hit_to_chunk(hit: Any) -> RetrievedChunk | None:
    payload = hit.payload or {}
    text = payload.get("text") or ""
    if not text.strip():
        return None
    return RetrievedChunk(
        text=text.strip(),
        score=float(hit.score),
        chunk_id=payload.get("chunk_id", ""),
        source=payload.get("source", "unknown"),
        page_number=payload.get("page_number"),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        chunk_index=payload.get("chunk_index"),
        file_type=payload.get("file_type"),
    )

def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    source_filter: str | None = None,
    score_threshold: float | None = None,
    cfg: dict | None = None,
) -> list[RetrievedChunk]:
    if not query or not query.strip():
        logger.warning("Empty query")
        return []
    
    rag_cfg = {**load_rag_config(), **(cfg or {})}
    top_k = top_k if top_k is not None else rag_cfg["top_k"]
    score_threshold = (
        score_threshold if score_threshold is not None else
        rag_cfg["score_threshold"]
    )

    logger.info("Retrieving top_%d for query: %s", top_k, query[:80])
    query_vector = _embed_query(query, cfg=cfg)
    response = qdrant_search(
        query_vector,
        limit=top_k,
        source_filter=source_filter,
        cfg=cfg,
    )

    chunks: list[RetrievedChunk] = []
    for hit in response.points:
        chunk = _hit_to_chunk(hit)
        if chunk is None:
            continue
        if score_threshold and chunk.score < score_threshold:
            continue
        chunks.append(chunk)

    logger.info("Retrieved %d chunk(s)", len(chunks))
    return chunks

def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks=[]

    for i, c in enumerate(chunks,start=1):
        blocks.append(
            f"[{i}] ({c.citation()})\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)