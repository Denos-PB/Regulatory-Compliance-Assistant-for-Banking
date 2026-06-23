import logging
import os
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
from ..config import load_indexing_config, load_rag_config
from ..indexing.embedder import _client as _embedding_client
from ..indexing.store import search as qdrant_search
from ..indexing.sparse import embed_sparse_query
from ..indexing.store import search_hybrid
from ..observability.langfuse_tracing import (
    langchain_callbacks,
    observe,
    retrieval_output,
    update_current_output,
)

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
    callbacks = langchain_callbacks()
    if callbacks:
        return embedder.embed_query(query.strip(), config={"callbacks": callbacks})
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

@observe(name="qdrant_retrieve")
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

    hybrid = bool(rag_cfg.get("hybrid_enabled"))
    rerank = bool(rag_cfg.get("rerank_enabled"))
    prefetch_limit = rag_cfg.get("hybrid_prefetch_limit", 20)
    candidate_limit = (
        max(int(rag_cfg.get("rerank_top_n", 20)), top_k) if rerank else top_k
    )
    if hybrid and rerank:
        prefetch_limit = max(prefetch_limit, candidate_limit)

    mode = "hybrid+rerank" if hybrid and rerank else "hybrid" if hybrid else "dense"
    if rerank and not hybrid:
        mode = f"{mode}+rerank"
    logger.info(
        "Retrieving (%s) top_%d (candidates=%d) for query: %s",
        mode,
        top_k,
        candidate_limit,
        query[:80],
    )
    if hybrid:
        dense_vector = _embed_query(query, cfg=cfg)
        sparse_vector = embed_sparse_query(query, cfg=cfg)
        response = search_hybrid(
            dense_vector,
            sparse_vector,
            limit=candidate_limit,
            prefetch_limit=prefetch_limit,
            source_filter=source_filter,
            cfg=cfg,
        )
    else:
        dense_vector = _embed_query(query, cfg=cfg)
        response = qdrant_search(
            dense_vector,
            limit=candidate_limit,
            source_filter=source_filter,
            cfg=cfg,
        )

    chunks: list[RetrievedChunk] = []
    for hit in response.points:
        chunk = _hit_to_chunk(hit)
        if chunk is None:
            continue
        if not rerank and score_threshold and chunk.score < score_threshold:
            continue
        chunks.append(chunk)

    if rerank and chunks:
        from .reranker import rerank_chunks

        chunks = rerank_chunks(query, chunks, top_k=top_k, cfg=rag_cfg)

    logger.info("Retrieved %d chunk(s)", len(chunks))

    update_current_output(retrieval_output(chunks))
    return chunks

def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks=[]

    for i, c in enumerate(chunks,start=1):
        blocks.append(
            f"[{i}] ({c.citation()})\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)