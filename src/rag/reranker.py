import logging
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ..config import load_rag_config
from ..observability.langfuse_tracing import observe
from .retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cross_encoder(model_name: str) -> TextCrossEncoder:
    logger.info("Loading rerank model: %s", model_name)
    return TextCrossEncoder(model_name=model_name)


@observe(name="cross_encoder_rerank")
def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
    cfg: dict | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    rag_cfg = {**load_rag_config(), **(cfg or {})}
    model_name = rag_cfg.get("rerank_model", "BAAI/bge-reranker-base")
    min_score = rag_cfg.get("rerank_min_score")
    encoder = _cross_encoder(model_name)
    texts = [c.text for c in chunks]
    scores = list(encoder.rerank(query.strip(), texts))
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    if min_score is not None:
        filtered = [pair for pair in ranked if pair[1] >= float(min_score)]
        if filtered:
            ranked = filtered
        else:
            logger.warning(
                "All rerank scores below min_score=%s; keeping top candidate",
                min_score,
            )
            ranked = ranked[:1]

    ranked = ranked[:top_k]

    logger.info("Reranked %d candidate(s) to top_%d", len(chunks), len(ranked))
    return [
        RetrievedChunk(
            text=chunk.text,
            score=float(score),
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page_number=chunk.page_number,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            file_type=chunk.file_type,
        )
        for chunk, score in ranked
    ]
