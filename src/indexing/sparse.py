import logging
from functools import lru_cache
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
from ..config import load_indexing_config
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _sparse_model(model_name: str) -> SparseTextEmbedding:
    logger.info("Loading sparse model: %s", model_name)
    return SparseTextEmbedding(model_name=model_name)

def _to_sparse_vector(embedding) -> SparseVector:
    return SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )

def embed_sparse_documents(
    texts: list[str],
    *,
    cfg: dict | None = None,
) -> list[SparseVector]:
    c = {**load_indexing_config(), **(cfg or {})}
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return []
    model=_sparse_model(c['sparse_model'])
    vectors: list[SparseVector] = []
    for emb in model.embed(texts,batch_size=256):
        vectors.append(_to_sparse_vector(emb))
    logger.info("Sparse-embedded %d chunk(s)", len(vectors))
    return vectors

def embed_sparse_query(
    query: str,
    *,
    cfg: dict | None = None,
) -> SparseVector:
    if not query or not query.strip():
        return SparseVector(indices=[], values=[])
    
    c = {**load_indexing_config(), **(cfg or {})}
    model = _sparse_model(c["sparse_model"])
    emb = next(model.query_embed(query.strip()))
    return _to_sparse_vector(emb)