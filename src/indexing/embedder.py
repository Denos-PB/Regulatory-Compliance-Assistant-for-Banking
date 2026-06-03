import logging
import os

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..config import load_indexing_config

logger = logging.getLogger(__name__)


def _client(model: str | None = None, cfg: dict | None = None) -> OpenAIEmbeddings:
    c = {**load_indexing_config(), **(cfg or {})}
    model = model or c["embedding_model"]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def embed_query(
    query: str,
    *,
    model: str | None = None,
    cfg: dict | None = None,
) -> list[float]:
    """Embed a single search query (used by the retriever)."""
    if not query or not query.strip():
        return []
    embedder = _client(model=model, cfg=cfg)
    return embedder.embed_query(query.strip())


def embed_chunks(
    chunks: list[Document],
    *,
    model: str | None = None,
    batch_size: int | None = None,
    cfg: dict | None = None,
) -> list[Document]:
    c = {**load_indexing_config(), **(cfg or {})}
    model = model or c["embedding_model"]
    batch_size = batch_size if batch_size is not None else c["embedding_batch_size"]

    chunks = [doc for doc in chunks if doc.page_content.strip()]
    if not chunks:
        logger.warning("No chunks to embed")
        return []

    embedder = _client(model=model, cfg=c)
    texts = [doc.page_content for doc in chunks]
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = embedder.embed_documents(batch)
        all_vectors.extend(vectors)
        logger.info(
            "Embedded %d / %d chunks",
            min(start + batch_size, len(texts)),
            len(texts),
        )

    embedded: list[Document] = []
    for chunk, vector in zip(chunks, all_vectors, strict=True):
        meta = dict(chunk.metadata)
        meta["embedding"] = vector
        meta["embedding_model"] = model
        meta["embedding_dim"] = len(vector)
        embedded.append(Document(page_content=chunk.page_content, metadata=meta))

    logger.info("Embedded %d chunks with %s", len(embedded), model)
    return embedded
