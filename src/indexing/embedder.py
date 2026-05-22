import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from ..config import load_indexing_config

load_dotenv()
logger = logging.getLogger(__name__)


def _client(model: str | None = None, cfg: dict | None = None) -> OpenAIEmbeddings:
    c = {**load_indexing_config(), **(cfg or {})}
    model = model or c["embedding_model"]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")
    return OpenAIEmbeddings(model=model, api_key=api_key)

def load_chunks_json(path: str | Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [
        Document(page_content=row["text"], metadata=row.get("metadata", {}))
        for row in rows
    ]

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

    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        logger.warning("No chunks to embed")
        return []
    
    embedder = _client(model=model,cfg=c)
    texts = [c.page_content for c in chunks]
    all_vectors = list[list[float]] = []

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
    for chunk, vector in zip(chunks,all_vectors,strict=True):
        meta = dict[chunk.metadata]
        meta["embedding"]=vector
        meta['embedding_model']=model
        meta['embedding_dim']=len(vector)
        embedded.append(Document(page_content=chunk.page_content, metadata=meta))

    logger.info("Embedded %d chunks with %s", len(embedded), model)
    return embedded

if __name__ == "__main__":
    from ..logging_config import setup_logging
    setup_logging()
    root = Path(__file__).resolve().parents[2]
    chunks_path = root / "data/processed/chunks.json"
    if not chunks_path.exists():
        raise SystemExit(f"Run chunking first. Missing: {chunks_path}")
    chunks = load_chunks_json(chunks_path)
    embedded = embed_chunks(chunks)
    logger.info("Done. %d chunks ready for vector store.", len(embedded))
    logger.info("Example chunk_id: %s", embedded[0].metadata.get("chunk_id"))