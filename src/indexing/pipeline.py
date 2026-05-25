import logging
import os
from pathlib import Path
from ..config import load_indexing_config
from ..logging_config import setup_logging
from .chunk import chunk_documents, load_parsed_json, save_chunks_json
from .embedder import embed_chunks
from .store import upsert_chunks
logger = logging.getLogger(__name__)
def run_indexing_pipeline(
    parsed_path: str | Path | None = None,
    output_dir: str | Path = "data/processed",
    *,
    save_chunks: bool = True,
    recreate_collection: bool = False,
) -> int:
    setup_logging()
    logger.info("Starting indexing pipeline")
    output_dir = Path(output_dir)
    parsed_path = Path(parsed_path or output_dir / "parsed_document.json")
    chunks_path = output_dir / "chunks.json"

    try:
        cfg=load_indexing_config()
    except Exception as e:
        logger.exception("Failed to load indexing config")
        raise RuntimeError("Indexing pipeline aborted: config load failed") from e
    
    if not parsed_path.exists():
        logger.error("Parsed documents not found: %s", parsed_path)
        raise FileNotFoundError(
            f"Missing {parsed_path}. Run ingestion first: python -m src.ingestion.pipeline"
        )
    
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before embedding.")
    
    parsed = load_parsed_json(parsed_path)
    logger.info("Loaded %d parsed document(s) from %s", len(parsed), parsed_path)
    if not parsed:
        logger.error("No parsed documents to index")
        return 0
    
    chunks = chunk_documents(parsed, cfg=cfg)
    logger.info("Created %d chunks", len(chunks))
    if not chunks:
        logger.error("Chunking produced no chunks; aborting")
        return 0
    
    if save_chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_chunks_json(chunks, chunks_path)

    embedded = embed_chunks(chunks, cfg=cfg)
    if not embedded:
        logger.error("Embedding produced no vectors; aborting")
        return 0
    
    n = upsert_chunks(embedded, cfg=cfg, recreate=recreate_collection)
    logger.info(
        "Indexing complete: %d points in collection %s",
        n,
        cfg["qdrant_collection"],
    )
    
    return n


if __name__ == "__main__":
    run_indexing_pipeline()
