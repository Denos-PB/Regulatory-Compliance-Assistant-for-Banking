"""End-to-end pipeline: ingest → index → optional ask."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import load_data_config, load_ingestion_config, load_indexing_config
from .ingestion.loader import load_documents
from .ingestion.parser import parse_documents
from .ingestion.robust_extraction import collect_valid_files
from .indexing.chunk import chunk_documents, load_parsed_json, save_chunks_json
from .indexing.embedder import embed_chunks
from .indexing.store import upsert_chunks
from .logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)


def _save_json(path: str | Path, data) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"Failed to write {path}: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize JSON for {path}: {e}") from e


def run_ingestion(
    raw_dir: str,
    output_dir: str | Path,
) -> list:
    """Raw PDF/HTML → parsed_document.json + extraction_quality.json."""
    output_dir = Path(output_dir)
    logger.info("Starting ingestion")

    try:
        cfg = load_ingestion_config()
    except Exception as e:
        logger.exception("Failed to load ingestion config")
        raise RuntimeError("Ingestion aborted: config load failed") from e

    try:
        valid = collect_valid_files(raw_dir, cfg["file_types"])
    except FileNotFoundError:
        logger.error("Raw directory not found: %s", raw_dir)
        raise
    except OSError as e:
        logger.error("Cannot access raw directory %s: %s", raw_dir, e)
        raise

    try:
        total = sum(1 for n in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, n)))
    except OSError as e:
        logger.warning("Could not count files in %s: %s", raw_dir, e)
        total = len(valid)

    logger.info("Valid files: %d out of %d total", len(valid), total)
    if not valid:
        logger.error("No valid files to process in %s", raw_dir)
        return []

    raw_docs, quality_by_file = load_documents(valid, cfg)
    logger.info("Loaded %d raw documents from %d file(s)", len(raw_docs), len(valid))

    extraction_errors = [p for p, r in quality_by_file.items() if r.get("status") == "error"]
    quality_failures = [p for p, r in quality_by_file.items() if r.get("status") == "failed"]

    if extraction_errors:
        logger.error(
            "%d file(s) failed extraction: %s",
            len(extraction_errors),
            ", ".join(os.path.basename(p) for p in extraction_errors),
        )
    if quality_failures:
        logger.warning(
            "%d file(s) had quality issues: %s",
            len(quality_failures),
            ", ".join(os.path.basename(p) for p in quality_failures),
        )

    if not raw_docs:
        logger.error("No documents extracted; saving quality report only")

    parsed = parse_documents(raw_docs, cfg)
    logger.info("Parsed %d documents", len(parsed))

    short = sum(1 for d in parsed if len(d.page_content.strip()) < cfg["min_text_length"])
    if short:
        logger.warning("%d documents have very short content (≤%d chars)", short, cfg["min_text_length"])

    out = output_dir / "parsed_document.json"
    report = output_dir / "extraction_quality.json"

    _save_json(out, [{"text": d.page_content, "metadata": d.metadata} for d in parsed])
    _save_json(report, {os.path.basename(p): r for p, r in quality_by_file.items()})

    logger.info("Saved %d parsed documents to %s", len(parsed), out)
    logger.info("Saved extraction quality report to %s", report)
    return parsed


def run_indexing(
    parsed_path: str | Path,
    output_dir: str | Path,
    *,
    save_chunks: bool = False,
    recreate_collection: bool = False,
) -> int:
    """parsed_document.json → chunk → embed → Qdrant."""
    output_dir = Path(output_dir)
    parsed_path = Path(parsed_path)
    chunks_path = output_dir / "chunks.json"

    logger.info("Starting indexing")

    try:
        cfg = load_indexing_config()
    except Exception as e:
        logger.exception("Failed to load indexing config")
        raise RuntimeError("Indexing aborted: config load failed") from e

    if not parsed_path.exists():
        raise FileNotFoundError(
            f"Missing {parsed_path}. Run: python -m src.cli run"
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
        save_chunks_json(chunks, chunks_path)

    embedded = embed_chunks(chunks, cfg=cfg)
    if not embedded:
        logger.error("Embedding produced no vectors; aborting")
        return 0

    n = upsert_chunks(embedded, cfg=cfg, recreate=recreate_collection)
    logger.info("Indexed %d points into collection %s", n, cfg["qdrant_collection"])
    return n


def run_all(
    raw_dir: str | None = None,
    output_dir: str | None = None,
    *,
    question: str | None = None,
    recreate_collection: bool = False,
    save_chunks: bool = False,
    skip_ingest: bool = False,
    skip_index: bool = False,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> dict[str, Any]:
    """
    Run the full RAG pipeline in one call.

    1. Ingestion — raw PDF/HTML → parsed_document.json
    2. Indexing — chunk → embed → Qdrant
    3. Optional — answer a question with RAG
    """
    paths = load_data_config()
    raw_dir = raw_dir or paths["raw_path"]
    output_dir = Path(output_dir or paths["processed_path"])
    setup_logging(log_dir=paths["logs"])

    parsed_path = output_dir / "parsed_document.json"

    summary: dict[str, Any] = {
        "raw_dir": raw_dir,
        "output_dir": str(output_dir),
        "ingested_documents": 0,
        "indexed_points": 0,
        "question": question,
        "answer": None,
        "sources": [],
    }

    if not skip_ingest:
        logger.info("=== Step 1/3: Ingestion ===")
        parsed = run_ingestion(raw_dir, output_dir)
        summary["ingested_documents"] = len(parsed)
        if not parsed:
            raise RuntimeError(
                f"Ingestion produced no documents from {raw_dir}. "
                "Add PDF/HTML files or check extraction_quality.json."
            )
    elif not parsed_path.exists():
        raise FileNotFoundError(
            f"skip_ingest=True but {parsed_path} is missing. Run: python -m src.cli run"
        )

    if not skip_index:
        logger.info("=== Step 2/3: Indexing ===")
        n = run_indexing(
            parsed_path,
            output_dir,
            save_chunks=save_chunks,
            recreate_collection=recreate_collection,
        )
        summary["indexed_points"] = n
        if n == 0:
            raise RuntimeError("Indexing produced no vectors in Qdrant. Check Qdrant is running.")
    elif question:
        logger.warning("skip_index=True: using existing Qdrant collection for ask")

    if question and question.strip():
        logger.info("=== Step 3/3: Ask ===")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env before generation.")

        from .rag.chain import answer

        result = answer(question.strip(), top_k=top_k, source_filter=source_filter)
        summary["answer"] = result.get("answer")
        summary["sources"] = result.get("sources") or []

    logger.info("Pipeline complete: %s", summary)
    return summary
