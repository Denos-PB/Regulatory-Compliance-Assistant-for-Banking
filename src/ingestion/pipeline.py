import json
import logging
import os

from ..logging_config import setup_logging
from ..config import load_ingestion_config
from .loader import load_documents
from .parser import parse_documents
from .robust_extraction import collect_valid_files

logger = logging.getLogger(__name__)


def _save_json(path: str, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise OSError(f"Failed to write {path}: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize JSON for {path}: {e}") from e


def run_pipeline(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    setup_logging()
    logger.info("Starting ingestion pipeline")

    try:
        cfg = load_ingestion_config()
    except Exception as e:
        logger.exception("Failed to load ingestion config")
        raise RuntimeError("Pipeline aborted: config load failed") from e

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

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "parsed_document.json")
    report = os.path.join(output_dir, "extraction_quality.json")

    _save_json(out, [{"text": d.page_content, "metadata": d.metadata} for d in parsed])
    _save_json(report, {os.path.basename(p): r for p, r in quality_by_file.items()})

    logger.info("Saved %d parsed documents to %s", len(parsed), out)
    logger.info("Saved extraction quality report to %s", report)
    return parsed


if __name__ == "__main__":
    run_pipeline()
