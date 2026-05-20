import json
import logging
import os

from ..logging_config import setup_logging
from .loader import load_documents, load_ingestion_config
from .parser import parse_documents
from .robust_extraction import collect_valid_files

logger = logging.getLogger(__name__)


def run_pipeline(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    setup_logging()
    logger.info("Starting ingestion pipeline")

    cfg = load_ingestion_config()
    file_types = cfg["file_types"]
    min_text_len = cfg["min_text_length"]

    valid_files = collect_valid_files(raw_dir, file_types=file_types)
    total_in_dir = sum(
        1
        for name in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, name))
    )
    logger.info("Valid files: %d out of %d total", len(valid_files), total_in_dir)

    if not valid_files:
        logger.error("No valid files to process in %s", raw_dir)
        return []

    raw_docs, quality_by_file = load_documents(valid_files)
    logger.info("Loaded %d raw documents from %d file(s)", len(raw_docs), len(valid_files))

    failed_quality = [
        path for path, report in quality_by_file.items() if report["status"] == "failed"
    ]
    if failed_quality:
        logger.warning(
            "%d file(s) had extraction quality issues: %s",
            len(failed_quality),
            ", ".join(os.path.basename(p) for p in failed_quality),
        )

    parser_cfg = {
        "dedupe_repeated_lines": cfg.get("dedupe_repeated_lines", True),
        "dedupe_min_pages": cfg.get("dedupe_min_pages", 10),
        "dedupe_min_line_length": cfg.get("dedupe_min_line_length", 25),
        "dedupe_max_line_length": cfg.get("dedupe_max_line_length", 220),
        "dedupe_ratio_threshold": cfg.get("dedupe_ratio_threshold", 0.6),
    }
    parsed_docs = parse_documents(raw_docs, parser_config=parser_cfg)
    logger.info("Parsed %d documents", len(parsed_docs))

    quality_issues = sum(
        1 for doc in parsed_docs if len(doc.page_content.strip()) < min_text_len
    )
    if quality_issues > 0:
        logger.warning(
            "%d documents have very short content (≤%d chars) after parsing",
            quality_issues,
            min_text_len,
        )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "parsed_document.json")
    output_data = [
        {"text": doc.page_content, "metadata": doc.metadata} for doc in parsed_docs
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    quality_report_path = os.path.join(output_dir, "extraction_quality.json")
    with open(quality_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {os.path.basename(path): report for path, report in quality_by_file.items()},
            f,
            indent=2,
        )

    logger.info("Saved %d parsed documents to %s", len(parsed_docs), output_file)
    logger.info("Saved extraction quality report to %s", quality_report_path)

    return parsed_docs


if __name__ == "__main__":
    run_pipeline()
