import logging
import os
from pathlib import Path

import yaml
from langchain_core.documents import Document
from unstructured.partition.html import partition_html

from .robust_extraction import assess_extraction_quality, extract_with_fallback

logger = logging.getLogger(__name__)

SKIP_CATEGORIES = frozenset({"Image", "Figure"})
DEFAULT_MIN_TEXT_LEN = 50

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_ingestion_config() -> dict:
    defaults = {
        "min_text_length": DEFAULT_MIN_TEXT_LEN,
        "skip_categories": sorted(SKIP_CATEGORIES),
        "languages": ["eng"],
        "file_types": [".pdf", ".html", ".htm"],
        "fail_on_quality_issues": False,
        "pdf_strategy": "fast_first",
        "fast_min_char_ratio": 0.05,
        "fast_min_absolute_chars": 500,
        "allow_hi_res_fallback": True,
        "dedupe_repeated_lines": True,
        "dedupe_min_pages": 10,
        "dedupe_min_line_length": 25,
        "dedupe_max_line_length": 220,
        "dedupe_ratio_threshold": 0.6,
    }
    if not _CONFIG_PATH.exists():
        return defaults

    with open(_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    ingestion = config.get("ingestion", {})
    defaults.update(
        {
            "min_text_length": ingestion.get("min_text_length", DEFAULT_MIN_TEXT_LEN),
            "skip_categories": ingestion.get("skip_categories", sorted(SKIP_CATEGORIES)),
            "languages": ingestion.get("languages", ["eng"]),
            "file_types": ingestion.get("file_types", [".pdf", ".html", ".htm"]),
            "fail_on_quality_issues": ingestion.get("fail_on_quality_issues", False),
            "pdf_strategy": ingestion.get("pdf_strategy", "fast_first"),
            "fast_min_char_ratio": ingestion.get("fast_min_char_ratio", 0.05),
            "fast_min_absolute_chars": ingestion.get("fast_min_absolute_chars", 500),
            "allow_hi_res_fallback": ingestion.get("allow_hi_res_fallback", True),
            "dedupe_repeated_lines": ingestion.get("dedupe_repeated_lines", True),
            "dedupe_min_pages": ingestion.get("dedupe_min_pages", 10),
            "dedupe_min_line_length": ingestion.get("dedupe_min_line_length", 25),
            "dedupe_max_line_length": ingestion.get("dedupe_max_line_length", 220),
            "dedupe_ratio_threshold": ingestion.get("dedupe_ratio_threshold", 0.6),
        }
    )
    return defaults


def _should_skip_element(element, skip_categories: frozenset, min_text_len: int) -> bool:
    if element.category in skip_categories:
        return True
    text = (element.text or "").strip()
    return len(text) < min_text_len


def _element_to_document(element, file_path: str, file_type: str) -> Document:
    return Document(
        page_content=element.text or "",
        metadata={
            "source": file_path,
            "file_type": file_type,
            "page_number": element.metadata.page_number,
            "type": element.category,
        },
    )


def _partition_file(file_path: str, cfg: dict) -> tuple[list, str, str]:
    if file_path.lower().endswith(".pdf"):
        elements, strategy = extract_with_fallback(
            file_path,
            pdf_strategy=cfg["pdf_strategy"],
            fast_min_char_ratio=cfg["fast_min_char_ratio"],
            fast_min_absolute_chars=cfg["fast_min_absolute_chars"],
            allow_hi_res_fallback=cfg["allow_hi_res_fallback"],
        )
        return elements, "pdf", strategy
    if file_path.lower().endswith((".html", ".htm")):
        return (
            partition_html(
                filename=file_path,
                languages=cfg["languages"],
                detect_language_per_element=False,
            ),
            "html",
            "html",
        )
    raise ValueError(f"Unsupported file type: {file_path}")


def load_file(file_path: str, cfg: dict | None = None) -> tuple[list[Document], dict]:
    """Partition one file, assess quality on raw elements, return kept documents."""
    cfg = cfg or load_ingestion_config()
    skip_categories = frozenset(cfg["skip_categories"])
    min_text_len = cfg["min_text_length"]

    elements, file_type, strategy = _partition_file(file_path, cfg)
    quality = assess_extraction_quality(elements)
    quality["source"] = file_path
    quality["strategy"] = strategy

    docs = []
    skipped = 0
    for element in elements:
        if _should_skip_element(element, skip_categories, min_text_len):
            skipped += 1
            continue
        docs.append(_element_to_document(element, file_path, file_type))

    quality["kept_elements"] = len(docs)
    quality["skipped_elements"] = skipped

    logger.info(
        "Loaded %s: %d kept, %d skipped (strategy=%s, raw elements: %d, quality: %s)",
        os.path.basename(file_path),
        len(docs),
        skipped,
        strategy,
        quality["total_elements"],
        quality["status"],
    )
    if quality["status"] == "failed":
        logger.warning(
            "Extraction quality issues for %s: %s",
            os.path.basename(file_path),
            ", ".join(quality["issues"]),
        )

    return docs, quality


def load_documents(file_paths: list[str]) -> tuple[list[Document], dict[str, dict]]:
    """Load only the given validated file paths. Returns all docs and per-file quality."""
    if not file_paths:
        logger.warning("No files to load")
        return [], {}

    cfg = load_ingestion_config()
    results: list[Document] = []
    quality_by_file: dict[str, dict] = {}
    failed_files: list[dict] = []

    for file_path in file_paths:
        try:
            docs, quality = load_file(file_path, cfg)
            quality_by_file[file_path] = quality
            if cfg["fail_on_quality_issues"] and quality["status"] == "failed":
                logger.error(
                    "Skipping output from %s due to quality failure (fail_on_quality_issues=true)",
                    os.path.basename(file_path),
                )
                continue
            results.extend(docs)
        except Exception as e:
            failed_files.append({"file": file_path, "error": str(e)})
            logger.error("Failed to load %s: %s", file_path, e)

    if failed_files:
        logger.warning("%d file(s) failed to load", len(failed_files))

    return results, quality_by_file
