import logging
import os
from langchain_core.documents import Document
from unstructured.partition.html import partition_html
from ..config import load_ingestion_config
from .robust_extraction import assess_extraction_quality, extract_with_fallback

logger = logging.getLogger(__name__)
_SKIP = frozenset({"Image", "Figure"})

def _error_quality(path: str, error: str, strategy: str | None = None) -> dict:
    return {
        "status": "error",
        "issues": ["extraction_error"],
        "error": error,
        "source": path,
        "strategy": strategy,
        "total_chars": 0,
        "total_elements": 0,
        "kept_elements": 0,
        "skipped_elements": 0,
    }


def _partition(path: str, cfg: dict) -> tuple[list, str, str]:
    if path.lower().endswith(".pdf"):
        elements, strategy = extract_with_fallback(path, cfg)
        return elements, "pdf", strategy
    if path.lower().endswith((".html", ".htm")):
        try:
            elements = partition_html(
                filename=path,
                languages=cfg["languages"],
                detect_language_per_element=False,
            )
        except Exception as e:
            raise RuntimeError(f"partition_html failed: {e}") from e
        return elements, "html", "html"
    raise ValueError(f"Unsupported file type: {path}")


def _element_metadata(el, path: str, file_type: str) -> dict:
    page_number = getattr(getattr(el, "metadata", None), "page_number", None)
    return {
        "source": path,
        "file_type": file_type,
        "page_number": page_number,
        "type": getattr(el, "category", "Text"),
    }


def load_file(path: str, cfg: dict) -> tuple[list[Document], dict]:
    """Load one file. On failure returns empty docs and an error quality report."""
    skip = frozenset(cfg.get("skip_categories", _SKIP))
    min_len = cfg["min_text_length"]

    try:
        elements, file_type, strategy = _partition(path, cfg)
    except Exception as e:
        logger.exception("Extraction failed for %s", path)
        return [], _error_quality(path, str(e))

    quality = assess_extraction_quality(elements)
    quality.update(source=path, strategy=strategy, kept_elements=0, skipped_elements=0)

    kept, skipped = [], 0
    for el in elements:
        try:
            text = (el.text or "").strip()
            if el.category in skip or len(text) < min_len:
                skipped += 1
                continue
            kept.append(Document(page_content=el.text or "", metadata=_element_metadata(el, path, file_type)))
        except Exception as e:
            skipped += 1
            logger.warning("Skipping element in %s: %s", os.path.basename(path), e)

    quality["kept_elements"] = len(kept)
    quality["skipped_elements"] = skipped

    if quality["status"] == "failed":
        logger.warning(
            "Quality issues for %s: %s",
            os.path.basename(path),
            ", ".join(quality["issues"]),
        )
    logger.info(
        "Loaded %s: %d kept, %d skipped (strategy=%s, quality=%s)",
        os.path.basename(path),
        len(kept),
        skipped,
        strategy,
        quality["status"],
    )
    return kept, quality


def load_documents(file_paths: list[str], cfg: dict | None = None) -> tuple[list[Document], dict[str, dict]]:
    if not file_paths:
        logger.warning("No files to load")
        return [], {}

    cfg = cfg or load_ingestion_config()
    docs: list[Document] = []
    quality_by_file: dict[str, dict] = {}
    load_errors = 0

    for path in file_paths:
        kept, quality = load_file(path, cfg)
        quality_by_file[path] = quality

        if quality["status"] == "error":
            load_errors += 1
            continue

        if cfg["fail_on_quality_issues"] and quality["status"] == "failed":
            logger.error(
                "Skipping output from %s (fail_on_quality_issues=true)",
                os.path.basename(path),
            )
            continue

        docs.extend(kept)

    if load_errors:
        logger.error("%d file(s) failed during extraction", load_errors)
    return docs, quality_by_file
