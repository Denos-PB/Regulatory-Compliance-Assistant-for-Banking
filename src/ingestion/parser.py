import re
import unicodedata
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_PARSER_CONFIG = {
    "dedupe_repeated_lines": True,
    "dedupe_min_pages": 10,
    "dedupe_min_line_length": 25,
    "dedupe_max_line_length": 220,
    "dedupe_ratio_threshold": 0.6,
}

def clean_text(text:str) -> str:
    text = text.replace('\x00', '')
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\n{3,}','\n\n',text)
    text = re.sub(r' {2,}', ' ',text)
    text = text.strip()

    return text


def _normalize_line_for_compare(line: str) -> str:
    line = unicodedata.normalize("NFKC", line or "")
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\bpage\s+\d+\b", "page #", line, flags=re.IGNORECASE)
    line = re.sub(r"\b\d+\b", "#", line)
    return line.lower()

def extract_section_headers(text:str) -> list:
    headers =[]
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+(\.\d+)*\s+', line):
            headers.append(line)
        elif line.isupper() and len(line) < 100:
            headers.append(line)
        elif not line.endswith('.') and len(line)<100 and line[0].isupper():
            headers.append(line)

    return headers


def _collect_repeated_lines_by_source(docs: list, cfg: dict) -> dict[str, set[str]]:
    per_source_page_lines: dict[str, list[set[str]]] = {}

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        lines = [line.strip() for line in doc.page_content.split("\n") if line.strip()]
        normalized = {
            _normalize_line_for_compare(line)
            for line in lines
            if cfg["dedupe_min_line_length"] <= len(line) <= cfg["dedupe_max_line_length"]
        }
        per_source_page_lines.setdefault(source, []).append(normalized)

    repeated_by_source: dict[str, set[str]] = {}
    for source, pages in per_source_page_lines.items():
        page_count = len(pages)
        if page_count < cfg["dedupe_min_pages"]:
            repeated_by_source[source] = set()
            continue

        counts: dict[str, int] = {}
        for page_set in pages:
            for norm_line in page_set:
                counts[norm_line] = counts.get(norm_line, 0) + 1

        threshold = max(2, int(page_count * cfg["dedupe_ratio_threshold"]))
        repeated_by_source[source] = {
            line for line, count in counts.items() if count >= threshold
        }
    return repeated_by_source

def parse_document(doc:Document, repeated_lines: set[str] | None = None) -> Document:
    original_text = doc.page_content
    metadata = dict(doc.metadata)
    doc_type = metadata.get("type", "Text")

    if repeated_lines:
        kept_lines = []
        removed = 0
        for line in original_text.split("\n"):
            stripped = line.strip()
            if stripped and _normalize_line_for_compare(stripped) in repeated_lines:
                removed += 1
                continue
            kept_lines.append(line)
        original_text = "\n".join(kept_lines)
        if removed:
            metadata["removed_repeated_lines"] = removed

    if doc_type == "Table":
        cleaned = clean_text(original_text)

    else:
        cleaned = clean_text(original_text)
        if doc_type in ("Title", "Header"):
            metadata["is_header"] = True

    metadata["char_count"] = len(cleaned)
    metadata["token_estimate"] = len(cleaned) // 4
    if doc_type == "Text":
        metadata["section_headers"] = extract_section_headers(cleaned)

    return Document(page_content=cleaned, metadata=metadata)


def parse_documents(docs:list, parser_config: dict | None = None)-> list:
    cfg = {**DEFAULT_PARSER_CONFIG, **(parser_config or {})}
    parsed = []
    failed = 0
    repeated_by_source: dict[str, set[str]] = {}

    if cfg["dedupe_repeated_lines"]:
        repeated_by_source = _collect_repeated_lines_by_source(docs, cfg)
        total_patterns = sum(len(v) for v in repeated_by_source.values())
        if total_patterns:
            logger.info("Detected %d repeated header/footer patterns", total_patterns)

    for doc in docs:
        try:
            source = doc.metadata.get("source", "unknown")
            repeated_lines = repeated_by_source.get(source, set())
            parsed_doc = parse_document(doc, repeated_lines=repeated_lines)
            parsed.append(parsed_doc)
        except Exception as e:
            failed += 1
            source = doc.metadata.get('source','unknown')
            logger.error(f"Failed to parse document {source}: {e}")
    if failed:
        logger.warning(f"{failed} document(s) failed to parse")
    
    return parsed