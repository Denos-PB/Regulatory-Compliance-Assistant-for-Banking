import logging
import re
import unicodedata

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
_DEDUPE_KEYS = (
    "dedupe_repeated_lines",
    "dedupe_min_pages",
    "dedupe_min_line_length",
    "dedupe_max_line_length",
    "dedupe_ratio_threshold",
)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\x00", ""))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r" {2,}", " ", text).strip()


def _norm_line(line: str) -> str:
    line = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line or "")).strip()
    line = re.sub(r"\bpage\s+\d+\b", "page #", line, flags=re.I)
    return re.sub(r"\b\d+\b", "#", line).lower()


def _headers(text: str) -> list:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+(\.\d+)*\s+", line) or (line.isupper() and len(line) < 100):
            out.append(line)
        elif not line.endswith(".") and len(line) < 100 and line[0].isupper():
            out.append(line)
    return out


def _repeated_lines(docs: list, cfg: dict) -> dict[str, set[str]]:
    by_source: dict[str, list[set[str]]] = {}
    lo, hi = cfg["dedupe_min_line_length"], cfg["dedupe_max_line_length"]

    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        norm = {_norm_line(l) for l in doc.page_content.split("\n") if lo <= len(l.strip()) <= hi}
        by_source.setdefault(src, []).append(norm)

    repeated = {}
    for src, pages in by_source.items():
        if len(pages) < cfg["dedupe_min_pages"]:
            repeated[src] = set()
            continue
        counts: dict[str, int] = {}
        for page in pages:
            for line in page:
                counts[line] = counts.get(line, 0) + 1
        need = max(2, int(len(pages) * cfg["dedupe_ratio_threshold"]))
        repeated[src] = {line for line, c in counts.items() if c >= need}
    return repeated


def parse_document(doc: Document, repeated_lines: set[str] | None = None) -> Document:
    text, meta = doc.page_content, dict(doc.metadata)

    if repeated_lines:
        lines, removed = [], 0
        for line in text.split("\n"):
            if line.strip() and _norm_line(line.strip()) in repeated_lines:
                removed += 1
            else:
                lines.append(line)
        text = "\n".join(lines)
        if removed:
            meta["removed_repeated_lines"] = removed

    cleaned = _clean(text)
    if meta.get("type") in ("Title", "Header"):
        meta["is_header"] = True
    meta["char_count"] = len(cleaned)
    meta["token_estimate"] = len(cleaned) // 4
    if meta.get("type", "Text") == "Text":
        meta["section_headers"] = _headers(cleaned)
    return Document(page_content=cleaned, metadata=meta)


def parse_documents(docs: list, cfg: dict | None = None) -> list:
    from ..config import INGESTION_DEFAULTS

    c = {**INGESTION_DEFAULTS, **(cfg or {})}
    dedupe_cfg = {k: c[k] for k in _DEDUPE_KEYS}

    repeated: dict[str, set[str]] = {}
    if dedupe_cfg["dedupe_repeated_lines"]:
        try:
            repeated = _repeated_lines(docs, dedupe_cfg)
            n = sum(len(v) for v in repeated.values())
            if n:
                logger.info("Detected %d repeated header/footer patterns", n)
        except Exception as e:
            logger.exception("Header/footer dedupe failed; continuing without dedupe: %s", e)
            repeated = {}

    parsed, failed = [], 0
    for doc in docs:
        try:
            src = doc.metadata.get("source", "unknown")
            parsed.append(parse_document(doc, repeated_lines=repeated.get(src, set())))
        except Exception as e:
            failed += 1
            logger.exception("Failed to parse document from %s", doc.metadata.get("source", "unknown"))

    if failed:
        logger.warning("%d document(s) failed to parse", failed)
    return parsed
