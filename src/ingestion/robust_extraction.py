import logging
import os
import subprocess

import PyPDF2
from pdfminer.high_level import extract_text
from unstructured.documents.elements import ElementMetadata, Text
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

_PDF_KWARGS = {"languages": ["eng"], "detect_language_per_element": False, "extract_images_in_pdf": False}
_HTML_EXT = {".html", ".htm"}
_SCANNED_THRESHOLD = 50


def _mime_type(path: str) -> str | None:
    try:
        import magic
        return magic.from_file(path, mime=True)
    except ImportError:
        logger.debug("python-magic not installed; using system 'file' command")
    except OSError as e:
        logger.warning("magic.from_file failed for %s: %s", path, e)

    try:
        out = subprocess.run(
            ["file", "--mime-type", path],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().split(": ")[-1]
    except FileNotFoundError:
        logger.error("'file' command not found; cannot detect MIME type for %s", path)
        return None
    except subprocess.CalledProcessError as e:
        logger.warning("file --mime-type failed for %s: %s", path, e)
        return None


def validation_file(path: str, allowed: set[str] | None = None) -> dict:
    if not os.path.exists(path):
        return {"status": "skip", "reason": "missing"}
    if os.path.getsize(path) == 0:
        return {"status": "skip", "reason": "empty"}

    ext = os.path.splitext(path)[1].lower()
    allowed = allowed or {".pdf", ".html", ".htm"}
    if ext not in allowed:
        return {"status": "skip", "reason": f"unsupported_extension:{ext}"}

    if ext in _HTML_EXT:
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, encoding=enc) as f:
                    f.read(4096)
                return {"status": "ok"}
            except UnicodeDecodeError:
                continue
            except OSError as e:
                return {"status": "skip", "reason": f"unreadable:{e}"}
        return {"status": "skip", "reason": "unreadable_encoding"}

    mime = _mime_type(path)
    if mime is None:
        return {"status": "skip", "reason": "mime_detection_failed"}
    if mime != "application/pdf":
        return {"status": "skip", "reason": f"unexpected_mime:{mime}"}

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                return {"status": "skip", "reason": "encrypted"}
    except PyPDF2.errors.PdfReadError as e:
        return {"status": "skip", "reason": f"corrupted_pdf:{e}"}
    except OSError as e:
        return {"status": "skip", "reason": f"unreadable:{e}"}
    except Exception as e:
        return {"status": "skip", "reason": f"corrupted:{e}"}
    return {"status": "ok"}


def collect_valid_files(raw_dir: str, file_types: list[str] | None = None) -> list[str]:
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    allowed = {t.lower() for t in (file_types or [".pdf", ".html", ".htm"])}
    valid = []
    try:
        names = sorted(os.listdir(raw_dir))
    except OSError as e:
        raise OSError(f"Cannot list directory {raw_dir}: {e}") from e

    for name in names:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in allowed:
            logger.debug("Skipping %s: extension not in file_types", name)
            continue
        result = validation_file(path, allowed)
        if result["status"] == "ok":
            valid.append(path)
        else:
            logger.warning("Skipping %s: %s", name, result["reason"])
    return valid


def _pdf_chars(path: str) -> int:
    try:
        return len(extract_text(path).strip())
    except Exception as e:
        logger.warning("pdfminer text probe failed for %s: %s", os.path.basename(path), e)
        return 0


def _pypdf_pages(path: str) -> list:
    elements = []
    try:
        reader = PyPDF2.PdfReader(path)
    except PyPDF2.errors.PdfReadError as e:
        raise RuntimeError(f"pypdf cannot read PDF: {e}") from e

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(
                "pypdf failed on page %d of %s: %s",
                page_num,
                os.path.basename(path),
                e,
            )
            continue
        if text.strip():
            elements.append(Text(text=text, metadata=ElementMetadata(page_number=page_num)))
    return elements


def _partition_pdf(path: str, strategy: str) -> list:
    try:
        if strategy == "ocr_only":
            return partition_pdf(
                filename=path,
                strategy="ocr_only",
                ocr_languages="eng",
                **_PDF_KWARGS,
            )
        if strategy == "fast":
            return partition_pdf(
                filename=path,
                strategy="fast",
                infer_table_structure=False,
                **_PDF_KWARGS,
            )
        return partition_pdf(filename=path, strategy=strategy, **_PDF_KWARGS)
    except Exception as e:
        raise RuntimeError(f"partition_pdf({strategy}) failed: {e}") from e


def _enough_text(elements, path: str, cfg: dict) -> bool:
    chars = sum(len(e.text or "") for e in elements)
    if len(elements) < 5:
        return False
    baseline = _pdf_chars(path)
    need = cfg["fast_min_absolute_chars"]
    if baseline > _SCANNED_THRESHOLD:
        need = max(need, int(baseline * cfg["fast_min_char_ratio"]))
    return chars >= need


def extract_with_fallback(path: str, cfg: dict) -> tuple[list, str]:
    name = os.path.basename(path)
    strategy = cfg["pdf_strategy"]

    if strategy == "ocr_only" or (strategy == "fast_first" and _pdf_chars(path) < _SCANNED_THRESHOLD):
        logger.info("Using ocr_only for %s", name)
        return _partition_pdf(path, "ocr_only"), "ocr_only"

    if strategy in ("fast", "hi_res", "auto"):
        logger.info("Using %s for %s", strategy, name)
        return _partition_pdf(path, strategy), strategy

    logger.info("Trying fast for %s", name)
    try:
        fast = _partition_pdf(path, "fast")
    except RuntimeError as e:
        logger.warning("Fast extraction failed for %s: %s", name, e)
        fast = []

    if fast and assess_extraction_quality(fast)["status"] == "passed" and _enough_text(fast, path, cfg):
        return fast, "fast"

    if _pdf_chars(path) > _SCANNED_THRESHOLD:
        logger.info("Trying pypdf for %s", name)
        try:
            pypdf = _pypdf_pages(path)
        except RuntimeError as e:
            logger.error("Pypdf extraction failed for %s: %s", name, e)
            pypdf = []
        if pypdf and _enough_text(pypdf, path, cfg):
            q = assess_extraction_quality(pypdf)
            logger.info("Pypdf ok for %s (%d chars, %d elements)", name, q["total_chars"], q["total_elements"])
            return pypdf, "pypdf"

    if not cfg["allow_hi_res_fallback"]:
        raise RuntimeError(
            f"Insufficient text from {name}. Enable allow_hi_res_fallback or fix the PDF."
        )

    logger.warning("Falling back to hi_res for %s (slow)", name)
    return _partition_pdf(path, "hi_res"), "hi_res"


def assess_extraction_quality(elements) -> dict:
    total_chars = sum(len(e.text or "") for e in elements)
    n = len(elements)
    issues = []
    if total_chars < 100:
        issues.append("very_low_text_content")
    if n and sum(1 for e in elements if e.category == "Image") > 0.5 * n:
        issues.append("high_image_ratio_possible_ocr_failure")
    if n < 5:
        issues.append("very_few_elements_possible_extraction_failure")
    return {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "total_chars": total_chars,
        "total_elements": n,
    }
