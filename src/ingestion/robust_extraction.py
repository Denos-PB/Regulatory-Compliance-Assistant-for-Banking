import logging
import os
import subprocess

import PyPDF2
from pdfminer.high_level import extract_text
from unstructured.documents.elements import ElementMetadata, Text
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

_COMMON_PDF_KWARGS = {
    "languages": ["eng"],
    "detect_language_per_element": False,
    "extract_images_in_pdf": False,
}

_HTML_EXTENSIONS = {".html", ".htm"}
_PDF_EXTENSION = ".pdf"

# Scanned if pdfminer finds almost no text
_SCANNED_CHAR_THRESHOLD = 50


def get_mime_type(filepath):
    try:
        import magic
        return magic.from_file(filepath, mime=True)
    except (ImportError, OSError):
        result = subprocess.run(
            ["file", "--mime-type", filepath],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split(": ")[-1]


def validation_file(filepath: str, file_types: list[str] | None = None) -> dict:
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {"status": "skip", "reason": "empty_or_missing"}

    ext = os.path.splitext(filepath)[1].lower()
    allowed = {t.lower() for t in (file_types or [".pdf", ".html", ".htm"])}
    if ext not in allowed:
        return {"status": "skip", "reason": f"unsupported_extension:{ext}"}

    if ext in _HTML_EXTENSIONS:
        try:
            with open(filepath, encoding="utf-8") as f:
                f.read(4096)
        except UnicodeDecodeError:
            try:
                with open(filepath, encoding="latin-1") as f:
                    f.read(4096)
            except OSError as e:
                return {"status": "skip", "reason": f"unreadable:{e}"}
        except OSError as e:
            return {"status": "skip", "reason": f"unreadable:{e}"}
        return {"status": "ok"}

    if ext == _PDF_EXTENSION:
        mime_type = get_mime_type(filepath)
        if mime_type != "application/pdf":
            return {"status": "skip", "reason": f"unexpected_mime:{mime_type}"}
        try:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    return {"status": "skip", "reason": "encrypted"}
        except Exception as e:
            return {"status": "skip", "reason": f"corrupted:{str(e)}"}
        return {"status": "ok"}

    return {"status": "skip", "reason": f"unsupported_extension:{ext}"}


def collect_valid_files(raw_dir: str, file_types: list[str] | None = None) -> list[str]:
    """Return paths under raw_dir that pass validation_file."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    allowed = {t.lower() for t in (file_types or [".pdf", ".html", ".htm"])}
    valid_paths = []

    for filename in sorted(os.listdir(raw_dir)):
        file_path = os.path.join(raw_dir, filename)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed:
            logger.debug("Skipping %s: extension not in file_types", filename)
            continue

        validation = validation_file(file_path, file_types=allowed)
        if validation["status"] == "ok":
            valid_paths.append(file_path)
        else:
            logger.warning("Skipping %s: %s", filename, validation["reason"])

    return valid_paths


def pdfminer_char_count(filepath: str) -> int:
    try:
        return len(extract_text(filepath).strip())
    except Exception:
        return 0


def is_scaned_pdf(filepath: str) -> bool:
    return pdfminer_char_count(filepath) < _SCANNED_CHAR_THRESHOLD


def element_text_chars(elements) -> int:
    return sum(len(e.text or "") for e in elements)


def partition_pdf_pypdf(filepath: str) -> list:
    """Extract text per page with pypdf when unstructured fast returns nothing."""
    elements = []
    reader = PyPDF2.PdfReader(filepath)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        elements.append(
            Text(text=text, metadata=ElementMetadata(page_number=page_num))
        )
    return elements


def fast_extraction_is_sufficient(
    elements,
    filepath: str,
    *,
    min_char_ratio: float = 0.05,
    min_absolute_chars: int = 500,
) -> bool:
    """True when fast-path output has enough text vs pdfminer baseline."""
    elem_chars = element_text_chars(elements)
    if len(elements) < 5:
        return False

    pdf_chars = pdfminer_char_count(filepath)
    if pdf_chars > _SCANNED_CHAR_THRESHOLD:
        threshold = max(min_absolute_chars, int(pdf_chars * min_char_ratio))
    else:
        threshold = min_absolute_chars

    return elem_chars >= threshold


def extract_with_fallback(
    filepath: str,
    *,
    infer_table_structure: bool = True,
    pdf_strategy: str = "fast_first",
    fast_min_char_ratio: float = 0.05,
    fast_min_absolute_chars: int = 500,
    allow_hi_res_fallback: bool = True,
) -> tuple[list, str]:
    """
    Partition a PDF and return (elements, strategy_used).

    pdf_strategy:
      - fast_first: try fast, fall back to hi_res if yield is low (default)
      - fast / hi_res / auto / ocr_only: single strategy, no fallback
    """
    basename = os.path.basename(filepath)

    if pdf_strategy == "ocr_only" or (pdf_strategy == "fast_first" and is_scaned_pdf(filepath)):
        logger.info("Scanned or low-text PDF — using ocr_only for %s", basename)
        return (
            partition_pdf(
                filename=filepath,
                strategy="ocr_only",
                ocr_languages="eng",
                **_COMMON_PDF_KWARGS,
            ),
            "ocr_only",
        )

    if pdf_strategy in ("fast", "hi_res", "auto"):
        logger.info("Using %s strategy for %s", pdf_strategy, basename)
        kwargs = {"filename": filepath, "strategy": pdf_strategy, **_COMMON_PDF_KWARGS}
        if pdf_strategy == "hi_res":
            kwargs["infer_table_structure"] = infer_table_structure
        return partition_pdf(**kwargs), pdf_strategy

    # fast_first (default for digital PDFs)
    logger.info("Trying fast strategy for %s", basename)
    fast_elements = partition_pdf(
        filename=filepath,
        strategy="fast",
        infer_table_structure=False,
        **_COMMON_PDF_KWARGS,
    )
    fast_quality = assess_extraction_quality(fast_elements)
    sufficient = fast_extraction_is_sufficient(
        fast_elements,
        filepath,
        min_char_ratio=fast_min_char_ratio,
        min_absolute_chars=fast_min_absolute_chars,
    )

    if fast_quality["status"] == "passed" and sufficient:
        logger.info(
            "Fast strategy sufficient for %s (%d chars, %d elements)",
            basename,
            fast_quality["total_chars"],
            fast_quality["total_elements"],
        )
        return fast_elements, "fast"

    pdf_chars = pdfminer_char_count(filepath)
    if pdf_chars > _SCANNED_CHAR_THRESHOLD:
        logger.info(
            "Unstructured fast returned little text for %s; trying pypdf per-page extraction",
            basename,
        )
        pypdf_elements = partition_pdf_pypdf(filepath)
        pypdf_quality = assess_extraction_quality(pypdf_elements)
        if fast_extraction_is_sufficient(
            pypdf_elements,
            filepath,
            min_char_ratio=fast_min_char_ratio,
            min_absolute_chars=fast_min_absolute_chars,
        ):
            logger.info(
                "Pypdf extraction sufficient for %s (%d chars, %d elements)",
                basename,
                pypdf_quality["total_chars"],
                pypdf_quality["total_elements"],
            )
            return pypdf_elements, "pypdf"

    if not allow_hi_res_fallback:
        raise RuntimeError(
            f"Could not extract sufficient text from {basename} without hi_res "
            f"(fast: {fast_quality['total_chars']} chars). "
            "Set allow_hi_res_fallback: true in config to enable slow layout extraction."
        )

    logger.warning(
        "Fast/pypdf extraction insufficient for %s — falling back to hi_res (slow)",
        basename,
    )
    hi_res_elements = partition_pdf(
        filename=filepath,
        strategy="hi_res",
        infer_table_structure=infer_table_structure,
        **_COMMON_PDF_KWARGS,
    )
    return hi_res_elements, "hi_res"


def assess_extraction_quality(elements) -> dict:
    total_chars = sum(len(e.text or "") for e in elements)
    total_elements = len(elements)

    issues = []

    if total_chars < 100:
        issues.append("very_low_text_content")

    image_elements = [e for e in elements if e.category == "Image"]
    if total_elements and len(image_elements) > 0.5 * total_elements:
        issues.append("high_image_ratio_possible_ocr_failure")

    if total_elements < 5:
        issues.append("very_few_elements_possible_extraction_failure")

    return {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "total_chars": total_chars,
        "total_elements": total_elements,
    }
