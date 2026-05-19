import os
import subprocess
import PyPDF2
from pdfminer.high_level import extract_text
from unstructured.partition.pdf import partition_pdf

def get_mime_type(filepath):
    try:
        import magic
        return magic.from_file(filepath,mime=True)
    except(ImportError, OSError):
        result = subprocess.run(
            ["file", "--mime-type", filepath],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split(": ")[-1]

def validation_file(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {"status": "skip", "reason": "empty_or_missing"}
    
    mime_type = get_mime_type(filepath)
    if mime_type != "application/pdf":
        return {"status": "skip", "reason": f"unexpected_mime:{mime_type}"}
    
    try:
        with open(filepath,"rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                return {"status": "skip", "reason": "encrypted"}
    except Exception as e:
        return {"status": "skip", "reason": f"corrupted:{str(e)}"}
    
    return {"status" : "ok"}

def is_scaned_pdf(filepath):
    try:
        text = extract_text(filepath)
        return len(text.strip()) < 50
    except:
        return True
    
def extract_with_fallback(filepath):
    if is_scaned_pdf(filepath):
        print(f"  [INFO] Scanned PDF detected — using OCR strategy")
        elements = partition_pdf(
            filepath,
            strategy="ocr_only",
            ocr_languages="eng",
        )
    else:
        print(f"  [INFO] Digital PDF detected — using auto strategy")
        elements = partition_pdf(
            filepath,
            strategy="auto",
        )

    return elements

def assess_extraction_quality(elements):
    total_chars = sum(len(e.text) for e in elements)
    total_elements = len(elements)

    issues=[]

    if total_chars < 100:
        issues.append("very_low_text_content")

    image_elements = [e for e in elements if e.category == "Image"]
    if len(image_elements) > 0.5 * total_elements:
        issues.append("high_image_ratio_possible_ocr_failure")

    if total_elements < 5:
        issues.append("very_few_elements_possible_extraction_failure")

    return {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "total_chars": total_chars,
        "total_elements": total_elements,
    }