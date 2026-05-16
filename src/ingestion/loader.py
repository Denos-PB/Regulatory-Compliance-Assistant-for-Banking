import os
import logging
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_documents(directory: str) -> list:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    results = []
    failed_files = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            if filename.endswith(".pdf"):
                elements = partition_pdf(
                    filename=file_path,
                    strategy="hi_res",
                    extract_images_in_pdf=True,
                    infer_table_structure=True
                )
            elif filename.endswith(".html"):
                elements = partition_html(filename=file_path)
            else:
                continue
            
            for element in elements:
                doc = Document(
                    page_content=element.text or "",
                    metadata={
                        "source": file_path,
                        "file_type": "pdf" if filename.endswith(".pdf") else "html",
                        "page_number": element.metadata.page_number,
                        "type": element.category,
                    }
                )

                if element.category == "Table" and hasattr(element.metadata, 'text_as_html'):
                    doc.metadata["text_as_html"] = element.metadata.text_as_html

                results.append(doc)
            
            logger.debug(f"Loaded {filename}: {len(elements)} element(s)")

        except Exception as e:
            failed_files.append({"file": filename, "error": str(e)})
            logger.error(f"Failed to load {filename}: {e}")
    
    if failed_files:
        logger.warning(f"{len(failed_files)} file(s) failed to load")
    
    return results