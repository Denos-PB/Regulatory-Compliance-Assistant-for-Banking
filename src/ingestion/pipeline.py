import os
import json
import logging
from ..logging_config import setup_logging
from .loader import load_documents
from .parser import parse_documents
from .robust_extraction import validation_file

logger = logging.getLogger(__name__)

def run_pipeline(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    setup_logging()
    logger.info("Starting ingestion pipeline")
    valid_files = []

    for filename in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        validation = validation_file(file_path)
        if validation["status"] == "ok":
            valid_files.append(file_path)
        else:
            logger.warning(f"Skipping {filename}: {validation['reason']}")
    
    logger.info(f"Valid files: {len(valid_files)} out of {len(os.listdir(raw_dir))} total")
    raw_docs = load_documents(raw_dir)
    logger.info(f"Loaded {len(raw_docs)} raw documents")
    parsed_docs = parse_documents(raw_docs)
    logger.info(f"Parsed {len(parsed_docs)} documents")
    quality_issues = 0

    for doc in parsed_docs:
        if len(doc.page_content.strip()) < 50:
            quality_issues += 1
    
    if quality_issues > 0:
        logger.warning(f"{quality_issues} documents have very short content (≤50 chars)")
    
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "parsed_document.json")
    output_data = []

    for doc in parsed_docs:
        output_data.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(parsed_docs)} parsed documents to {output_file}")
    
    return parsed_docs

if __name__ == "__main__":
    run_pipeline()