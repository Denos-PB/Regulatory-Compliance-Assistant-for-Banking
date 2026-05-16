import os
import sys
import json
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.logging_config import setup_logging
from src.ingestion.loader import load_documents
from src.ingestion.parser import parse_documents

logger = logging.getLogger(__name__)

def run_pipeline(raw_dir: str = "data/raw", output_dir: str = "data/processed"):
    setup_logging()
    logger.info("Starting ingestion pipeline")
    raw_docs = load_documents("data/raw")
    logger.info(f"Loaded {len(raw_docs)} raw documents")
    parsed_docs = parse_documents(raw_docs)
    logger.info(f"Parsed {len(parsed_docs)} documents")
    os.makedirs(output_dir,exist_ok=True)
    output_file=os.path.join(output_dir,"parsed_document.json")

    output_data=[]
    for doc in parsed_docs:
        output_data.append({
            "text": doc.page_content,
            "metadata": doc.metadata
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data,f,indent=2,ensure_ascii=False)

    logger.info(f"Saved {len(parsed_docs)} parsed documents to {output_file}")

    return parsed_docs

if __name__ == "__main__":
    run_pipeline()