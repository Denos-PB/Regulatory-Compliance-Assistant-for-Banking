import logging
import os
from langchain_community.document_loaders import UnstructuredPDFLoader,UnstructuredHTMLLoader

logger = logging.getLogger(__name__)
def load_documents(directory: str) -> list:
    logger.info(f"Starting document load from: {directory}")
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
                loader = UnstructuredPDFLoader(file_path)
            elif filename.endswith(".html"):
                loader = UnstructuredHTMLLoader(file_path)
            else:
                continue
            
            docs = loader.load()
            results.extend(docs)
            logger.debug(f"Loaded {filename}: {len(docs)} document(s)")

        except Exception as e:
            failed_files.append({"file": filename, "error": str(e)})
            logger.error(f"Failed to load {filename}: {e}")

    if failed_files:
        logger.warning(f"Warning: {len(failed_files)} file(s) failed to load")
    
    logger.info(f"Document load complete: {len(results)} total documents")

    return results

if __name__ == "__main__":
    docs = load_documents("data/raw")
    print(f"Total documents loaded: {len(docs)}")