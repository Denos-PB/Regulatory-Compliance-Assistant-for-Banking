import re
import unicodedata
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def clean_text(text:str) -> str:
    text = text.replace('\x00', '')
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\n{3,}','\n\n',text)
    text = re.sub(r' {2,}', ' ',text)
    text = text.strip()

    return text

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

def parse_document(doc:Document) -> Document:
    original_text = doc.page_content
    metadata = dict(doc.metadata)
    doc_type = metadata.get("type", "Text")
        
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

def parse_documents(docs:list)-> list:
    parsed = []
    failed = 0

    for doc in docs:
        try:
            parsed_doc = parse_document(doc)
            parsed.append(parsed_doc)
        except Exception as e:
            failed += 1
            source = doc.metadata.get('source','unknown')
            logger.error(f"Failed to parse document {source}: {e}")
    if failed:
        logger.warning(f"{failed} document(s) failed to parse")
    
    return parsed