import json
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import load_indexing_config

logger = logging.getLogger(__name__)


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def merge_by_source(docs: list[Document]) -> list[Document]:
    by_source: dict[str, list[Document]] = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        by_source.setdefault(src,[]).append(doc)

    merged = []
    for src, group in by_source.items():
        group.sort(key=lambda d: d.metadata.get("page_number") or 0)
        text = "\n\n".join(d.page_content.strip() for d in group if d.page_content.strip())
        if not text:
            continue
        meta = dict(group[0].metadata)
        meta['merged_elements'] = len(group)
        meta['page_start'] = group[0].metadata.get("page_number")
        meta['page_end'] = group[-1].metadata.get('page_number')
        merged.append(Document(page_content=text,metadata=meta))

    return merged

def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    cfg: dict | None = None,
) -> list[Document]:
    c = {**load_indexing_config(), **(cfg or {})}
    chunk_size = chunk_size if chunk_size is not None else c["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else c["chunk_overlap"]

    merged = merge_by_source(docs)
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks : list[Document] = []

    for doc in merged:
        src = doc.metadata.get("source", "unknown")
        for i, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_id"] = f"{src}::chunk_{i}"
            chunk.metadata["char_count"] = len(chunk.page_content)
            chunk.metadata["token_estimate"] = len(chunk.page_content) // 4
            chunks.append(chunk)

    logger.info("Created %d chunks from %d source file(s)", len(chunks), len(merged))
    return chunks

def load_parsed_json(path: str | Path) -> list[Document]:
    with open(path,encoding="utf-8") as f:
        rows = json.load(f)
    return [
        Document(page_content=row["text"],metadata=row.get("metadata", {}))
        for row in rows
    ]

def save_chunks_json(chunks: list[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"text": c.page_content , "metadata":c.metadata} for c in chunks]
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)
    logger.info("Saved %d chunks to %s", len(chunks), path)


if __name__ == "__main__":
    from ..logging_config import setup_logging

    setup_logging()
    root = Path(__file__).resolve().parents[2]
    parsed_path = root / "data/processed/parsed_document.json"
    out_path = root / "data/processed/chunks.json"
    if not parsed_path.exists():
        raise SystemExit(f"Run pipeline first. Missing: {parsed_path}")
    parsed = load_parsed_json(parsed_path)
    chunks = chunk_documents(parsed)
    save_chunks_json(chunks, out_path)
