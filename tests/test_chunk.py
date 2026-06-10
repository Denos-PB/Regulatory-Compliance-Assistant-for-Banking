from langchain_core.documents import Document

from src.indexing.chunk import chunk_documents, merge_by_source


def test_merge_by_source_combines_pages():
    docs = [
        Document(page_content="Page one", metadata={"source": "a.pdf", "page_number": 1}),
        Document(page_content="Page two", metadata={"source": "a.pdf", "page_number": 2}),
        Document(page_content="Other doc", metadata={"source": "b.pdf", "page_number": 1}),
    ]
    merged = merge_by_source(docs)
    assert len(merged) == 2
    by_src = {d.metadata["source"]: d for d in merged}
    assert "Page one" in by_src["a.pdf"].page_content
    assert "Page two" in by_src["a.pdf"].page_content
    assert by_src["a.pdf"].metadata["page_start"] == 1
    assert by_src["a.pdf"].metadata["page_end"] == 2


def test_chunk_documents_assigns_chunk_ids():
    docs = [
        Document(
            page_content="A" * 500 + "\n\n" + "B" * 500,
            metadata={"source": "policy.pdf", "page_number": 1},
        ),
    ]
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["chunk_id"].startswith("policy.pdf::chunk_")
        assert chunk.metadata["source"] == "policy.pdf"
