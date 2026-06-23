from langchain_core.documents import Document

from src.indexing.chunk import chunk_documents, merge_by_page


def test_merge_by_page_keeps_pages_separate():
    docs = [
        Document(page_content="Page one", metadata={"source": "a.pdf", "page_number": 1}),
        Document(page_content="Page two", metadata={"source": "a.pdf", "page_number": 2}),
        Document(page_content="Other doc", metadata={"source": "b.pdf", "page_number": 1}),
    ]
    merged = merge_by_page(docs)
    assert len(merged) == 3
    pages = {(d.metadata["source"], d.metadata["page_number"]) for d in merged}
    assert pages == {("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1)}


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
