from langchain_core.documents import Document

from src.rag.chain import answer
from src.rag.prompts import build_messages
from src.rag.retriever import RetrievedChunk, format_context, retrieve


def test_build_messages_includes_context_and_question():
    messages = build_messages("What is PCI DSS?", "[1] (doc.pdf)\nRequirement text")
    assert len(messages) == 2
    human = messages[1].content
    assert "PCI DSS" in human
    assert "Requirement text" in human


def test_build_messages_without_context():
    messages = build_messages("Any question?", "")
    assert "cannot answer" in messages[1].content.lower()


def test_format_context_numbering():
    chunks = [
        RetrievedChunk(
            text="First excerpt",
            score=0.9,
            chunk_id="c1",
            source="pci.pdf",
            page_number=3,
        ),
        RetrievedChunk(
            text="Second excerpt",
            score=0.8,
            chunk_id="c2",
            source="basel.pdf",
            page_start=10,
            page_end=12,
        ),
    ]
    ctx = format_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "pci.pdf" in ctx
    assert "pp.10-12" in ctx


def test_retrieved_chunk_citation():
    chunk = RetrievedChunk(
        text="x",
        score=1.0,
        chunk_id="id-1",
        source="data/raw/pci.pdf",
        page_number=7,
    )
    citation = chunk.citation()
    assert "data/raw/pci.pdf" in citation
    assert "p.7" in citation
    assert "id-1" in citation


def test_answer_empty_query():
    result = answer("   ")
    assert result["answer"] == "Please provide a question."
    assert result["sources"] == []
    assert result["chunks"] == []


def test_retrieve_empty_query():
    assert retrieve("") == []
    assert retrieve("   ") == []
