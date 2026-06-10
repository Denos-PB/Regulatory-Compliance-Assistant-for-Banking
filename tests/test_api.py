from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.rag.retriever import RetrievedChunk


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_env(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "ok" in body
    assert "OPENAI_API_KEY" in body["env"]
    assert "DEEPSEEK_API_KEY" in body["env"]


def test_index_serves_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "Compliance" in res.text


def test_ask_validation_rejects_empty_question(client):
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422


@patch("src.rag.chain.answer")
def test_ask_returns_answer(mock_answer, client):
    mock_answer.return_value = {
        "answer": "PCI DSS protects cardholder data.",
        "sources": ["pci.pdf | p.1 | chunk_0"],
        "chunks": [],
    }
    res = client.post("/ask", json={"question": "What is PCI DSS?"})
    assert res.status_code == 200
    body = res.json()
    assert "cardholder" in body["answer"]
    assert body["sources"]
    mock_answer.assert_called_once()


@patch("src.rag.chain.answer")
def test_ask_passes_session_header(mock_answer, client):
    mock_answer.return_value = {"answer": "ok", "sources": [], "chunks": []}
    client.post(
        "/ask",
        json={"question": "test"},
        headers={"X-Session-Id": "sess-123"},
    )
    _, kwargs = mock_answer.call_args
    assert kwargs.get("session_id") == "sess-123"
    assert kwargs.get("trace_tags") == ["api"]


@patch("src.rag.chain.answer")
def test_ask_return_chunks_serializes_dataclass(mock_answer, client):
    chunk = RetrievedChunk(
        text="excerpt",
        score=0.5,
        chunk_id="c1",
        source="doc.pdf",
    )
    mock_answer.return_value = {
        "answer": "answer",
        "sources": ["doc.pdf"],
        "chunks": [chunk],
    }
    res = client.post(
        "/ask",
        json={"question": "test", "return_chunks": True},
    )
    assert res.status_code == 200
    chunks = res.json()["chunks"]
    assert chunks[0]["text"] == "excerpt"
    assert chunks[0]["chunk_id"] == "c1"
