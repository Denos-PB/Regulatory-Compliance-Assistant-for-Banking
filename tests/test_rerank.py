from src.rag import reranker
from src.rag.retriever import RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        chunk_id=chunk_id,
        source="data/raw/pci.pdf",
    )


def test_rerank_chunks_reorders_by_cross_encoder_score(monkeypatch):
    class FakeEncoder:
        def rerank(self, query, docs):
            return [0.2 if "PAN" in doc else 0.9 for doc in docs]

    monkeypatch.setattr(reranker, "_cross_encoder", lambda _name: FakeEncoder())

    chunks = [
        _chunk("c1", "PAN is cardholder data"),
        _chunk("c2", "Network segmentation guidance"),
    ]
    result = reranker.rerank_chunks(
        "What is network segmentation?",
        chunks,
        top_k=1,
        cfg={"rerank_model": "test-model"},
    )

    assert len(result) == 1
    assert result[0].chunk_id == "c2"
    assert result[0].score == 0.9


def test_rerank_chunks_empty_input():
    assert reranker.rerank_chunks("query", [], top_k=3) == []


def test_rerank_min_score_filters_weak_chunks(monkeypatch):
    class FakeEncoder:
        def rerank(self, query, docs):
            return [0.9, 0.5, 0.1][: len(docs)]

    monkeypatch.setattr(reranker, "_cross_encoder", lambda _name: FakeEncoder())

    chunks = [
        _chunk("c1", "best match"),
        _chunk("c2", "medium match"),
        _chunk("c3", "weak match"),
    ]
    result = reranker.rerank_chunks(
        "query",
        chunks,
        top_k=3,
        cfg={"rerank_model": "test-model", "rerank_min_score": 0.4},
    )

    assert [c.chunk_id for c in result] == ["c1", "c2"]


def test_rerank_min_score_keeps_top_when_all_below_threshold(monkeypatch):
    class FakeEncoder:
        def rerank(self, query, docs):
            return [0.1, 0.2][: len(docs)]

    monkeypatch.setattr(reranker, "_cross_encoder", lambda _name: FakeEncoder())

    chunks = [_chunk("c1", "a"), _chunk("c2", "b")]
    result = reranker.rerank_chunks(
        "query",
        chunks,
        top_k=2,
        cfg={"rerank_model": "test-model", "rerank_min_score": 0.5},
    )

    assert len(result) == 1
    assert result[0].chunk_id == "c2"
