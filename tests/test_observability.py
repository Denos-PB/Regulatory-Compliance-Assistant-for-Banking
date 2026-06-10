from src.observability.langfuse_tracing import (
    is_enabled,
    langchain_callbacks,
    langfuse_installed,
    retrieval_output,
    trace_context,
)


def test_langfuse_disabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert is_enabled() is False
    assert langchain_callbacks() == []


def test_trace_context_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    with trace_context(name="test", tags=["unit"]):
        pass


def test_retrieval_output_shape():
    class Chunk:
        source = "pci.pdf"
        score = 0.91

    out = retrieval_output([Chunk(), Chunk()])
    assert out["chunk_count"] == 2
    assert out["sources"] == ["pci.pdf"]
    assert out["scores"] == [0.91, 0.91]


def test_langfuse_install_state_is_bool():
    assert isinstance(langfuse_installed(), bool)
