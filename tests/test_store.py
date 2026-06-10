import uuid

import pytest
from langchain_core.documents import Document

from src.indexing.store import _build_filter, _build_payload, _hybrid_enabled, _point_id


def test_point_id_is_stable_uuid():
    a = _point_id("doc::chunk_0")
    b = _point_id("doc::chunk_0")
    assert a == b
    uuid.UUID(a)


def test_build_payload_requires_chunk_id():
    doc = Document(page_content="text", metadata={"source": "a.pdf"})
    with pytest.raises(ValueError, match="chunk_id"):
        _build_payload(doc)


def test_build_payload_includes_text_and_metadata():
    doc = Document(
        page_content="regulatory text",
        metadata={
            "chunk_id": "a.pdf::chunk_0",
            "source": "a.pdf",
            "page_number": 2,
        },
    )
    payload = _build_payload(doc)
    assert payload["text"] == "regulatory text"
    assert payload["chunk_id"] == "a.pdf::chunk_0"
    assert payload["source"] == "a.pdf"
    assert payload["page_number"] == 2


def test_build_filter_none_without_source():
    assert _build_filter(None) is None
    assert _build_filter("") is None


def test_build_filter_with_source():
    filt = _build_filter("data/raw/pci.pdf")
    assert filt is not None
    assert filt.must[0].key == "source"


def test_hybrid_enabled_reads_rag_config():
    assert isinstance(_hybrid_enabled(), bool)
