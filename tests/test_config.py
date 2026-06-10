from src.config import (
    load_data_config,
    load_indexing_config,
    load_ingestion_config,
    load_rag_config,
)


def test_load_rag_config_has_hybrid_defaults():
    cfg = load_rag_config()
    assert cfg["top_k"] == 5
    assert cfg["hybrid_enabled"] is True
    assert cfg["llm_model"]


def test_load_indexing_config_qdrant_collection():
    cfg = load_indexing_config()
    assert cfg["qdrant_collection"] == "regulatory_docs"
    assert cfg["qdrant_vector_size"] == 1536
    assert cfg["dense_vector_name"] == "dense"


def test_load_data_paths():
    cfg = load_data_config()
    assert cfg["raw_path"] == "data/raw"
    assert cfg["processed_path"] == "data/processed"


def test_load_ingestion_pdf_strategy():
    cfg = load_ingestion_config()
    assert cfg["pdf_strategy"] == "fast_first"
    assert ".pdf" in cfg["file_types"]
