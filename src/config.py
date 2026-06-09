import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DATA_DEFAULTS = {
    "raw_path": "data/raw",
    "processed_path": "data/processed",
    "logs": "logs",
}

INGESTION_DEFAULTS = {
    "min_text_length": 50,
    "skip_categories": ["Image", "Figure"],
    "languages": ["eng"],
    "file_types": [".pdf", ".html", ".htm"],
    "fail_on_quality_issues": False,
    "pdf_strategy": "fast_first",
    "fast_min_char_ratio": 0.05,
    "fast_min_absolute_chars": 500,
    "allow_hi_res_fallback": True,
    "dedupe_repeated_lines": True,
    "dedupe_min_pages": 10,
    "dedupe_min_line_length": 25,
    "dedupe_max_line_length": 220,
    "dedupe_ratio_threshold": 0.6,
}

INDEXING_DEFAULTS = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-3-small",
    "embedding_batch_size": 100,
    "qdrant_url": "http://localhost:6333",
    "qdrant_api_key": "",
    "qdrant_collection": "regulatory_docs",
    "qdrant_vector_size": 1536,
    "qdrant_batch_size": 64,
    "sparse_model": "Qdrant/bm25",
    "dense_vector_name": "dense",
    "sparse_vector_name": "sparse",
}

RAG_DEFAULTS = {
    "top_k": 5,
    "score_threshold": 0.0,
    "llm_model": "deepseek-chat",
    "llm_base_url": "https://api.deepseek.com",
    "temperature": 0,
    "max_tokens": 1024,
    "hybrid_enabled": True,
    "hybrid_prefetch_limit": 20,
}

_SECTION_DEFAULTS = {
    "data": DATA_DEFAULTS,
    "ingestion": INGESTION_DEFAULTS,
    "indexing": INDEXING_DEFAULTS,
    "rag": RAG_DEFAULTS,
}


def _load_section(section: str) -> dict:
    defaults = _SECTION_DEFAULTS[section]
    cfg = dict(defaults)
    if not _CONFIG_PATH.exists():
        logger.warning("Config file not found at %s; using defaults", _CONFIG_PATH)
        return cfg

    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except OSError as e:
        logger.error("Cannot read config %s: %s; using defaults", _CONFIG_PATH, e)
        return cfg
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in %s: %s; using defaults", _CONFIG_PATH, e)
        return cfg

    data = raw.get(section)
    if not isinstance(data, dict):
        logger.warning("Missing or invalid '%s' section in config; using defaults", section)
        return cfg

    if section == "data" and "processed_data" in data and "processed_path" not in data:
        data["processed_path"] = data["processed_data"]

    for key in defaults:
        if key in data:
            cfg[key] = data[key]
    return cfg


def load_data_config() -> dict:
    return _load_section("data")


def load_ingestion_config() -> dict:
    return _load_section("ingestion")


def load_indexing_config() -> dict:
    return _load_section("indexing")


def load_rag_config() -> dict:
    return _load_section("rag")
