import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

DEFAULTS = {
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


def load_ingestion_config() -> dict:
    cfg = dict(DEFAULTS)
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

    ingestion = raw.get("ingestion")
    if not isinstance(ingestion, dict):
        logger.warning("Missing or invalid 'ingestion' section in config; using defaults")
        return cfg

    for key in DEFAULTS:
        if key in ingestion:
            cfg[key] = ingestion[key]
    return cfg
