from pathlib import Path

import pytest

from src.eval.ragas_eval import load_golden


def test_load_golden_default_file():
    items = load_golden()
    assert len(items) >= 3
    assert "question" in items[0]


def test_load_golden_wrapped_format(tmp_path: Path):
    path = tmp_path / "golden.json"
    path.write_text(
        '{"questions": [{"question": "Q?", "ground_truth": "A"}]}',
        encoding="utf-8",
    )
    items = load_golden(path)
    assert items[0]["question"] == "Q?"


def test_load_golden_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "missing.json")


def test_load_golden_invalid_format(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid golden set"):
        load_golden(path)
