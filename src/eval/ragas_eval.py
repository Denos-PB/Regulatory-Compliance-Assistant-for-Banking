from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

from src.config import load_indexing_config, load_rag_config
from src.eval._ragas_compat import ensure_ragas_importable
from src.rag.chain import answer

ensure_ragas_importable()

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_PATH = Path("data/eval/golden_questions.json")


def _ragas_llm() -> LangchainLLMWrapper:
    cfg = load_rag_config()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for RAGAS judge.")

    llm = ChatDeepSeek(
        model=cfg["llm_model"],
        temperature=0,
        max_tokens=cfg["max_tokens"],
        api_key=api_key,
        api_base=os.getenv("DEEPSEEK_BASE_URL") or cfg["llm_base_url"],
    )
    return LangchainLLMWrapper(llm)


def _ragas_embeddings() -> LangchainEmbeddingsWrapper:
    indexing = load_indexing_config()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS embeddings.")

    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=indexing["embedding_model"], api_key=api_key)
    )


def load_golden(path: str | Path = DEFAULT_GOLDEN_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data["questions"]
    raise ValueError(f"Invalid golden set format in {path}")


def build_eval_dataset(
    items: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> dict[str, list]:
    questions: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truths: list[str] = []

    for i, item in enumerate(items, start=1):
        question = (item.get("question") or "").strip()
        if not question:
            logger.warning("Skipping empty question at index %d", i)
            continue

        logger.info("Eval %d/%d: %s", i, len(items), question[:80])
        result = answer(question, top_k=top_k, trace_tags=["eval"])
        chunks = result.get("chunks") or []
        contexts = [c.text for c in chunks if getattr(c, "text", None)]

        questions.append(question)
        answers.append(result.get("answer") or "")
        contexts_list.append(contexts)
        ground_truths.append(item.get("ground_truth") or "")

    if not questions:
        raise ValueError("No valid questions in golden set.")

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def run_ragas_evaluation(
    golden_path: str | Path = DEFAULT_GOLDEN_PATH,
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    load_dotenv()
    items = load_golden(golden_path)
    records = build_eval_dataset(items, top_k=top_k)
    dataset = Dataset.from_dict(records)

    metrics = [faithfulness, answer_relevancy, context_precision]
    logger.info("Running RAGAS on %d question(s)...", len(records["question"]))

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=_ragas_llm(),
        embeddings=_ragas_embeddings(),
    )

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    return {
        "questions": len(records["question"]),
        "golden_path": str(golden_path),
        "scores": scores,
        "per_row": result.to_pandas().to_dict(orient="records"),
    }
