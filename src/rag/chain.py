import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

from ..config import load_rag_config
from ..observability.langfuse_tracing import (
    langchain_callbacks,
    observe,
    trace_context,
    flush_traces,
)
from .prompts import build_messages
from .retriever import format_context, retrieve, unique_citations

load_dotenv()
logger = logging.getLogger(__name__)

def _llm(cfg: dict | None = None) -> ChatDeepSeek:
    c = {**load_rag_config(), **(cfg or {})}
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env")
    return ChatDeepSeek(
        model=c["llm_model"],
        temperature=c["temperature"],
        max_tokens=c["max_tokens"],
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL") or c["llm_base_url"],
    )

@observe(name="rag_answer")
def answer(
    query: str,
    *,
    top_k: int | None = None,
    source_filter: str | None = None,
    cfg: dict | None = None,
    trace_tags: list[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    if not query or not query.strip():
        return {
            "answer": "Please provide a question.",
            "sources": [],
            "chunks": [],
        }

    tags = trace_tags or ["rag"]
    with trace_context(
        name="rag_answer",
        session_id=session_id,
        tags=tags,
        metadata={
            "top_k": top_k,
            "source_filter": source_filter,
        },
    ):
        chunks = retrieve(
            query,
            top_k=top_k,
            source_filter=source_filter,
            cfg=cfg,
        )

        if not chunks:
            logger.warning("No chunks retrieved for query")
            return {
                "answer": (
                    "I could not find relevant information in the indexed regulatory "
                    "documents for this question. Try rephrasing or ensure the documents "
                    "have been ingested and indexed."
                ),
                "sources": [],
                "chunks": [],
            }

        context = format_context(chunks)
        messages = build_messages(query.strip(), context)
        llm = _llm(cfg)
        response = llm.invoke(messages, config={"callbacks": langchain_callbacks()})

        sources = unique_citations(chunks)
        result = {
            "answer": response.content if hasattr(response, "content") else str(response),
            "sources": sources,
            "chunks": chunks,
        }

    flush_traces()
    return result
