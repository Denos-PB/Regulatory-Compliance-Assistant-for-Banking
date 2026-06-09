import os
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from langfuse import get_client, observe, propagate_attributes
    from langfuse.langchain import CallbackHandler

    _LANGFUSE_INSTALLED = True
except ImportError:
    _LANGFUSE_INSTALLED = False

    def observe(name=None, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    propagate_attributes = None
    CallbackHandler = None

    def get_client():
        return None


def langfuse_installed() -> bool:
    return _LANGFUSE_INSTALLED


def is_enabled() -> bool:
    if not _LANGFUSE_INSTALLED:
        return False
    if os.getenv("LANGFUSE_ENABLED", "true").lower() in ("0", "false", "no"):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langchain_callbacks() -> list:
    if not is_enabled() or CallbackHandler is None:
        return []
    return [CallbackHandler()]


@contextmanager
def trace_context(
    *,
    name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    if not is_enabled() or propagate_attributes is None:
        yield
        return
    with propagate_attributes(
        trace_name=name,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
        metadata=metadata or {},
    ):
        yield


def flush_traces() -> None:
    if not is_enabled():
        return
    client = get_client()
    if client is not None:
        client.flush()


def retrieval_output(chunks: list) -> dict[str, Any]:
    return {
        "chunk_count": len(chunks),
        "sources": list({getattr(c, "source", "") for c in chunks if getattr(c, "source", None)}),
        "scores": [float(getattr(c, "score", 0)) for c in chunks],
    }


def update_current_output(output: dict[str, Any]) -> None:
    if not is_enabled():
        return
    try:
        client = get_client()
        if client is None:
            return
        observation = client.get_current_observation()
        if observation is not None:
            observation.update(output=output)
    except Exception:
        pass
