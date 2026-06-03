"""Shared environment and dependency checks for CLI and API."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .config import load_indexing_config

logger = logging.getLogger(__name__)


def env_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, hint: str, *, required: bool = True) -> None:
        checks.append(
            {"env": name, "ok": ok, "hint": hint, "required": required},
        )

    add("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY")), "Embeddings (index + retrieve).")
    add("DEEPSEEK_API_KEY", bool(os.getenv("DEEPSEEK_API_KEY")), "Answer generation.")
    add(
        "DEEPSEEK_BASE_URL",
        bool(os.getenv("DEEPSEEK_BASE_URL")),
        "Optional; defaults to config rag.llm_base_url.",
        required=False,
    )
    qdrant_url = os.getenv("QDRANT_URL") or load_indexing_config()["qdrant_url"]
    add(
        "QDRANT_URL",
        bool(qdrant_url),
        "Vector store URL (env or config indexing.qdrant_url).",
        required=False,
    )
    return checks


def env_status_map(checks: list[dict[str, Any]]) -> dict[str, bool]:
    return {c["env"]: c["ok"] for c in checks}


def required_env_ok(checks: list[dict[str, Any]]) -> bool:
    return all(c["ok"] for c in checks if c.get("required", True))


def check_qdrant_ready(*, timeout: float = 2.0) -> dict[str, Any]:
    cfg = load_indexing_config()
    base = (os.getenv("QDRANT_URL") or cfg["qdrant_url"]).rstrip("/")
    ready_url = f"{base}/readyz"
    try:
        with urlopen(ready_url, timeout=timeout) as resp:
            ok = resp.status == 200
        message = "Qdrant is reachable." if ok else f"Unexpected status from {ready_url}"
    except URLError as e:
        ok = False
        message = f"Cannot reach Qdrant at {base}: {e.reason}"
    except OSError as e:
        ok = False
        message = f"Cannot reach Qdrant at {base}: {e}"

    if not ok:
        logger.warning("Qdrant health check failed: %s", message)

    return {"qdrant_url": base, "ok": ok, "message": message}


def doctor_report(*, ping_qdrant: bool = True) -> dict[str, Any]:
    checks = env_checks()
    ok = required_env_ok(checks)
    report: dict[str, Any] = {
        "checks": checks,
        "env": env_status_map(checks),
        "ok": ok,
    }
    if ping_qdrant:
        qdrant = check_qdrant_ready()
        report["qdrant"] = qdrant
        report["status"] = {"qdrant": qdrant}
        if not qdrant["ok"]:
            report["ok"] = False
    else:
        report["status"] = {}
    return report
