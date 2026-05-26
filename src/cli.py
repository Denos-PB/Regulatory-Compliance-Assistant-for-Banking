import os
import json
import typer
from typing import Any, Optional

app = typer.Typer(help="Regulatory Compliance Assistant CLI")


def _doctor_env() -> dict[str, Any]:
    checks = []

    def add(name: str, ok: bool, hint: str) -> None:
        checks.append(
            {
                "env": name,
                "ok": ok,
                "hint": hint,
            }
        )

    add(
        "OPENAI_API_KEY",
        bool(os.getenv("OPENAI_API_KEY")),
        "Required for embeddings (indexing + retrieval). Put it in .env.",
    )
    add(
        "DEEPSEEK_API_KEY",
        bool(os.getenv("DEEPSEEK_API_KEY")),
        "Required for generation in chain.py (if using DeepSeek). Put it in .env.",
    )
    add(
        "DEEPSEEK_BASE_URL",
        bool(os.getenv("DEEPSEEK_BASE_URL") or True),
        "Optional if you hardcode base_url in config.yaml. Otherwise set it.",
    )
    add(
        "QDRANT_URL",
        bool(os.getenv("QDRANT_URL") or True),
        "Optional; config.yaml has a default. For prod, you can set it in .env.",
    )

    return {"checks": checks}


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    result = _doctor_env()
    if json_out:
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    typer.echo("Environment checks:")
    for c in result["checks"]:
        status = "OK" if c["ok"] else "MISSING"
        typer.echo(f"- {c['env']}: {status}")
        if not c["ok"]:
            typer.echo(f"  Hint: {c['hint']}")


@app.command()
def ingest(
    raw_dir: str = typer.Option("data/raw", "--raw-dir", help="Folder with raw PDFs/HTMLs"),
    output_dir: str = typer.Option("data/processed", "--output-dir", help="Where parsed JSON is saved"),
):
    from src.ingestion.pipeline import run_pipeline

    run_pipeline(raw_dir=raw_dir, output_dir=output_dir)


@app.command()
def index(
    parsed_path: Optional[str] = typer.Option(
        None,
        "--parsed-path",
        help="Path to parsed_document.json. Defaults to data/processed/parsed_document.json",
    ),
    output_dir: str = typer.Option("data/processed", "--output-dir", help="Folder with processed artifacts"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate Qdrant collection (destructive)"),
    save_chunks: bool = typer.Option(True, "--save-chunks", help="Save chunks.json (debug/re-embed checkpoint)"),
):
    from src.indexing.pipeline import run_indexing_pipeline

    run_indexing_pipeline(
        parsed_path=parsed_path,
        output_dir=output_dir,
        save_chunks=save_chunks,
        recreate_collection=recreate,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="User question to answer from your regulatory corpus"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="How many chunks to retrieve"),
    source_filter: Optional[str] = typer.Option(
        None,
        "--source-filter",
        help="Optional: restrict retrieval to a specific document source path (matches payload 'source')",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
):
    try:
        from src.rag.chain import answer  # your chain.py must expose answer()
    except Exception as e:
        raise typer.Exit(
            f"Cannot import src.rag.chain.answer. Make sure src/rag/chain.py defines `answer()`.\nDetails: {e}"
        )

    result: dict[str, Any] = answer(
        question,
        top_k=top_k,
        source_filter=source_filter,
    )

    if json_out:
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
        raise typer.Exit(0)

    typer.echo("\n=== Answer ===\n")
    typer.echo(result.get("answer", ""))

    sources = result.get("sources") or []
    if sources:
        typer.echo("\n=== Sources ===\n")
        for s in sources:
            typer.echo(f"- {s}")


if __name__ == "__main__":
    app()