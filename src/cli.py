import json
import os
from typing import Any, Optional
from dotenv import load_dotenv
import typer
from src.config import load_data_config

load_dotenv()

app = typer.Typer(
    help="Regulatory Compliance Assistant — use `run` for the full pipeline.",
)


def _doctor_env() -> dict[str, Any]:
    checks = []

    def add(name: str, ok: bool, hint: str, required: bool = True) -> None:
        checks.append({"env": name, "ok": ok, "hint": hint, "required": required})

    add("OPENAI_API_KEY", bool(os.getenv("OPENAI_API_KEY")), "Embeddings (index + retrieve).")
    add("DEEPSEEK_API_KEY", bool(os.getenv("DEEPSEEK_API_KEY")), "Answer generation.")
    add(
        "DEEPSEEK_BASE_URL",
        bool(os.getenv("DEEPSEEK_BASE_URL")),
        "Optional; defaults to config rag.llm_base_url.",
        required=False,
    )
    add(
        "QDRANT_URL",
        bool(os.getenv("QDRANT_URL")),
        "Optional; defaults to config indexing.qdrant_url.",
        required=False,
    )
    from src.observability.langfuse_tracing import is_enabled, langfuse_installed

    tracing_ok = langfuse_installed() and is_enabled()
    add(
        "LANGFUSE_PUBLIC_KEY",
        tracing_ok,
        "Optional; install observability extra and set LANGFUSE_* keys for tracing.",
        required=False,
    )
    return {"checks": checks}


def _print_ask_result(result: dict[str, Any]) -> None:
    typer.echo("\n=== Answer ===\n")
    typer.echo(result.get("answer", ""))
    sources = result.get("sources") or []
    if sources:
        typer.echo("\n=== Sources ===\n")
        for s in sources:
            typer.echo(f"- {s}")


def _default_paths() -> tuple[str, str]:
    data = load_data_config()
    return data["raw_path"], data["processed_path"]


@app.command()
def run(
    question: Optional[str] = typer.Option(
        None,
        "-q",
        "--question",
        help="Optional question to answer after indexing",
    ),
    raw_dir: Optional[str] = typer.Option(None, "--raw-dir", help="Folder with raw PDFs/HTMLs"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Processed output folder"),
    recreate: bool = typer.Option(False, "--recreate", help="Recreate Qdrant collection"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Use existing parsed_document.json"),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip indexing (ingest only, or ask only)"),
    save_chunks: bool = typer.Option(False, "--save-chunks", help="Write chunks.json checkpoint"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Chunks to retrieve when asking"),
    source_filter: Optional[str] = typer.Option(None, "--source-filter", help="Limit retrieval to one source path"),
    json_out: bool = typer.Option(False, "--json", help="JSON output"),
):
    from src.pipeline import run_all

    default_raw, default_out = _default_paths()
    try:
        summary = run_all(
            raw_dir=raw_dir or default_raw,
            output_dir=output_dir or default_out,
            question=question,
            recreate_collection=recreate,
            skip_ingest=skip_ingest,
            skip_index=skip_index,
            save_chunks=save_chunks,
            top_k=top_k,
            source_filter=source_filter,
        )
    except Exception as e:
        typer.secho(f"Pipeline failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    if json_out:
        typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
        raise typer.Exit(0)

    typer.echo(
        f"\nDone. Ingested {summary['ingested_documents']} document(s), "
        f"indexed {summary['indexed_points']} point(s)."
    )
    if summary.get("answer"):
        _print_ask_result(summary)


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="JSON output"),
):
    result = _doctor_env()
    if json_out:
        typer.echo(json.dumps(result, indent=2))
        raise typer.Exit(0)

    typer.echo("Environment checks:")
    for c in result["checks"]:
        label = "OK" if c["ok"] else ("MISSING" if c.get("required", True) else "optional")
        typer.echo(f"- {c['env']}: {label}")
        if not c["ok"] and c.get("required", True):
            typer.echo(f"  {c['hint']}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Chunks to retrieve"),
    source_filter: Optional[str] = typer.Option(None, "--source-filter", help="Filter by document source"),
    json_out: bool = typer.Option(False, "--json", help="JSON output"),
):
    from src.rag.chain import answer

    result: dict[str, Any] = answer(
        question,
        top_k=top_k,
        source_filter=source_filter,
        trace_tags=["cli"],
    )

    if json_out:
        output = dict(result)
        chunks = output.get("chunks")
        if chunks is not None:
            output["chunks"] = [
                item.__dict__ if hasattr(item, "__dict__") else str(item) for item in chunks
            ]
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False))
        raise typer.Exit(0)

    _print_ask_result(result)


@app.command()
def eval_ragas(
    golden: str = typer.Option(
        "data/eval/golden_questions.json",
        "--golden",
        help="Path to golden questions JSON",
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Chunks to retrieve per question"),
    json_out: bool = typer.Option(False, "--json", help="JSON output"),
):
    from src.eval.ragas_eval import run_ragas_evaluation

    try:
        report = run_ragas_evaluation(golden, top_k=top_k)
    except Exception as e:
        typer.secho(f"RAGAS evaluation failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    if json_out:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        raise typer.Exit(0)

    typer.echo(f"RAGAS evaluation ({report['questions']} question(s))")
    typer.echo(f"Golden set: {report['golden_path']}\n")
    typer.echo("Overall:")
    for metric, value in sorted(report["scores"].items()):
        if value is None:
            continue
        typer.echo(f"- {metric}: {value:.4f}")

    by_topic = report.get("by_topic") or {}
    if by_topic:
        typer.echo("\nBy topic:")
        for topic in sorted(by_topic):
            typer.echo(f"  [{topic}]")
            for metric, value in sorted(by_topic[topic].items()):
                if value is None:
                    continue
                typer.echo(f"    - {metric}: {value:.4f}")

    worst = report.get("worst_context_precision") or []
    if worst:
        typer.echo("\nLowest context_precision:")
        for row in worst:
            q = row.get("question", "")
            preview = q[:70] + ("..." if len(q) > 70 else "")
            typer.echo(
                f"  - {row.get('context_precision', 0):.4f} [{row.get('topic', '?')}] {preview}"
            )


if __name__ == "__main__":
    app()
