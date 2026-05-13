from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from llm_trace_inspector.evaluator import EvaluatorConfig, TraceEvaluator
from llm_trace_inspector.io import load_trace, write_json
from llm_trace_inspector.reports import summarize_batch, write_csv, write_html, write_jsonl

app = typer.Typer(help="Inspect RAG and agent traces for groundedness, relevance, and hallucination risk.")
console = Console()


@app.command("eval")
def eval_trace(
    trace_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to a trace JSON file."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write full JSON result to a file."),
    html: Path | None = typer.Option(None, "--html", help="Write an HTML report to a file."),
    max_risk: float | None = typer.Option(None, "--max-risk", help="Exit non-zero if hallucination risk is above this value."),
    llm_judge: bool = typer.Option(False, "--llm-judge", help="Enable optional LLM-as-judge pass."),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model", help="Judge model name."),
    judge_provider: str = typer.Option("openai-compatible", "--judge-provider", help="Judge provider label."),
    judge_api_base: str = typer.Option("https://api.openai.com/v1", "--judge-api-base", help="OpenAI-compatible API base URL."),
) -> None:
    trace = load_trace(trace_path)
    result = TraceEvaluator(
        EvaluatorConfig(
            use_llm_judge=llm_judge,
            judge_model=judge_model,
            judge_provider=judge_provider,
            api_base=judge_api_base,
        )
    ).evaluate(trace)

    table = Table(title=f"LLM Trace Inspector: {result.trace_id}")
    table.add_column("Metric")
    table.add_column("Score")
    table.add_row("Groundedness", f"{result.groundedness_score:.3f}")
    table.add_row("Hallucination risk", f"{result.hallucination_risk_score:.3f}")
    table.add_row("Relevance", f"{result.relevance_score:.3f}")
    table.add_row("Citation/support coverage", f"{result.citation_support_coverage:.3f}")
    console.print(table)
    console.print(f"\n[bold]Diagnostic[/bold]\n{result.diagnostic_report}")
    if result.failure_modes:
        console.print("\n[bold yellow]Failure modes[/bold yellow]")
        console.print(", ".join(result.failure_modes))

    if result.unsupported_claims:
        console.print("\n[bold red]Unsupported claims[/bold red]")
        for claim in result.unsupported_claims:
            console.print(f"- {escape(claim)}")

    if result.citation_issues:
        console.print("\n[bold yellow]Citation issues[/bold yellow]")
        for issue in result.citation_issues:
            console.print(f"- {escape(issue)}")

    if output:
        write_json(output, result.model_dump())
        console.print(f"\nWrote JSON result to {output}")
    if html:
        write_html(html, [result])
        console.print(f"Wrote HTML report to {html}")
    if max_risk is not None and result.hallucination_risk_score > max_risk:
        console.print(
            f"\n[bold red]Failed threshold[/bold red]: hallucination risk "
            f"{result.hallucination_risk_score:.3f} > {max_risk:.3f}"
        )
        raise typer.Exit(1)
    if not output and not html:
        console.print("\n[dim]Use --output result.json to save the full machine-readable report.[/dim]")


@app.command("json")
def json_eval(trace_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    trace = load_trace(trace_path)
    result = TraceEvaluator().evaluate(trace)
    console.print(json.dumps(result.model_dump(), indent=2))


@app.command("eval-dir")
def eval_dir(
    traces_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="Directory of trace JSON files."),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o", help="Directory for batch artifacts."),
    max_risk: float | None = typer.Option(None, "--max-risk", help="Mark traces above this hallucination risk as failed and exit non-zero."),
    llm_judge: bool = typer.Option(False, "--llm-judge", help="Enable optional LLM-as-judge pass."),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model", help="Judge model name."),
    judge_provider: str = typer.Option("openai-compatible", "--judge-provider", help="Judge provider label."),
    judge_api_base: str = typer.Option("https://api.openai.com/v1", "--judge-api-base", help="OpenAI-compatible API base URL."),
) -> None:
    paths = sorted(traces_dir.glob("*.json"))
    if not paths:
        console.print(f"[bold red]No JSON traces found in {traces_dir}[/bold red]")
        raise typer.Exit(1)

    evaluator = TraceEvaluator(
        EvaluatorConfig(
            use_llm_judge=llm_judge,
            judge_model=judge_model,
            judge_provider=judge_provider,
            api_base=judge_api_base,
        )
    )
    results = [evaluator.evaluate(load_trace(path)) for path in paths]
    summary = summarize_batch(results, max_risk=max_risk)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary.model_dump())
    write_jsonl(output_dir / "results.jsonl", results)
    write_csv(output_dir / "results.csv", results)
    write_html(output_dir / "report.html", results)

    table = Table(title=f"Batch evaluation: {traces_dir}")
    table.add_column("Trace")
    table.add_column("Grounded")
    table.add_column("Risk")
    table.add_column("Relevance")
    table.add_column("Failures")
    for result in results:
        table.add_row(
            result.trace_id,
            f"{result.groundedness_score:.3f}",
            f"{result.hallucination_risk_score:.3f}",
            f"{result.relevance_score:.3f}",
            ", ".join(result.failure_modes) or "none",
        )
    console.print(table)
    console.print(
        f"Wrote {output_dir / 'summary.json'}, {output_dir / 'results.jsonl'}, "
        f"{output_dir / 'results.csv'}, and {output_dir / 'report.html'}"
    )

    if summary.failed_trace_ids:
        console.print(
            f"[bold red]Failed threshold[/bold red]: {', '.join(summary.failed_trace_ids)} "
            f"exceeded --max-risk {max_risk:.3f}"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
