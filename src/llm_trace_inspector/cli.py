from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_trace_inspector.evaluator import EvaluatorConfig, TraceEvaluator
from llm_trace_inspector.io import load_trace, write_json

app = typer.Typer(help="Inspect RAG and agent traces for groundedness, relevance, and hallucination risk.")
console = Console()


@app.command("eval")
def eval_trace(
    trace_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to a trace JSON file."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write full JSON result to a file."),
    llm_judge: bool = typer.Option(False, "--llm-judge", help="Enable optional LLM-as-judge pass."),
) -> None:
    trace = load_trace(trace_path)
    result = TraceEvaluator(EvaluatorConfig(use_llm_judge=llm_judge)).evaluate(trace)

    table = Table(title=f"LLM Trace Inspector: {result.trace_id}")
    table.add_column("Metric")
    table.add_column("Score")
    table.add_row("Groundedness", f"{result.groundedness_score:.3f}")
    table.add_row("Hallucination risk", f"{result.hallucination_risk_score:.3f}")
    table.add_row("Relevance", f"{result.relevance_score:.3f}")
    table.add_row("Citation/support coverage", f"{result.citation_support_coverage:.3f}")
    console.print(table)
    console.print(f"\n[bold]Diagnostic[/bold]\n{result.diagnostic_report}")

    if result.unsupported_claims:
        console.print("\n[bold red]Unsupported claims[/bold red]")
        for claim in result.unsupported_claims:
            console.print(f"- {claim}")

    if output:
        write_json(output, result.model_dump())
        console.print(f"\nWrote JSON result to {output}")
    else:
        console.print("\n[dim]Use --output result.json to save the full machine-readable report.[/dim]")


@app.command("json")
def json_eval(trace_path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    trace = load_trace(trace_path)
    result = TraceEvaluator().evaluate(trace)
    console.print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    app()

