from pathlib import Path

from llm_trace_inspector.evaluator import TraceEvaluator
from llm_trace_inspector.io import load_trace
from llm_trace_inspector.reports import summarize_batch, write_csv, write_html, write_jsonl


def test_batch_summary_and_exports(tmp_path: Path) -> None:
    evaluator = TraceEvaluator()
    results = [
        evaluator.evaluate(load_trace("examples/rag_trace_good.json")),
        evaluator.evaluate(load_trace("examples/rag_trace_hallucinated.json")),
    ]

    summary = summarize_batch(results, max_risk=0.4)
    assert summary.trace_count == 2
    assert "rag-risk-001" in summary.failed_trace_ids

    write_jsonl(tmp_path / "results.jsonl", results)
    write_csv(tmp_path / "results.csv", results)
    write_html(tmp_path / "report.html", results)

    assert (tmp_path / "results.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert "trace_id" in (tmp_path / "results.csv").read_text(encoding="utf-8")
    assert "LLM Trace Inspector Report" in (tmp_path / "report.html").read_text(encoding="utf-8")

