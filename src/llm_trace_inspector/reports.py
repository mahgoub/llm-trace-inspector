from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from llm_trace_inspector.models import BatchSummary, EvaluationResult


def summarize_batch(results: list[EvaluationResult], max_risk: float | None = None) -> BatchSummary:
    trace_count = len(results)
    failed = [
        result.trace_id
        for result in results
        if max_risk is not None and result.hallucination_risk_score > max_risk
    ]
    if trace_count == 0:
        return BatchSummary(
            trace_count=0,
            average_groundedness=0.0,
            average_hallucination_risk=0.0,
            average_relevance=0.0,
            failed_trace_ids=failed,
        )
    return BatchSummary(
        trace_count=trace_count,
        average_groundedness=round(sum(item.groundedness_score for item in results) / trace_count, 3),
        average_hallucination_risk=round(
            sum(item.hallucination_risk_score for item in results) / trace_count,
            3,
        ),
        average_relevance=round(sum(item.relevance_score for item in results) / trace_count, 3),
        failed_trace_ids=failed,
    )


def write_jsonl(path: Path, results: list[EvaluationResult]) -> None:
    path.write_text(
        "\n".join(json.dumps(result.model_dump()) for result in results) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, results: list[EvaluationResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trace_id",
                "groundedness_score",
                "hallucination_risk_score",
                "relevance_score",
                "citation_support_coverage",
                "unsupported_claim_count",
                "missing_fact_count",
                "failure_modes",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "trace_id": result.trace_id,
                    "groundedness_score": result.groundedness_score,
                    "hallucination_risk_score": result.hallucination_risk_score,
                    "relevance_score": result.relevance_score,
                    "citation_support_coverage": result.citation_support_coverage,
                    "unsupported_claim_count": len(result.unsupported_claims),
                    "missing_fact_count": len(result.missing_facts_from_context),
                    "failure_modes": ",".join(result.failure_modes),
                }
            )


def write_html(path: Path, results: list[EvaluationResult]) -> None:
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.trace_id)}</td>"
            f"<td>{result.groundedness_score:.3f}</td>"
            f"<td>{result.hallucination_risk_score:.3f}</td>"
            f"<td>{result.relevance_score:.3f}</td>"
            f"<td>{result.citation_support_coverage:.3f}</td>"
            f"<td>{html.escape(', '.join(result.failure_modes) or 'none')}</td>"
            f"<td>{html.escape(result.diagnostic_report)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Trace Inspector Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #18202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee8; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f6fa; }}
    .metric {{ display: inline-block; margin: 0 18px 18px 0; }}
  </style>
</head>
<body>
  <h1>LLM Trace Inspector Report</h1>
  <table>
    <thead>
      <tr>
        <th>Trace</th><th>Groundedness</th><th>Risk</th><th>Relevance</th>
        <th>Citation Coverage</th><th>Failure Modes</th><th>Diagnostic</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")

