from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from llm_trace_inspector.models import BatchSummary, EvaluationResult


def summarize_batch(
    results: list[EvaluationResult],
    threshold_failures: dict[str, list[str]] | None = None,
) -> BatchSummary:
    trace_count = len(results)
    threshold_failures = threshold_failures or {}
    failed = list(threshold_failures)
    if trace_count == 0:
        return BatchSummary(
            trace_count=0,
            average_groundedness=0.0,
            average_hallucination_risk=0.0,
            average_relevance=0.0,
            failed_trace_ids=failed,
            threshold_failures=threshold_failures,
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
        threshold_failures=threshold_failures,
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


def _risk_class(score: float) -> str:
    if score >= 0.6:
        return "risk-high"
    if score >= 0.3:
        return "risk-medium"
    return "risk-low"


def write_html(path: Path, results: list[EvaluationResult]) -> None:
    summary_rows = []
    detail_sections = []
    for result in results:
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(result.trace_id)}</td>"
            f"<td>{result.groundedness_score:.3f}</td>"
            f"<td><span class=\"pill {_risk_class(result.hallucination_risk_score)}\">"
            f"{result.hallucination_risk_score:.3f}</span></td>"
            f"<td>{result.relevance_score:.3f}</td>"
            f"<td>{result.citation_support_coverage:.3f}</td>"
            f"<td>{html.escape(', '.join(result.failure_modes) or 'none')}</td>"
            f"<td>{html.escape(result.diagnostic_report)}</td>"
            "</tr>"
        )
        claim_rows = []
        for claim in result.claim_assessments:
            claim_rows.append(
                "<tr>"
                f"<td>{html.escape(claim.claim)}</td>"
                f"<td>{'yes' if claim.supported else 'no'}</td>"
                f"<td>{html.escape(claim.best_chunk_id or 'none')}</td>"
                f"<td>{claim.support_score:.3f}</td>"
                f"<td>{html.escape(claim.citation_status)}</td>"
                f"<td>{html.escape(claim.explanation)}</td>"
                "</tr>"
            )
        chunk_rows = []
        for chunk in result.chunk_rankings:
            chunk_rows.append(
                "<tr>"
                f"<td>{html.escape(chunk.chunk_id)}</td>"
                f"<td>{chunk.usefulness_score:.3f}</td>"
                f"<td>{chunk.relevance_score:.3f}</td>"
                f"<td>{html.escape(chunk.verdict)}</td>"
                "</tr>"
            )
        detail_sections.append(
            f"""
  <section class="trace-detail">
    <h2>{html.escape(result.trace_id)}</h2>
    <div class="cards">
      <div><span>Groundedness</span><strong>{result.groundedness_score:.3f}</strong></div>
      <div><span>Risk</span><strong>{result.hallucination_risk_score:.3f}</strong></div>
      <div><span>Relevance</span><strong>{result.relevance_score:.3f}</strong></div>
      <div><span>Citation Coverage</span><strong>{result.citation_support_coverage:.3f}</strong></div>
    </div>
    <p>{html.escape(result.diagnostic_report)}</p>
    <h3>Claim Support</h3>
    <table>
      <thead><tr><th>Claim</th><th>Supported</th><th>Best Chunk</th><th>Score</th><th>Citation</th><th>Explanation</th></tr></thead>
      <tbody>{''.join(claim_rows)}</tbody>
    </table>
    <h3>Chunk Ranking</h3>
    <table>
      <thead><tr><th>Chunk</th><th>Usefulness</th><th>Relevance</th><th>Verdict</th></tr></thead>
      <tbody>{''.join(chunk_rows)}</tbody>
    </table>
    <details>
      <summary>Raw evaluation JSON</summary>
      <pre>{html.escape(json.dumps(result.model_dump(), indent=2))}</pre>
    </details>
  </section>
"""
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Trace Inspector Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #18202a; background: #f6f8fb; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .report, .trace-detail {{ background: #fff; border: 1px solid #d8dee8; border-radius: 10px; padding: 22px; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee8; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f6fa; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 16px 0; }}
    .cards div {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 12px; background: #f8fafc; }}
    .cards span {{ display: block; color: #526070; font-size: 13px; }}
    .cards strong {{ display: block; font-size: 24px; margin-top: 4px; }}
    .pill {{ border-radius: 999px; padding: 3px 9px; font-weight: 700; }}
    .risk-low {{ background: #dcfce7; color: #166534; }}
    .risk-medium {{ background: #fef3c7; color: #92400e; }}
    .risk-high {{ background: #fee2e2; color: #991b1b; }}
    pre {{ overflow: auto; background: #111827; color: #e5e7eb; padding: 16px; border-radius: 8px; }}
  </style>
</head>
<body>
  <main class="report">
    <h1>LLM Trace Inspector Report</h1>
  <table>
    <thead>
      <tr>
        <th>Trace</th><th>Groundedness</th><th>Risk</th><th>Relevance</th>
        <th>Citation Coverage</th><th>Failure Modes</th><th>Diagnostic</th>
      </tr>
    </thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table>
  </main>
  {''.join(detail_sections)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
