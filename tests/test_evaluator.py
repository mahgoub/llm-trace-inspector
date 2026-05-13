from llm_trace_inspector.evaluator import TraceEvaluator
from llm_trace_inspector.io import load_trace


def test_good_trace_scores_high_groundedness() -> None:
    trace = load_trace("examples/rag_trace_good.json")
    result = TraceEvaluator().evaluate(trace)

    assert result.groundedness_score >= 0.8
    assert result.hallucination_risk_score <= 0.35
    assert result.citation_support_coverage >= 0.8
    assert result.unsupported_claims == []


def test_hallucinated_trace_flags_unsupported_claims() -> None:
    trace = load_trace("examples/rag_trace_hallucinated.json")
    result = TraceEvaluator().evaluate(trace)

    assert result.hallucination_risk_score >= 0.5
    assert result.unsupported_claims
    assert result.groundedness_score < 0.6


def test_bad_citations_are_flagged() -> None:
    trace = load_trace("examples/bad_citations.json")
    result = TraceEvaluator().evaluate(trace)

    assert "citation_mismatch" in result.failure_modes
    assert result.citation_issues
    assert result.claim_assessments[0].citation_status == "mismatch"


def test_partial_answer_has_score_explanations() -> None:
    trace = load_trace("examples/partially_grounded_answer.json")
    result = TraceEvaluator().evaluate(trace)

    assert result.score_explanations["hallucination_risk"]
    assert "unsupported_claim" in result.failure_modes
