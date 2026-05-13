from __future__ import annotations

from fastapi import FastAPI

from llm_trace_inspector import __version__
from llm_trace_inspector.evaluator import EvaluatorConfig, TraceEvaluator
from llm_trace_inspector.models import EvaluationResult, Trace

app = FastAPI(
    title="LLM Trace Inspector API",
    version=__version__,
    description="Evaluate RAG and agent traces for groundedness, hallucination risk, relevance, and support coverage.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/eval", response_model=EvaluationResult)
def evaluate_trace(trace: Trace, use_llm_judge: bool = False) -> EvaluationResult:
    evaluator = TraceEvaluator(EvaluatorConfig(use_llm_judge=use_llm_judge))
    return evaluator.evaluate(trace)

