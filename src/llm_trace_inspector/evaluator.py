from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from llm_trace_inspector.models import (
    ChunkAssessment,
    ClaimAssessment,
    EvaluationResult,
    Trace,
)
from llm_trace_inspector.text_utils import clamp, cosine_similarity, lexical_overlap, split_sentences


@dataclass(frozen=True)
class EvaluatorConfig:
    use_llm_judge: bool = False
    judge_model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str = "https://api.openai.com/v1"
    support_threshold: float = 0.46


class TraceEvaluator:
    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self.config = config or EvaluatorConfig()

    def evaluate(self, trace: Trace) -> EvaluationResult:
        deterministic = self._deterministic_eval(trace)
        if not self.config.use_llm_judge:
            return deterministic

        llm_adjustments = self._call_llm_judge(trace, deterministic)
        if not llm_adjustments:
            return deterministic

        return deterministic.model_copy(
            update={
                "judge_mode": "hybrid",
                "raw_judge": llm_adjustments,
                "diagnostic_report": deterministic.diagnostic_report
                + "\n\nLLM judge notes:\n"
                + llm_adjustments.get("notes", "No notes returned."),
            }
        )

    def _deterministic_eval(self, trace: Trace) -> EvaluationResult:
        chunks_text = "\n\n".join(chunk.text for chunk in trace.retrieved_context)
        claims = split_sentences(trace.llm_answer)
        claim_assessments = [self._assess_claim(claim, trace) for claim in claims]

        supported = [assessment for assessment in claim_assessments if assessment.supported]
        groundedness = len(supported) / len(claim_assessments) if claim_assessments else 0.0
        unsupported_claims = [assessment.claim for assessment in claim_assessments if not assessment.supported]
        relevance = self._relevance_score(trace)
        citation_coverage = self._citation_support_coverage(trace, claim_assessments)
        missing_facts = self._missing_facts(trace, chunks_text)
        hallucination_risk = clamp((1.0 - groundedness) * 0.65 + (1.0 - citation_coverage) * 0.2 + len(missing_facts) * 0.05)
        chunk_rankings = self._rank_chunks(trace, claim_assessments)

        result = EvaluationResult(
            trace_id=trace.trace_id,
            groundedness_score=round(groundedness, 3),
            hallucination_risk_score=round(hallucination_risk, 3),
            relevance_score=round(relevance, 3),
            citation_support_coverage=round(citation_coverage, 3),
            unsupported_claims=unsupported_claims,
            missing_facts_from_context=missing_facts,
            chunk_rankings=chunk_rankings,
            diagnostic_report="",
            judge_mode="deterministic",
        )
        return result.model_copy(update={"diagnostic_report": self._report(trace, result)})

    def _assess_claim(self, claim: str, trace: Trace) -> ClaimAssessment:
        best_chunk_id = None
        best_score = 0.0
        for chunk in trace.retrieved_context:
            score = max(lexical_overlap(claim, chunk.text), cosine_similarity(claim, chunk.text))
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.id
        return ClaimAssessment(
            claim=claim,
            supported=best_score >= self.config.support_threshold,
            best_chunk_id=best_chunk_id,
            support_score=round(clamp(best_score), 3),
        )

    def _relevance_score(self, trace: Trace) -> float:
        if not trace.retrieved_context:
            return 0.0
        scores = [cosine_similarity(trace.user_query, chunk.text) for chunk in trace.retrieved_context]
        return clamp(sum(scores) / len(scores))

    def _citation_support_coverage(self, trace: Trace, claims: list[ClaimAssessment]) -> float:
        if not claims:
            return 0.0
        covered = 0
        for assessment in claims:
            if assessment.best_chunk_id and f"[{assessment.best_chunk_id}]" in assessment.claim:
                covered += 1
            elif assessment.supported and assessment.best_chunk_id and f"[{assessment.best_chunk_id}]" in trace.llm_answer:
                covered += 1
        return covered / len(claims)

    def _missing_facts(self, trace: Trace, context: str) -> list[str]:
        if not trace.reference_answer:
            return []
        missing = []
        for fact in split_sentences(trace.reference_answer):
            if lexical_overlap(fact, context) < 0.4:
                missing.append(fact)
        return missing

    def _rank_chunks(self, trace: Trace, claims: list[ClaimAssessment]) -> list[ChunkAssessment]:
        rankings = []
        for chunk in trace.retrieved_context:
            used_claims = [claim.claim for claim in claims if claim.best_chunk_id == chunk.id and claim.supported]
            usefulness = clamp(len(used_claims) / max(len(claims), 1))
            relevance = clamp(cosine_similarity(trace.user_query, chunk.text))
            verdict = "useful" if usefulness >= 0.25 else "relevant_unused" if relevance >= 0.25 else "likely_irrelevant"
            rankings.append(
                ChunkAssessment(
                    chunk_id=chunk.id,
                    usefulness_score=round(usefulness, 3),
                    relevance_score=round(relevance, 3),
                    used_by_claims=used_claims,
                    verdict=verdict,
                )
            )
        return sorted(rankings, key=lambda item: (item.usefulness_score, item.relevance_score), reverse=True)

    def _report(self, trace: Trace, result: EvaluationResult) -> str:
        risk = "high" if result.hallucination_risk_score >= 0.6 else "medium" if result.hallucination_risk_score >= 0.3 else "low"
        top_chunk = result.chunk_rankings[0].chunk_id if result.chunk_rankings else "none"
        unsupported = len(result.unsupported_claims)
        missing = len(result.missing_facts_from_context)
        return (
            f"Trace {trace.trace_id} has {risk} hallucination risk. "
            f"Groundedness is {result.groundedness_score:.2f}, relevance is {result.relevance_score:.2f}, "
            f"and citation/support coverage is {result.citation_support_coverage:.2f}. "
            f"{unsupported} answer claim(s) appear unsupported by retrieved context. "
            f"{missing} reference fact(s) are missing from the retrieved context. "
            f"Most useful chunk: {top_chunk}."
        )

    def _call_llm_judge(self, trace: Trace, deterministic: EvaluationResult) -> dict[str, Any] | None:
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        prompt = {
            "task": "Audit this RAG trace. Return compact JSON with notes and any severe_disagreements.",
            "trace": trace.model_dump(),
            "deterministic_result": deterministic.model_dump(),
        }
        try:
            response = httpx.post(
                f"{self.config.api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.config.judge_model,
                    "messages": [
                        {"role": "system", "content": "You are a strict LLM evaluation judge. Return JSON only."},
                        {"role": "user", "content": json.dumps(prompt)},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as exc:
            return {"notes": f"LLM judge unavailable, deterministic result used. Error: {exc}"}

