from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ContextChunk(BaseModel):
    id: str = Field(..., description="Stable chunk identifier used in citations and rankings.")
    text: str = Field(..., min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    trace_id: str = Field(..., description="Human-readable trace identifier.")
    user_query: str = Field(..., min_length=1)
    retrieved_context: list[ContextChunk] = Field(default_factory=list)
    llm_answer: str = Field(..., min_length=1)
    reference_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("retrieved_context")
    @classmethod
    def chunk_ids_must_be_unique(cls, chunks: list[ContextChunk]) -> list[ContextChunk]:
        ids = [chunk.id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieved_context chunk ids must be unique")
        return chunks


class ClaimAssessment(BaseModel):
    claim: str
    supported: bool
    best_chunk_id: str | None = None
    support_score: float = Field(ge=0.0, le=1.0)


class ChunkAssessment(BaseModel):
    chunk_id: str
    usefulness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    used_by_claims: list[str] = Field(default_factory=list)
    verdict: str


class EvaluationResult(BaseModel):
    trace_id: str
    groundedness_score: float = Field(ge=0.0, le=1.0)
    hallucination_risk_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    citation_support_coverage: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str]
    missing_facts_from_context: list[str]
    chunk_rankings: list[ChunkAssessment]
    diagnostic_report: str
    judge_mode: str = Field(description="deterministic, llm, or hybrid")
    raw_judge: dict[str, Any] | None = None

