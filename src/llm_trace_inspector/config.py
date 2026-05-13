from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from llm_trace_inspector.models import EvaluationResult


class ThresholdConfig(BaseModel):
    max_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    min_groundedness: float | None = Field(default=None, ge=0.0, le=1.0)
    min_citation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)


class InspectorConfig(BaseModel):
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)


def load_config(path: Path | None) -> InspectorConfig:
    if path is None:
        default_path = Path("llm-trace-inspector.toml")
        if not default_path.exists():
            return InspectorConfig()
        path = default_path
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return InspectorConfig.model_validate(data)


def threshold_failures(result: EvaluationResult, thresholds: ThresholdConfig) -> list[str]:
    failures = []
    if thresholds.max_risk is not None and result.hallucination_risk_score > thresholds.max_risk:
        failures.append(
            f"hallucination risk {result.hallucination_risk_score:.3f} > {thresholds.max_risk:.3f}"
        )
    if thresholds.min_groundedness is not None and result.groundedness_score < thresholds.min_groundedness:
        failures.append(
            f"groundedness {result.groundedness_score:.3f} < {thresholds.min_groundedness:.3f}"
        )
    if (
        thresholds.min_citation_coverage is not None
        and result.citation_support_coverage < thresholds.min_citation_coverage
    ):
        failures.append(
            "citation coverage "
            f"{result.citation_support_coverage:.3f} < {thresholds.min_citation_coverage:.3f}"
        )
    return failures

