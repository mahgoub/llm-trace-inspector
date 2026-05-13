from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_trace_inspector.evaluator import TraceEvaluator  # noqa: E402
from llm_trace_inspector.models import Trace  # noqa: E402

st.set_page_config(page_title="LLM Trace Inspector", layout="wide")
st.title("LLM Trace Inspector")
st.caption("RAG and agent trace diagnostics for groundedness, hallucination risk, relevance, and support coverage.")

uploaded = st.file_uploader("Upload trace JSON", type=["json"])
example_path = ROOT / "examples" / "rag_trace_good.json"

if uploaded:
    data = json.loads(uploaded.read().decode("utf-8"))
else:
    data = json.loads(example_path.read_text(encoding="utf-8"))

trace = Trace.model_validate(data)
result = TraceEvaluator().evaluate(trace)

cols = st.columns(4)
cols[0].metric("Groundedness", f"{result.groundedness_score:.2f}")
cols[1].metric("Hallucination risk", f"{result.hallucination_risk_score:.2f}")
cols[2].metric("Relevance", f"{result.relevance_score:.2f}")
cols[3].metric("Support coverage", f"{result.citation_support_coverage:.2f}")

st.subheader("Diagnostic Report")
st.write(result.diagnostic_report)

left, right = st.columns(2)
with left:
    st.subheader("Unsupported Claims")
    if result.unsupported_claims:
        for claim in result.unsupported_claims:
            st.error(claim)
    else:
        st.success("No unsupported claims detected by deterministic heuristics.")

with right:
    st.subheader("Missing Facts From Context")
    if result.missing_facts_from_context:
        for fact in result.missing_facts_from_context:
            st.warning(fact)
    else:
        st.success("No reference facts appear missing from retrieved context.")

st.subheader("Chunk Usefulness Ranking")
st.dataframe([row.model_dump() for row in result.chunk_rankings], use_container_width=True)

with st.expander("Input trace"):
    st.json(trace.model_dump())

