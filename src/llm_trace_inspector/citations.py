from __future__ import annotations

import re

from llm_trace_inspector.models import ContextChunk


def extract_citation_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    patterns = [
        r"\[([A-Za-z0-9_.:/#-]+)\]",
        r"\[\^([A-Za-z0-9_.:/#-]+)\]",
        r"\(([A-Za-z0-9_.:/#-]+\.(?:md|txt|html|pdf)(?::\d+)?)\)",
        r"\b([A-Za-z0-9_.-]+\.(?:md|txt|html|pdf):\d+)\b",
    ]
    for pattern in patterns:
        tokens.extend(match.group(1) for match in re.finditer(pattern, text))
    return list(dict.fromkeys(tokens))


def resolve_citations(text: str, chunks: list[ContextChunk]) -> list[str]:
    tokens = extract_citation_tokens(text)
    resolved: list[str] = []
    for token in tokens:
        for chunk in chunks:
            source = chunk.source or ""
            candidates = {
                chunk.id,
                source,
                source.split("/")[-1] if source else "",
                f"{source}:{chunk.metadata.get('page')}" if chunk.metadata.get("page") else "",
                f"{source}:{chunk.metadata.get('line')}" if chunk.metadata.get("line") else "",
            }
            if token in candidates or token.startswith(f"{chunk.id}:"):
                resolved.append(chunk.id)
    return list(dict.fromkeys(resolved))

