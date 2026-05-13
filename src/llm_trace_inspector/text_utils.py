from __future__ import annotations

import re
from collections import Counter

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text.lower()) if token not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if len(tokenize(part)) >= 3]


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def cosine_similarity(a: str, b: str) -> float:
    a_counts = Counter(tokenize(a))
    b_counts = Counter(tokenize(b))
    if not a_counts or not b_counts:
        return 0.0
    shared = set(a_counts) & set(b_counts)
    numerator = sum(a_counts[token] * b_counts[token] for token in shared)
    a_norm = sum(value * value for value in a_counts.values()) ** 0.5
    b_norm = sum(value * value for value in b_counts.values()) ** 0.5
    return numerator / (a_norm * b_norm)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

