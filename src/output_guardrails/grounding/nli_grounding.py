"""NLI grounding with a cross-encoder (PRIMARY method).

(source, claim) go through the model *together*, so it sees the logical relation
directly: "$10,000" in the source vs "$50,000" in the claim reads as contradiction,
where cosine similarity would only see two similar texts.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence

import numpy as np

from src.output_guardrails.grounding.base import Grounder, GroundingResult, classify

# Split at sentence ends only. Splitting at ";" or ":" separates a definition
# ("beneficial owner means:") from its content ("owns 25 percent or more").
_SENTENCE_SPLIT = re.compile(r"(?<=\.)\s+(?=[A-Z(])")

# When a whole law is cited it can run to 100+ windows, and among that many some
# unrelated window is bound to look contradictory. For such long sources only the
# windows sharing the most words with the claim may count as contradicting it.
_LONG_SOURCE_WINDOWS = 40
_RELEVANT_WINDOWS = 5
_TOKEN = re.compile(r"[a-z0-9]+")


def _overlap(a: str, b: str) -> float:
    ta, tb = set(_TOKEN.findall(a.lower())), set(_TOKEN.findall(b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


class NLIModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """Return {'entailment', 'contradiction', 'neutral'} probabilities per pair."""


class CrossEncoderNLI:
    """sentence-transformers CrossEncoder, loaded on first use."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._labels: list[str] = []

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)
        id2label = self._model.model.config.id2label
        self._labels = [id2label[i].lower() for i in range(len(id2label))]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        if self._model is None:
            self._load()
        logits = np.atleast_2d(self._model.predict(list(pairs), convert_to_numpy=True))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [{label: float(p) for label, p in zip(self._labels, row)} for row in probs]


def chunk_source(source: str, max_chars: int = 800) -> list[str]:
    """Split long sources into overlapping sentence windows.

    Each window repeats the last sentence of the previous one, so a rule that
    spans a window boundary is still seen whole by one window.
    """
    if len(source) <= max_chars:
        return [source]
    sentences = [s for s in _SENTENCE_SPLIT.split(source) if s]
    chunks, start = [], 0
    while start < len(sentences):
        current, end = sentences[start], start + 1
        while end < len(sentences) and len(current) + len(sentences[end]) + 1 <= max_chars:
            current = f"{current} {sentences[end]}"
            end += 1
        chunks.append(current)
        if end >= len(sentences):
            break
        start = max(start + 1, end - 1)
    return chunks


class NLIGrounder(Grounder):
    method = "nli"

    def __init__(
        self,
        entailment_threshold: float,
        contradiction_threshold: float,
        model: Optional[NLIModel] = None,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ) -> None:
        self._entail_t = entailment_threshold
        self._contra_t = contradiction_threshold
        self._model = model or CrossEncoderNLI(model_name)

    def score(self, source: str, claim: str) -> GroundingResult:
        chunks = chunk_source(source)
        results = self._model.predict([(chunk, claim) for chunk in chunks])
        # Judge the claim by the passage that supports it best. Only when no passage
        # supports it does the strongest contradiction count, so an unrelated
        # paragraph of a long regulation cannot overrule the right one.
        best = max(results, key=lambda r: r.get("entailment", 0.0))
        if best.get("entailment", 0.0) < self._entail_t:
            candidates = list(zip(chunks, results))
            if len(candidates) > _LONG_SOURCE_WINDOWS:
                candidates.sort(key=lambda cr: -_overlap(cr[0], claim))
                candidates = candidates[:_RELEVANT_WINDOWS]
            best = max((r for _, r in candidates), key=lambda r: r.get("contradiction", 0.0))
        entail = round(best.get("entailment", 0.0), 4)
        contra = round(best.get("contradiction", 0.0), 4)
        return GroundingResult(
            verdict=classify(entail, contra, self._entail_t, self._contra_t),
            entailment_score=entail,
            contradiction_score=contra,
            neutral_score=round(best.get("neutral", 0.0), 4),
            method=self.method,
        )
