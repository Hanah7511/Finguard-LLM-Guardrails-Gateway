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

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+")


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


def chunk_source(source: str, max_chars: int = 1200) -> list[str]:
    """Split long sources into sentence windows that fit the model's context."""
    if len(source) <= max_chars:
        return [source]
    chunks, current = [], ""
    for sentence in _SENTENCE_SPLIT.split(source):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
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
        # The most relevant chunk is the one the model is least neutral about.
        best = min(results, key=lambda r: r.get("neutral", 0.0))
        entail = round(best.get("entailment", 0.0), 4)
        contra = round(best.get("contradiction", 0.0), 4)
        return GroundingResult(
            verdict=classify(entail, contra, self._entail_t, self._contra_t),
            entailment_score=entail,
            contradiction_score=contra,
            neutral_score=round(best.get("neutral", 0.0), 4),
            method=self.method,
        )
