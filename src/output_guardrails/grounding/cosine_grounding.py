"""Embedding-similarity grounding: faster, but cannot see contradictions.

Similar topic + similar words score high even when a number is wrong, so this can
only return GROUNDED or UNGROUNDED. Use NLI where correctness matters.
"""
from __future__ import annotations

import numpy as np

from src.output_guardrails.grounding.base import Grounder, GroundingResult, GroundingVerdict
from src.vector_store.retriever import Embedder


class CosineGrounder(Grounder):
    method = "cosine"

    def __init__(self, embedder: Embedder, threshold: float) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def score(self, source: str, claim: str) -> GroundingResult:
        vectors = self._embedder.encode([source, claim])
        similarity = round(float(np.dot(vectors[0], vectors[1])), 4)
        return GroundingResult(
            verdict=GroundingVerdict.GROUNDED if similarity >= self._threshold else GroundingVerdict.UNGROUNDED,
            entailment_score=similarity,
            contradiction_score=0.0,
            neutral_score=round(1.0 - similarity, 4),
            method=self.method,
        )
