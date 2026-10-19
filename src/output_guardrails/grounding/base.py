"""Swappable grounding interface. Implement ``score`` to add a new method."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum


class GroundingVerdict(str, Enum):
    GROUNDED = "GROUNDED"          # source supports the claim
    HALLUCINATED = "HALLUCINATED"  # source contradicts the claim -> BLOCK
    UNGROUNDED = "UNGROUNDED"      # neither -> ESCALATE to a human


@dataclass
class GroundingResult:
    verdict: GroundingVerdict
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    method: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


def classify(entailment: float, contradiction: float, entailment_threshold: float, contradiction_threshold: float) -> GroundingVerdict:
    """Contradiction is checked first: a contradicted claim is never 'grounded'."""
    if contradiction >= contradiction_threshold:
        return GroundingVerdict.HALLUCINATED
    if entailment >= entailment_threshold:
        return GroundingVerdict.GROUNDED
    return GroundingVerdict.UNGROUNDED


class Grounder(ABC):
    method: str = "base"

    @abstractmethod
    def score(self, source: str, claim: str) -> GroundingResult:
        """Judge whether ``source`` supports ``claim``."""
