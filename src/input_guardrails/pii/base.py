"""Swappable PII detector interface. Implement ``detect`` to plug in a new backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIIEntity:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class PIIResult:
    masked_text: str
    entities: list[PIIEntity] = field(default_factory=list)

    @property
    def entity_types(self) -> list[str]:
        return sorted({e.entity_type for e in self.entities})

    @property
    def found(self) -> bool:
        return bool(self.entities)


class PIIDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, text: str) -> list[PIIEntity]:
        """Return all PII spans in ``text`` (may overlap)."""

    def mask(self, text: str) -> PIIResult:
        """Replace each detected span with ``[ENTITY_TYPE]``."""
        kept = _resolve_overlaps(self.detect(text))
        masked = text
        for ent in sorted(kept, key=lambda e: e.start, reverse=True):
            masked = masked[: ent.start] + f"[{ent.entity_type}]" + masked[ent.end :]
        return PIIResult(masked_text=masked, entities=sorted(kept, key=lambda e: e.start))


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep the highest-scoring (then longest) span where spans overlap."""
    kept: list[PIIEntity] = []
    for ent in sorted(entities, key=lambda e: (-e.score, -(e.end - e.start), e.start)):
        if all(ent.end <= k.start or ent.start >= k.end for k in kept):
            kept.append(ent)
    return kept
