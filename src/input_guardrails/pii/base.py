"""Swappable PII detector interface. Implement ``detect`` to plug in a new backend."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


# Text that is already masked, such as [PERSON_1] or [IBAN_CODE]. It must not be masked again.
_ALREADY_MASKED = re.compile(r"\[[A-Z][A-Z_]*(?:_\d+)?\]")


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

    def mask(self, text: str, token_for: Optional[Callable[[str, str], str]] = None) -> PIIResult:
        """Replace each detected span with ``[ENTITY_TYPE]``, or with ``token_for(type, value)`` if given."""
        masked_spans = [m.span() for m in _ALREADY_MASKED.finditer(text)]
        found = [e for e in self.detect(text) if all(e.end <= a or e.start >= b for a, b in masked_spans)]
        kept = sorted(_resolve_overlaps(found), key=lambda e: e.start)
        # Tokens are made in reading order, so the first value in the text gets number 1.
        labels = [token_for(e.entity_type, text[e.start : e.end]) if token_for else f"[{e.entity_type}]" for e in kept]
        masked = text
        for ent, label in reversed(list(zip(kept, labels))):
            masked = masked[: ent.start] + label + masked[ent.end :]
        return PIIResult(masked_text=masked, entities=kept)


def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """Keep the highest-scoring (then longest) span where spans overlap."""
    kept: list[PIIEntity] = []
    for ent in sorted(entities, key=lambda e: (-e.score, -(e.end - e.start), e.start)):
        if all(ent.end <= k.start or ent.start >= k.end for k in kept):
            kept.append(ent)
    return kept
