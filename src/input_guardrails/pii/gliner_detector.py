"""GLiNER: a small model that finds names and addresses, including ones Presidio's model misses.

Used together with Presidio's pattern rules, which stay in charge of numbers such as IBANs and
cards. The model only reads a limited length at once, so long text is split into overlapping
pieces: otherwise a name further in would silently stay visible. The model loads on first use.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping

from src.input_guardrails.pii.base import PIIDetector, PIIEntity

# About 170 words, well inside the model's 384-word limit. The overlap keeps a name that
# falls across two pieces whole in at least one of them.
_WINDOW_CHARS = 1000
_OVERLAP_CHARS = 100


def _load_model(model_id: str):
    from gliner import GLiNER

    return GLiNER.from_pretrained(model_id, load_tokenizer=True)


def windows(text: str, size: int = _WINDOW_CHARS, overlap: int = _OVERLAP_CHARS) -> Iterator[tuple[int, str]]:
    """Yields (start position, piece). Pieces end at a space where possible, so words stay whole."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size - overlap, end)
            if space > start:
                end = space
        yield start, text[start:end]
        if end == len(text):
            return
        start = max(end - overlap, start + 1)


class GlinerPIIDetector(PIIDetector):
    name = "gliner"

    def __init__(self, model_id: str, threshold: float, labels: Mapping[str, str]) -> None:
        self._model_id = model_id
        self._threshold = threshold
        self._labels = dict(labels)  # what we ask the model for -> FinGuard's entity type
        self._model = None

    def detect(self, text: str) -> list[PIIEntity]:
        if self._model is None:
            self._model = _load_model(self._model_id)
        found: dict[tuple[str, int, int], PIIEntity] = {}
        for offset, piece in windows(text):
            for span in self._model.predict_entities(piece, list(self._labels), threshold=self._threshold):
                entity = self._labels[span["label"]]
                start, end = offset + span["start"], offset + span["end"]
                found[(entity, start, end)] = PIIEntity(entity, start, end, span["score"])
        return list(found.values())


class HybridPIIDetector(PIIDetector):
    """Several detectors at once. A span any of them finds gets masked."""
    name = "hybrid"

    def __init__(self, detectors: Iterable[PIIDetector]) -> None:
        self._detectors = list(detectors)

    def detect(self, text: str) -> list[PIIEntity]:
        return [entity for detector in self._detectors for entity in detector.detect(text)]
