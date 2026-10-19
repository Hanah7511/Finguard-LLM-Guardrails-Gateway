"""Microsoft Presidio PII backend, plus the factory that picks a backend."""
from __future__ import annotations

import logging
from typing import Iterable

from src.config.policy_schema import CustomPIIRecognizer, InputGuardrailsPolicy
from src.input_guardrails.pii.base import PIIDetector, PIIEntity
from src.input_guardrails.pii.regex_detector import RegexPIIDetector

logger = logging.getLogger(__name__)


class PresidioPIIDetector(PIIDetector):
    name = "presidio"

    def __init__(
        self,
        entities: Iterable[str],
        custom_recognizers: Iterable[CustomPIIRecognizer] = (),
        score_threshold: float = 0.4,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        )
        self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        custom = list(custom_recognizers)
        for rec in custom:
            # Local account formats not covered by Presidio's built-ins.
            self._analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=rec.entity_type,
                    patterns=[Pattern(name=f"{rec.entity_type.lower()}_pattern", regex=rec.regex, score=rec.score)],
                    context=rec.context or None,
                )
            )
        self._entities = list(dict.fromkeys([*entities, *(r.entity_type for r in custom)]))
        self._threshold = score_threshold

    def detect(self, text: str) -> list[PIIEntity]:
        results = self._analyzer.analyze(
            text=text, entities=self._entities, language="en", score_threshold=self._threshold
        )
        return [PIIEntity(r.entity_type, r.start, r.end, r.score) for r in results]


def build_pii_detector(policy: InputGuardrailsPolicy, backend: str = "auto", spacy_model: str = "en_core_web_sm") -> PIIDetector:
    if backend in ("auto", "presidio"):
        try:
            return PresidioPIIDetector(
                policy.pii_entities,
                policy.custom_pii_recognizers,
                policy.pii_score_threshold,
                spacy_model,
            )
        except Exception as exc:
            if backend == "presidio":
                raise
            logger.warning("Presidio unavailable (%s); using regex PII detector", exc)
    return RegexPIIDetector(policy.pii_entities, policy.custom_pii_recognizers)
