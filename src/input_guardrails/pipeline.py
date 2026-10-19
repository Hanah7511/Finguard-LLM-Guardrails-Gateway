"""Runs input checks cheapest-first and stops at the first block.

Order: length -> consent -> injection -> safety patterns -> PII masking -> Llama Guard.
Llama Guard sees the *masked* text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.config.policy_schema import GuardrailPolicy
from src.input_guardrails.injection_detector import InjectionDetector
from src.input_guardrails.pii.base import PIIDetector
from src.input_guardrails.safety_classifier import (
    LlamaGuardClassifier,
    PatternSafetyClassifier,
    SafetyModel,
)
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch

BLOCK_MESSAGES = {
    "input.length": "Request is too long.",
    "input.consent": "AI processing consent required.",
    "input.injection": "Request blocked: it contains instructions that attempt to override system behaviour.",
    "input.safety_patterns": "Request blocked: it appears to seek help evading financial-crime controls.",
    "input.llama_guard": "Request blocked by the safety classifier.",
}


@dataclass
class InputPipelineResult:
    allowed: bool
    processed_text: str
    decisions: list[GuardrailDecision] = field(default_factory=list)
    blocked_by: Optional[str] = None
    pii_entities: list[str] = field(default_factory=list)

    @property
    def message(self) -> Optional[str]:
        return BLOCK_MESSAGES.get(self.blocked_by) if self.blocked_by else None


class InputGuardrailPipeline:
    def __init__(
        self,
        policy: GuardrailPolicy,
        pii_detector: PIIDetector,
        safety_model: Optional[SafetyModel] = None,
    ) -> None:
        cfg = policy.input_guardrails
        self._cfg = cfg
        self._version = policy.version
        self.injection = InjectionDetector(cfg.injection_patterns, self._version)
        self.safety_patterns = PatternSafetyClassifier(cfg.safety_patterns, self._version)
        self.pii_detector = pii_detector
        self.llama_guard = LlamaGuardClassifier(cfg.llama_guard, self._version, safety_model)

    def run(
        self,
        text: str,
        consent: Optional[bool] = None,
        *,
        enforce_consent: bool = True,
        run_pii: bool = True,
        run_llama_guard: bool = True,
    ) -> InputPipelineResult:
        decisions: list[GuardrailDecision] = []

        def blocked(decision: GuardrailDecision) -> InputPipelineResult:
            decisions.append(decision)
            return InputPipelineResult(False, text, decisions, blocked_by=decision.layer)

        length = self._check_length(text)
        if length.stops_pipeline:
            return blocked(length)
        decisions.append(length)

        if enforce_consent:
            consent_decision = self._check_consent(consent)
            if consent_decision.stops_pipeline:
                return blocked(consent_decision)
            decisions.append(consent_decision)

        for check in (self.injection, self.safety_patterns):
            decision = check.check(text)
            if decision.stops_pipeline:
                return blocked(decision)
            decisions.append(decision)

        processed, entity_types = text, []
        if run_pii:
            pii_decision, processed, entity_types = self.check_pii(text)
            decisions.append(pii_decision)

        if run_llama_guard:
            guard = self.llama_guard.check(processed)
            if guard.stops_pipeline:
                decisions.append(guard)
                return InputPipelineResult(False, processed, decisions, guard.layer, entity_types)
            decisions.append(guard)

        return InputPipelineResult(True, processed, decisions, None, entity_types)

    def check_pii(self, text: str, layer: str = "input.pii") -> tuple[GuardrailDecision, str, list[str]]:
        sw = Stopwatch()
        result = self.pii_detector.mask(text)
        if result.found:
            spans = "; ".join(f"{e.entity_type} at {e.start}-{e.end}" for e in result.entities)
            trace = f"{spans}; masked"
        else:
            trace = "no_pii_found"
        decision = GuardrailDecision(
            layer=layer,
            decision=Decision.MASK if result.found else Decision.PASS,
            reasoning_trace=trace,
            policy_version=self._version,
            latency_ms=sw.ms(),
            metadata={"entities_found": result.entity_types, "backend": self.pii_detector.name},
        )
        return decision, result.masked_text, result.entity_types

    def _check_length(self, text: str) -> GuardrailDecision:
        too_long = len(text) > self._cfg.max_query_chars
        return GuardrailDecision(
            layer="input.length",
            decision=Decision.BLOCK if too_long else Decision.PASS,
            reasoning_trace=f"chars={len(text)} max={self._cfg.max_query_chars}",
            policy_version=self._version,
        )

    def _check_consent(self, consent: Optional[bool]) -> GuardrailDecision:
        if not self._cfg.require_consent:
            return GuardrailDecision("input.consent", Decision.SKIP, "consent_not_required_by_policy", self._version)
        ok = consent is True
        return GuardrailDecision(
            layer="input.consent",
            decision=Decision.PASS if ok else Decision.BLOCK,
            reasoning_trace="consent_given" if ok else "ai_processing_consent_missing",
            policy_version=self._version,
        )
