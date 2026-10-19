"""Never auto-approve an uncertain answer: below threshold goes to human review."""
from __future__ import annotations

from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch


class ConfidenceGate:
    layer = "output.confidence_gate"

    def __init__(self, minimum_confidence: float, policy_version: str) -> None:
        self._threshold = minimum_confidence
        self._policy_version = policy_version

    def check(self, confidence_score: float) -> GuardrailDecision:
        sw = Stopwatch()
        ok = confidence_score >= self._threshold
        op = ">=" if ok else "<"
        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.APPROVE if ok else Decision.ESCALATE,
            reasoning_trace=f"score={confidence_score:.2f} {op} threshold={self._threshold:.2f}",
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
        )
