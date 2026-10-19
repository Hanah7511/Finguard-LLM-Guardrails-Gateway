"""Validates a tool's result before it is fed back to the agent."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.input_guardrails.pii.base import PIIDetector
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch
from src.tool_call_guardrails.indirect_injection_scanner import IndirectInjectionScanner


@dataclass
class PostExecutionResult:
    safe_to_use: bool
    masked_text: str
    pii_found: list[str] = field(default_factory=list)
    decisions: list[GuardrailDecision] = field(default_factory=list)


class PostExecutionGate:
    def __init__(
        self,
        pii_detector: PIIDetector,
        scanner: IndirectInjectionScanner,
        max_result_chars: int,
        policy_version: str,
    ) -> None:
        self._pii = pii_detector
        self._scanner = scanner
        self._max_chars = max_result_chars
        self._policy_version = policy_version

    def check(self, result_text: str) -> PostExecutionResult:
        decisions: list[GuardrailDecision] = []

        too_long = len(result_text) > self._max_chars
        decisions.append(
            GuardrailDecision(
                layer="tool.post_execution.size",
                decision=Decision.BLOCK if too_long else Decision.PASS,
                reasoning_trace=f"chars={len(result_text)} max={self._max_chars}",
                policy_version=self._policy_version,
            )
        )

        # Always mask, even if the result is then rejected: the masked text is what
        # callers may log or show to an analyst.
        sw = Stopwatch()
        pii = self._pii.mask(result_text)
        decisions.append(
            GuardrailDecision(
                layer="tool.post_execution.pii",
                decision=Decision.MASK if pii.found else Decision.PASS,
                reasoning_trace=f"masked: {', '.join(pii.entity_types)}" if pii.found else "no_pii_found",
                policy_version=self._policy_version,
                latency_ms=sw.ms(),
                metadata={"entities_found": pii.entity_types},
            )
        )

        scan = self._scanner.scan([result_text])
        decisions.append(scan.decision)

        safe = not too_long and not scan.flagged_indices
        return PostExecutionResult(safe, pii.masked_text, pii.entity_types, decisions)
