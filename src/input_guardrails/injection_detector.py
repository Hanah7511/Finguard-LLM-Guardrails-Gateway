"""Prompt-injection detection by pattern matching. No model, cheapest check."""
from __future__ import annotations

from typing import Iterable

from src.input_guardrails.patterns import PatternMatcher
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch


class InjectionDetector:
    layer = "input.injection"

    def __init__(self, patterns: Iterable[str], policy_version: str) -> None:
        self._matcher = PatternMatcher(patterns)
        self._policy_version = policy_version

    def check(self, text: str) -> GuardrailDecision:
        sw = Stopwatch()
        matches = self._matcher.find_all(text)
        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.BLOCK if matches else Decision.PASS,
            reasoning_trace=f"pattern_matched: {', '.join(matches)}" if matches else "no_patterns_found",
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata={"patterns_matched": matches} if matches else {},
        )
