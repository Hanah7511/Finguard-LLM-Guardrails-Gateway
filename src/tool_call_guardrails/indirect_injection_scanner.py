"""Scans retrieved documents and tool output for hidden instructions.

Same matcher as the input injection check, applied to DOCUMENTS rather than user
queries, with extra patterns that only make sense inside documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.input_guardrails.patterns import PatternMatcher
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch


@dataclass
class ScanResult:
    decision: GuardrailDecision
    clean_indices: list[int] = field(default_factory=list)
    flagged_indices: list[int] = field(default_factory=list)


class IndirectInjectionScanner:
    layer = "tool.indirect_injection"

    def __init__(self, patterns: Iterable[str], policy_version: str) -> None:
        self._matcher = PatternMatcher(patterns)
        self._policy_version = policy_version

    def scan(self, documents: list[str]) -> ScanResult:
        sw = Stopwatch()
        clean, flagged, hits = [], [], {}
        for i, doc in enumerate(documents):
            matches = self._matcher.find_all(doc)
            if matches:
                flagged.append(i)
                hits[i] = matches
            else:
                clean.append(i)
        trace = (
            "; ".join(f"doc[{i}]: {', '.join(m)}" for i, m in hits.items())
            if hits
            else f"no_hidden_instructions ({len(documents)} docs)"
        )
        decision = GuardrailDecision(
            layer=self.layer,
            decision=Decision.BLOCK if flagged else Decision.PASS,
            reasoning_trace=trace,
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata={"flagged_indices": flagged, "documents_scanned": len(documents)},
        )
        return ScanResult(decision, clean, flagged)
