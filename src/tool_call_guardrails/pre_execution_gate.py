"""Validates tool parameters before the tool runs."""
from __future__ import annotations

import re
from typing import Any, Iterator

from src.input_guardrails.patterns import PatternMatcher
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch


def _iter_strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _iter_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _iter_strings(v, f"{path}[{i}]")


class PreExecutionGate:
    layer = "tool.pre_execution"

    def __init__(
        self,
        parameter_rules: dict[str, dict[str, str]],
        injection_matcher: PatternMatcher,
        max_parameter_chars: int,
        policy_version: str,
    ) -> None:
        self._rules = {
            tool: {name: re.compile(regex) for name, regex in params.items()}
            for tool, params in parameter_rules.items()
        }
        self._matcher = injection_matcher
        self._max_chars = max_parameter_chars
        self._policy_version = policy_version

    def check(self, tool_name: str, parameters: dict[str, Any]) -> GuardrailDecision:
        sw = Stopwatch()
        violations: list[str] = []

        for name, regex in self._rules.get(tool_name, {}).items():
            if name not in parameters:
                violations.append(f"missing_required_parameter:{name}")
            elif not isinstance(parameters[name], str) or not regex.fullmatch(parameters[name]):
                violations.append(f"invalid_format:{name}")

        for path, value in _iter_strings(parameters, ""):
            if len(value) > self._max_chars:
                violations.append(f"parameter_too_long:{path}")
            if self._matcher.find_all(value):
                violations.append(f"injection_pattern_in_parameter:{path}")

        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.BLOCK if violations else Decision.PASS,
            reasoning_trace="; ".join(violations) if violations else "parameters_valid",
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata={"violations": violations} if violations else {},
        )
