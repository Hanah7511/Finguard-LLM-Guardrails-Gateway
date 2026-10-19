"""Runs tool-call checks: allowlist -> parameters before execution; size/PII/injection after."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.policy_schema import GuardrailPolicy
from src.input_guardrails.patterns import PatternMatcher
from src.input_guardrails.pii.base import PIIDetector
from src.monitoring.telemetry import GuardrailDecision
from src.tool_call_guardrails.indirect_injection_scanner import IndirectInjectionScanner, ScanResult
from src.tool_call_guardrails.post_execution_gate import PostExecutionGate, PostExecutionResult
from src.tool_call_guardrails.pre_execution_gate import PreExecutionGate
from src.tool_call_guardrails.tool_allowlist import ToolAllowlist


@dataclass
class ToolCallValidation:
    approved: bool
    reason: str
    decisions: list[GuardrailDecision] = field(default_factory=list)
    agents_allowed_for_tool: list[str] = field(default_factory=list)


class ToolCallGuardrailPipeline:
    def __init__(self, policy: GuardrailPolicy, pii_detector: PIIDetector) -> None:
        cfg = policy.tool_call_guardrails
        version = policy.version
        injection_patterns = policy.input_guardrails.injection_patterns
        self.allowlist = ToolAllowlist(cfg.allowlist, version)
        self.pre_gate = PreExecutionGate(
            cfg.parameter_rules, PatternMatcher(injection_patterns), cfg.max_parameter_chars, version
        )
        self.scanner = IndirectInjectionScanner(
            [*injection_patterns, *cfg.indirect_injection_patterns], version
        )
        self.post_gate = PostExecutionGate(pii_detector, self.scanner, cfg.max_result_chars, version)

    def validate_call(self, agent_name: str, tool_name: str, parameters: dict[str, Any]) -> ToolCallValidation:
        allow = self.allowlist.check(agent_name, tool_name)
        if allow.stops_pipeline:
            return ToolCallValidation(
                False,
                allow.reasoning_trace,
                [allow],
                allow.metadata.get("agents_allowed_for_tool", []),
            )
        params = self.pre_gate.check(tool_name, parameters)
        if params.stops_pipeline:
            return ToolCallValidation(False, params.reasoning_trace, [allow, params])
        return ToolCallValidation(True, "within_allowlist", [allow, params])

    def validate_result(self, result_text: str) -> PostExecutionResult:
        return self.post_gate.check(result_text)

    def scan_documents(self, documents: list[str]) -> ScanResult:
        return self.scanner.scan(documents)
