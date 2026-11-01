"""Runs tool-call checks: identity -> allowlist -> parameters before execution; size/PII/injection after."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.config.policy_schema import GuardrailPolicy
from src.input_guardrails.patterns import PatternMatcher
from src.input_guardrails.pii.base import PIIDetector
from src.monitoring.telemetry import GuardrailDecision
from src.tool_call_guardrails.agent_identity import AgentIdentity
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
    verified_agent: Optional[str] = None  # set only when agent keys are required


class ToolCallGuardrailPipeline:
    def __init__(self, policy: GuardrailPolicy, pii_detector: PIIDetector, identity: Optional[AgentIdentity] = None) -> None:
        cfg = policy.tool_call_guardrails
        version = policy.version
        # With keys required but none given, every call is refused (fail closed).
        self.identity = (identity or AgentIdentity({}, version)) if cfg.require_agent_keys else None
        injection_patterns = policy.input_guardrails.injection_patterns
        self.allowlist = ToolAllowlist(cfg.allowlist, version)
        self.pre_gate = PreExecutionGate(
            cfg.parameter_rules, PatternMatcher(injection_patterns), cfg.max_parameter_chars, version
        )
        self.scanner = IndirectInjectionScanner(
            [*injection_patterns, *cfg.indirect_injection_patterns], version
        )
        self.post_gate = PostExecutionGate(pii_detector, self.scanner, cfg.max_result_chars, version)

    def validate_call(
        self, agent_name: str, tool_name: str, parameters: dict[str, Any], agent_key: Optional[str] = None
    ) -> ToolCallValidation:
        decisions, verified = [], None
        if self.identity:
            who = self.identity.check(agent_name, agent_key)
            decisions.append(who)
            verified = who.metadata["verified_agent"]  # who really called, even if refused
            if who.stops_pipeline:
                return ToolCallValidation(False, who.reasoning_trace, decisions, verified_agent=verified)
        allow = self.allowlist.check(agent_name, tool_name)
        decisions.append(allow)
        if allow.stops_pipeline:
            return ToolCallValidation(
                False, allow.reasoning_trace, decisions, allow.metadata.get("agents_allowed_for_tool", []), verified
            )
        params = self.pre_gate.check(tool_name, parameters)
        decisions.append(params)
        if params.stops_pipeline:
            return ToolCallValidation(False, params.reasoning_trace, decisions, verified_agent=verified)
        return ToolCallValidation(True, "within_allowlist", decisions, verified_agent=verified)

    def validate_result(self, result_text: str) -> PostExecutionResult:
        return self.post_gate.check(result_text)

    def scan_documents(self, documents: list[str]) -> ScanResult:
        return self.scanner.scan(documents)
