"""Which agent may call which tool, per the policy allowlist. Unknown agents get nothing."""
from __future__ import annotations

from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch


class ToolAllowlist:
    layer = "tool.allowlist"

    def __init__(self, allowlist: dict[str, list[str]], policy_version: str) -> None:
        self._allowlist = {agent: frozenset(tools) for agent, tools in allowlist.items()}
        self._policy_version = policy_version

    def agents_allowed(self, tool_name: str) -> list[str]:
        return sorted(a for a, tools in self._allowlist.items() if tool_name in tools)

    def check(self, agent_name: str, tool_name: str) -> GuardrailDecision:
        sw = Stopwatch()
        allowed_tools = self._allowlist.get(agent_name)
        if allowed_tools is None:
            reason, ok = "unknown_agent", False
        elif tool_name in allowed_tools:
            reason, ok = "within_allowlist", True
        else:
            reason, ok = "tool_not_in_allowlist", False
        metadata = {"agent": agent_name, "tool": tool_name}
        if not ok:
            # Tells the caller who to hand off to, e.g. sanctions_check -> compliance_agent.
            metadata["agents_allowed_for_tool"] = self.agents_allowed(tool_name)
        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.PASS if ok else Decision.BLOCK,
            reasoning_trace=reason,
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata=metadata,
        )
