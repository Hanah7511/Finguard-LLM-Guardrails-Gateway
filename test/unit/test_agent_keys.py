"""Tests for per-agent secret keys (policy v1.4.0): FinGuard works out who is calling
from the key, not from the agent name the caller claims.

The keys below are made-up test values, not real secrets.

Run:  python -m pytest test -v
"""
from __future__ import annotations

import json

import pytest

from src.config.policy_loader import load_policy
from src.config.settings import get_settings
from src.core.orchestrator import Orchestrator
from src.core.sanctions_screener import SanctionsScreener
from src.monitoring.decision_logger import DecisionLogger
from src.tool_call_guardrails.agent_identity import AgentIdentity, key_env_name

POLICY = load_policy("1.4.0")
KEYS = {
    "investigation_agent": "test-key-investigation-0001",
    "compliance_agent": "test-key-compliance-0002",
    "summarization_agent": "test-key-summarization-0003",
}
LOOKUP = {"agent_name": "investigation_agent", "tool_name": "transaction_lookup", "parameters": {"transaction_id": "TXN-1"}}


class CapturingLogger(DecisionLogger):
    """Keeps the audit records in memory so a test can read them."""

    def __init__(self) -> None:
        super().__init__(None)
        self.records: list[dict] = []

    def log(self, record):
        self.records.append(super().log(record))
        return self.records[-1]


def orchestrator(policy=POLICY, audit=None) -> Orchestrator:
    test_list = SanctionsScreener.from_policy(get_settings().sanctions_test_list_path, policy.sanctions_screening, "test")
    return Orchestrator(policy, get_settings(), sanctions=test_list, audit=audit or DecisionLogger(None), agent_keys=KEYS)


@pytest.fixture(scope="module")
def orch() -> Orchestrator:
    return orchestrator()


def test_right_key_is_approved(orch):
    r = orch.handle_validate_tool_call(**LOOKUP, agent_key=KEYS["investigation_agent"])
    assert r["approved"] is True
    assert r["verified_agent"] == "investigation_agent"


def test_missing_key_is_refused(orch):
    r = orch.handle_validate_tool_call(**LOOKUP)
    assert (r["approved"], r["reason"]) == (False, "missing_agent_key")


def test_unknown_key_is_refused(orch):
    r = orch.handle_validate_tool_call(**LOOKUP, agent_key="a-guessed-key-that-is-long-enough")
    assert (r["approved"], r["reason"]) == (False, "unknown_agent_key")


def test_agent_cannot_borrow_another_agents_name(orch):
    # The summarization agent says it is the investigation agent, but sends its own key.
    r = orch.handle_validate_tool_call(**LOOKUP, agent_key=KEYS["summarization_agent"])
    assert (r["approved"], r["reason"]) == (False, "key_belongs_to_another_agent")
    assert r["verified_agent"] == "summarization_agent"  # the audit shows who really called


def test_key_is_never_logged_or_returned():
    audit = CapturingLogger()
    o = orchestrator(audit=audit)
    r = o.handle_validate_tool_call(**LOOKUP, agent_key=KEYS["investigation_agent"])
    assert KEYS["investigation_agent"] not in json.dumps(r)
    assert KEYS["investigation_agent"] not in json.dumps(audit.records)
    assert audit.records[-1]["verified_agent"] == "investigation_agent"


def test_older_policy_needs_no_key():
    r = orchestrator(load_policy("1.3.0")).handle_validate_tool_call(**LOOKUP)
    assert r["approved"] is True
    assert r["verified_agent"] is None


def test_no_keys_configured_refuses_everyone():
    test_list = SanctionsScreener.from_policy(get_settings().sanctions_test_list_path, POLICY.sanctions_screening, "test")
    o = Orchestrator(POLICY, get_settings(), sanctions=test_list, audit=DecisionLogger(None), agent_keys={})
    assert o.handle_validate_tool_call(**LOOKUP, agent_key=KEYS["investigation_agent"])["approved"] is False


def test_two_agents_cannot_share_a_key():
    with pytest.raises(ValueError, match="same key"):
        AgentIdentity({"a_agent": "same-key-for-both-agents", "b_agent": "same-key-for-both-agents"}, "1.4.0")


def test_short_key_is_not_accepted():
    identity = AgentIdentity({"investigation_agent": "short"}, "1.4.0")
    assert identity.owner("short") is None


def test_keys_are_read_from_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_KEY_INVESTIGATION", raising=False)
    env = tmp_path / ".env"
    env.write_text("AGENT_KEY_INVESTIGATION=from-the-env-file-0001\n", encoding="utf-8")
    identity = AgentIdentity.from_env(["investigation_agent"], "1.4.0", env_file=str(env))
    assert identity.owner("from-the-env-file-0001") == "investigation_agent"
    assert key_env_name("summarization_agent") == "AGENT_KEY_SUMMARIZATION"
