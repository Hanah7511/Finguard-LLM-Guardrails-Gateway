"""Tests for who may call FinGuard (policy v1.7.0). All keys and names are made up.

Run:  python -m pytest test/unit/test_access.py -v
"""
from __future__ import annotations

import json

import anyio
import pytest
from fastapi.testclient import TestClient
from mcp import Client
from pydantic import SecretStr

from src.api.main import app
from src.config.policy_loader import load_policy
from src.config.policy_schema import GuardrailPolicy
from src.config.settings import get_settings
from src.core.orchestrator import AccessDenied, Orchestrator, get_orchestrator
from src.core.sanctions_screener import SanctionsScreener
from src.mcp_server.server import create_server
from src.monitoring.decision_logger import DecisionLogger
from src.output_guardrails.grounding.base import Grounder, GroundingResult, GroundingVerdict

POLICY = load_policy("1.7.0")
OLD_POLICY = load_policy("1.6.0")
API_KEY = "test-app-key-not-a-real-secret"
KEYS = {"investigation_agent": "test-key-investigation-0001", "summarization_agent": "test-key-summarization-0003"}
GOOD = {"agent_name": "investigation_agent", "agent_key": KEYS["investigation_agent"], "case_id": "CASE-1"}
RISK = {"query": "Review this alert", "signals": {"structuring_flag": 1}}
CASE_TOOLS = [
    ("finguard_assess_risk", RISK),
    ("finguard_pii_and_compliance_check", {"text": "Nothing private here"}),
    ("finguard_sanctions_screen", {"entity_name": "Nobody Matchingname"}),
    ("finguard_validate_grounding", {"claims": [{"claim": "Banks report.", "citation": "31 CFR 1020.320(a)"}]}),
]


class FakeGrounder(Grounder):
    """Keeps these tests from loading the real NLI model, which is slow and uses a lot of memory."""
    method = "fake"

    def score(self, source: str, claim: str) -> GroundingResult:
        return GroundingResult(GroundingVerdict.GROUNDED, 1.0, 0.0, 0.0, self.method)


class CapturingLogger(DecisionLogger):
    def __init__(self) -> None:
        super().__init__(None)
        self.records: list[dict] = []

    def log(self, record):
        self.records.append(super().log(record))
        return self.records[-1]


def orchestrator(policy=POLICY, audit=None, api_key=API_KEY) -> Orchestrator:
    settings = get_settings().model_copy(update={"finguard_api_key": SecretStr(api_key) if api_key else None})
    test_list = SanctionsScreener.from_policy(settings.sanctions_test_list_path, policy.sanctions_screening, "test")
    return Orchestrator(policy, settings, sanctions=test_list, grounder=FakeGrounder(), audit=audit or DecisionLogger(None), agent_keys=KEYS)


def call(orch: Orchestrator, tool: str, args: dict):
    async def run():
        async with Client(create_server(orch)) as client:
            return await client.call_tool(tool, args)
    return anyio.run(run)


@pytest.fixture(scope="module")
def orch() -> Orchestrator:
    return orchestrator()


@pytest.fixture
def rest(orch):
    app.dependency_overrides[get_orchestrator] = lambda: orch
    yield TestClient(app)
    app.dependency_overrides.clear()


# The REST door

AUTHORISED_ROUTES = [
    ("get", "/policy", None),
    ("post", "/validate", {"text": "Hello"}),
    ("post", "/chat", {"query": "When is a CTR due?"}),
    ("post", "/assess-risk", {**RISK, "case_id": "CASE-1"}),
    ("post", "/sanctions-screen", {"entity_name": "Nobody Matchingname", "case_id": "CASE-1"}),
]


@pytest.mark.parametrize("method, path, body", AUTHORISED_ROUTES)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "a-wrong-key-that-is-long-enough"}])
def test_every_route_refuses_a_missing_or_wrong_key(rest, method, path, body, headers):
    r = getattr(rest, method)(path, headers=headers, **({"json": body} if body else {}))
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_or_missing_api_key"


def test_health_needs_no_key(rest):
    assert rest.get("/health").status_code == 200


def test_the_right_key_gets_through(rest):
    r = rest.post("/assess-risk", json={**RISK, "case_id": "CASE-1"}, headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert r.json()["policy_version"] == "1.7.0"


def test_a_refusal_never_shows_the_real_key(rest):
    assert API_KEY not in rest.get("/policy", headers={"X-API-Key": "a-wrong-key-that-is-long-enough"}).text


@pytest.mark.parametrize("path, body", [("/assess-risk", RISK), ("/sanctions-screen", {"entity_name": "Nobody Matchingname"})])
def test_rest_tools_that_change_a_case_need_a_case_id(rest, path, body):
    r = rest.post(path, json=body, headers={"X-API-Key": API_KEY})
    assert (r.status_code, r.json()["detail"]) == (422, "case_id_required")


def test_rest_refuses_a_bad_case_id(rest):
    r = rest.post("/validate", json={"text": "Hello", "case_id": "jane.doe@example.com"}, headers={"X-API-Key": API_KEY})
    assert (r.status_code, r.json()["detail"]) == (422, "case_id_bad_format")


def test_validate_works_without_a_case_id(rest):
    assert rest.post("/validate", json={"text": "Hello"}, headers={"X-API-Key": API_KEY}).status_code == 200


def test_the_door_refuses_to_start_without_a_key():
    with pytest.raises(RuntimeError, match="FINGUARD_API_KEY"):
        orchestrator(api_key=None).require_app_key_is_set()


def test_a_short_key_counts_as_no_key():
    with pytest.raises(RuntimeError, match="FINGUARD_API_KEY"):
        orchestrator(api_key="short").require_app_key_is_set()


def test_the_older_policy_needs_no_key():
    o = orchestrator(OLD_POLICY, api_key=None)
    o.require_app_key_is_set()
    o.check_app_key(None)


# The MCP door

@pytest.mark.parametrize("tool, args", CASE_TOOLS)
def test_case_tools_work_with_the_right_key(orch, tool, args):
    assert not call(orch, tool, {**args, **GOOD}).is_error


@pytest.mark.parametrize("tool, args", CASE_TOOLS)
@pytest.mark.parametrize("change, reason", [
    ({"agent_key": None}, "missing_agent_key"),
    ({"agent_key": "a-guessed-key-that-is-long-enough"}, "unknown_agent_key"),
    ({"agent_key": KEYS["summarization_agent"]}, "key_belongs_to_another_agent"),
])
def test_case_tools_refuse_a_bad_key(orch, tool, args, change, reason):
    r = call(orch, tool, {**args, **GOOD, **change})
    assert r.is_error
    assert reason in r.content[0].text


@pytest.mark.parametrize("tool, args", CASE_TOOLS)
def test_case_tools_need_a_case_id(orch, tool, args):
    r = call(orch, tool, {**args, "agent_name": GOOD["agent_name"], "agent_key": GOOD["agent_key"]})
    assert r.is_error
    assert "case_id_required" in r.content[0].text


@pytest.mark.parametrize("bad", ["ab", "has space", "jane.doe@example.com", "-starts-with-dash", "x" * 65, "CASE-1\n"])
def test_a_badly_formed_case_id_is_refused(orch, bad):
    r = call(orch, "finguard_assess_risk", {**RISK, **GOOD, "case_id": bad})
    assert r.is_error
    assert "case_id_bad_format" in r.content[0].text


@pytest.mark.parametrize("good", ["abc", "CASE-2026-001", "ALT_2026_001", "DEMO-1793709325"])
def test_normal_case_ids_are_accepted(orch, good):
    orch.check_case_id(good)


def test_tool_2_also_checks_the_case_id(orch):
    r = call(orch, "finguard_validate_tool_call", {
        "agent_name": "investigation_agent", "tool_name": "transaction_lookup",
        "parameters": {"transaction_id": "TXN-1"}, "agent_key": KEYS["investigation_agent"], "case_id": "has space",
    })
    assert r.is_error and "case_id_bad_format" in r.content[0].text


def test_tool_8_also_checks_the_case_id(orch):
    r = call(orch, "finguard_finalize_check", {
        "agent_name": "summarization_agent", "report_text": "Report",
        "agent_key": KEYS["summarization_agent"], "case_id": "has space",
    })
    assert r.is_error and "case_id_bad_format" in r.content[0].text


@pytest.mark.parametrize("tool, args", [("finguard_get_policy_info", {}), ("finguard_scan_documents", {"documents": ["A clean file."]})])
def test_tools_that_change_no_case_need_no_key(orch, tool, args):
    assert not call(orch, tool, args).is_error


def test_the_older_policy_keeps_working_without_keys():
    assert not call(orchestrator(OLD_POLICY), "finguard_assess_risk", RISK).is_error


# The gap this closes, and what a refusal must not do

def test_a_stranger_cannot_mark_a_case_as_grounded():
    o = orchestrator()
    args = {"claims": CASE_TOOLS[3][1]["claims"], "case_id": "CASE-1", "agent_name": "summarization_agent"}
    assert call(o, "finguard_validate_grounding", args).is_error
    assert o.cases.missing("CASE-1", ["claims_grounded"]) == ["case_not_found"]


def test_a_wrong_key_cannot_spoil_someone_elses_case():
    o = orchestrator()
    call(o, "finguard_assess_risk", {**RISK, **GOOD})
    for _ in range(3):
        call(o, "finguard_assess_risk", {**RISK, **GOOD, "agent_key": "a-guessed-key-that-is-long-enough"})
    assert o.cases.missing("CASE-1", ["risk_assessed", "no_refused_calls"]) == []


def test_a_refusal_is_audited_without_any_key():
    audit = CapturingLogger()
    o = orchestrator(audit=audit)
    guess = "a-guessed-key-that-is-long-enough"
    with pytest.raises(AccessDenied):
        o.check_agent("assess_risk", "CASE-1", "investigation_agent", guess)
    logged = json.dumps(audit.records)
    assert audit.records[-1]["event"] == "access_denied"
    assert audit.records[-1]["tool"] == "assess_risk"
    assert guess not in logged and KEYS["investigation_agent"] not in logged


# Policy rules

def test_policy_cannot_ask_for_keys_on_case_tools_without_agent_keys():
    data = POLICY.model_dump()
    data["tool_call_guardrails"]["require_agent_keys"] = False
    data["finalize_check"]["enabled"] = False
    with pytest.raises(ValueError, match="require_agent_keys"):
        GuardrailPolicy.model_validate(data)


def test_policy_refuses_a_broken_case_id_pattern():
    data = POLICY.model_dump()
    data["access_control"]["case_id_pattern"] = "["
    with pytest.raises(ValueError, match="case_id_pattern"):
        GuardrailPolicy.model_validate(data)
