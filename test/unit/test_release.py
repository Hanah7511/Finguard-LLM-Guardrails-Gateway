"""Tests for analyst release (policy v1.8.0): the AI side never receives real values.
All keys, names and values are made up.

Run:  python -m pytest test/unit/test_release.py -v
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

POLICY = load_policy("1.8.0")
API_KEY = "test-app-key-not-a-real-secret"
ANALYST_KEY = "test-analyst-key-not-a-real-secret"
ANALYSTS = {"primary": ANALYST_KEY}
KEYS = {"investigation_agent": "test-key-investigation-0001", "summarization_agent": "test-key-summarization-0003"}
IBAN = "GB82 WEST 1234 5698 7654 32"
CLAIMS = [{"claim": "Banks must report suspicious transactions.", "citation": "31 CFR 1020.320(a)"}]
FALSE = "this claim is false"


class FakeGrounder(Grounder):
    method = "fake"

    def score(self, source: str, claim: str) -> GroundingResult:
        verdict = GroundingVerdict.HALLUCINATED if claim == FALSE else GroundingVerdict.GROUNDED
        return GroundingResult(verdict, 1.0, 0.0, 0.0, self.method)


class CapturingLogger(DecisionLogger):
    def __init__(self) -> None:
        super().__init__(None)
        self.records: list[dict] = []

    def log(self, record):
        self.records.append(super().log(record))
        return self.records[-1]


def orchestrator(policy=POLICY, audit=None, analyst_keys=ANALYSTS) -> Orchestrator:
    settings = get_settings().model_copy(update={"finguard_api_key": SecretStr(API_KEY)})
    test_list = SanctionsScreener.from_policy(settings.sanctions_test_list_path, policy.sanctions_screening, "test")
    return Orchestrator(
        policy, settings, sanctions=test_list, grounder=FakeGrounder(), audit=audit or DecisionLogger(None),
        agent_keys=KEYS, analyst_keys=analyst_keys,
    )


def approved_report(o: Orchestrator, case: str = "CASE-1") -> str:
    """Runs an honest case to Tool 8 and returns the masked report it approved."""
    o.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id=case)
    masked = o.handle_pii_check(f"Deposit from IBAN {IBAN}.", case)["masked_text"]
    o.handle_validate_grounding(CLAIMS, case)
    r = o.handle_finalize_check("summarization_agent", masked, case, KEYS["summarization_agent"])
    assert r["approved"] is True
    return r["report_text"]


@pytest.fixture
def orch() -> Orchestrator:
    return orchestrator()


# Tool 8 no longer hands out real values

def test_an_approved_report_stays_masked_for_the_agent(orch):
    orch.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id="CASE-1")
    masked = orch.handle_pii_check(f"Deposit from IBAN {IBAN}.", "CASE-1")["masked_text"]
    orch.handle_validate_grounding(CLAIMS, "CASE-1")
    r = orch.handle_finalize_check("summarization_agent", masked, "CASE-1", KEYS["summarization_agent"])
    assert (r["approved"], r["analyst_release_needed"], r["tokens_restored"]) == (True, True, 0)
    assert r["report_text"] == "Deposit from IBAN [IBAN_CODE_1]."
    assert IBAN not in json.dumps(r)


def test_a_real_value_written_into_the_report_is_masked_for_the_agent(orch):
    orch.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id="CASE-1")
    orch.handle_pii_check(f"Deposit from IBAN {IBAN}.", "CASE-1")
    orch.handle_validate_grounding(CLAIMS, "CASE-1")
    r = orch.handle_finalize_check("summarization_agent", f"Deposit from IBAN {IBAN}.", "CASE-1", KEYS["summarization_agent"])
    assert IBAN not in json.dumps(r)
    assert r["values_caught"] == 1


def test_the_mcp_door_has_no_release_tool(orch):
    async def run():
        async with Client(create_server(orch)) as client:
            return (await client.list_tools()).tools
    tools = anyio.run(run)
    assert len(tools) == 8
    assert not [t.name for t in tools if "release" in t.name]


def test_the_older_policy_still_returns_real_values_from_tool_8():
    o = orchestrator(load_policy("1.7.0"))
    assert approved_report(o) == f"Deposit from IBAN {IBAN}."


# Release

def test_the_analyst_gets_the_real_values(orch):
    r = orch.handle_release("CASE-1", approved_report(orch), ANALYST_KEY)
    assert (r["released"], r["report_text"], r["tokens_restored"]) == (True, f"Deposit from IBAN {IBAN}.", 1)


@pytest.mark.parametrize("key", [None, "a-wrong-key-that-is-long-enough", KEYS["summarization_agent"], API_KEY])
def test_only_an_analyst_key_works(orch, key):
    report = approved_report(orch)
    with pytest.raises(AccessDenied, match="invalid_analyst_key"):
        orch.handle_release("CASE-1", report, key)


def test_a_report_tool_8_never_approved_is_refused(orch):
    approved_report(orch)
    r = orch.handle_release("CASE-1", "Different text with [IBAN_CODE_1].", ANALYST_KEY)
    assert r["released"] is False
    assert r["missing"] == ["report_not_approved"]
    assert r["report_text"] is None


def test_a_report_approved_in_another_case_is_refused(orch):
    report = approved_report(orch, "CASE-A")
    orch.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id="CASE-B")
    orch.handle_validate_grounding(CLAIMS, "CASE-B")
    r = orch.handle_release("CASE-B", report, ANALYST_KEY)
    assert r["released"] is False
    assert IBAN not in json.dumps(r)


def test_a_case_never_seen_is_refused(orch):
    r = orch.handle_release("CASE-UNKNOWN", "Report", ANALYST_KEY)
    assert r["released"] is False
    assert "case_not_found" in r["missing"]


def test_a_sanctions_block_after_approval_stops_the_release(orch):
    report = approved_report(orch)
    orch.handle_sanctions_screen("Khalid Testani", case_id="CASE-1", dob="1975-05-05")
    r = orch.handle_release("CASE-1", report, ANALYST_KEY)
    assert r["released"] is False
    assert r["missing"] == ["no_sanctions_block"]


def test_failed_grounding_after_approval_stops_the_release(orch):
    report = approved_report(orch)
    orch.handle_validate_grounding([{"claim": FALSE, "citation": "31 CFR 1020.320(a)"}], "CASE-1")
    r = orch.handle_release("CASE-1", report, ANALYST_KEY)
    assert r["missing"] == ["claims_grounded"]


def test_a_bad_case_id_is_refused(orch):
    with pytest.raises(AccessDenied, match="case_id_bad_format"):
        orch.handle_release("has space", "Report", ANALYST_KEY)


def test_the_older_policy_has_no_release():
    with pytest.raises(AccessDenied, match="release_is_off"):
        orchestrator(load_policy("1.7.0")).handle_release("CASE-1", "Report", ANALYST_KEY)


# Audit trail

def test_the_audit_log_says_who_released_and_never_what():
    audit = CapturingLogger()
    o = orchestrator(audit=audit)
    report = approved_report(o)
    o.handle_release("CASE-1", report, ANALYST_KEY)
    last = audit.records[-1]
    assert (last["event"], last["analyst"], last["released"]) == ("release", "primary", True)
    logged = json.dumps(audit.records)
    assert IBAN not in logged and ANALYST_KEY not in logged


def test_a_refused_release_is_audited():
    audit = CapturingLogger()
    o = orchestrator(audit=audit)
    with pytest.raises(AccessDenied):
        o.handle_release("CASE-1", "Report", "a-wrong-key-that-is-long-enough")
    assert (audit.records[-1]["event"], audit.records[-1]["released"]) == ("release", False)


def test_if_the_audit_write_fails_no_values_are_returned():
    class BrokenLogger(DecisionLogger):
        def log(self, record):
            if record.get("event") == "release":
                raise OSError("audit store is down")
            return super().log(record)

    o = orchestrator(audit=BrokenLogger(None))
    report = approved_report(o)
    with pytest.raises(OSError):
        o.handle_release("CASE-1", report, ANALYST_KEY)


# The REST endpoint

@pytest.fixture
def rest(orch):
    app.dependency_overrides[get_orchestrator] = lambda: orch
    yield TestClient(app)
    app.dependency_overrides.clear()


BOTH = {"X-API-Key": API_KEY, "X-Analyst-Key": ANALYST_KEY}


def test_the_endpoint_releases_and_says_do_not_cache(orch, rest):
    report = approved_report(orch)
    r = rest.post("/cases/CASE-1/release", json={"report_text": report}, headers=BOTH)
    assert r.status_code == 200
    assert r.json()["report_text"] == f"Deposit from IBAN {IBAN}."
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("headers, status, detail", [
    ({}, 401, "invalid_or_missing_api_key"),
    ({"X-Analyst-Key": ANALYST_KEY}, 401, "invalid_or_missing_api_key"),
    ({"X-API-Key": API_KEY}, 401, "invalid_analyst_key"),
    ({"X-API-Key": API_KEY, "X-Analyst-Key": "a-wrong-key-that-is-long-enough"}, 401, "invalid_analyst_key"),
])
def test_the_endpoint_needs_both_keys(orch, rest, headers, status, detail):
    r = rest.post("/cases/CASE-1/release", json={"report_text": approved_report(orch)}, headers=headers)
    assert (r.status_code, r.json()["detail"]) == (status, detail)
    assert IBAN not in r.text


def test_the_endpoint_answers_409_when_it_cannot_release(orch, rest):
    approved_report(orch)
    r = rest.post("/cases/CASE-1/release", json={"report_text": "Different [IBAN_CODE_1]"}, headers=BOTH)
    assert r.status_code == 409
    assert r.json()["detail"]["missing"] == ["report_not_approved"]
    assert IBAN not in r.text


def test_the_endpoint_refuses_a_bad_case_id(rest):
    r = rest.post("/cases/has%20space/release", json={"report_text": "Report"}, headers=BOTH)
    assert (r.status_code, r.json()["detail"]) == (422, "case_id_bad_format")


# Starting up and policy rules

def test_the_door_refuses_to_start_without_an_analyst_key():
    with pytest.raises(RuntimeError, match="primary"):
        orchestrator(analyst_keys={}).require_analyst_key_is_set()


def test_policy_cannot_hide_real_values_with_no_way_to_release_them():
    data = POLICY.model_dump()
    data["analyst_release"]["enabled"] = False
    with pytest.raises(ValueError, match="return_real_values"):
        GuardrailPolicy.model_validate(data)


def test_policy_cannot_turn_on_release_with_no_analysts():
    data = POLICY.model_dump()
    data["analyst_release"]["allowed_analysts"] = []
    with pytest.raises(ValueError, match="allowed analyst"):
        GuardrailPolicy.model_validate(data)
