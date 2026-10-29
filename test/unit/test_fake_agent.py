"""Step 5: a scripted agent runs whole investigations through FinGuard's MCP tools.

Uses a fake grounder that believes every claim except the one known false claim, so the
test is fast; run scripts/fake_agent.py to use the real NLI model.

Run:  python -m pytest test -v
"""
from __future__ import annotations

import json

import anyio
import pytest

from scripts.fake_agent import ALERTS_PATH, CLOSED, FALSE_CLAIM, HELD, READY, SANCTIONS_HIT, build_server, run
from src.output_guardrails.grounding.base import Grounder, GroundingResult, GroundingVerdict

ALERTS = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))["alerts"]


class FakeGrounder(Grounder):
    """Supports every claim except the known false one."""
    method = "fake"

    def score(self, source: str, claim: str) -> GroundingResult:
        if claim == FALSE_CLAIM["claim"]:
            return GroundingResult(GroundingVerdict.HALLUCINATED, 0.0, 1.0, 0.0, self.method)
        return GroundingResult(GroundingVerdict.GROUNDED, 1.0, 0.0, 0.0, self.method)


@pytest.fixture(scope="module")
def server():
    return build_server(FakeGrounder())


def outcomes(server, rogue=False):
    return {inv.alert_id: inv for inv in anyio.run(run, ALERTS, server, rogue)}


def test_every_alert_reaches_the_expected_outcome(server):
    got = {k: v.outcome for k, v in outcomes(server).items()}
    assert got["ALT-2026-001"] == READY          # structuring, HIGH
    assert got["ALT-2026-002"] == CLOSED         # false alarm, LOW
    assert got["ALT-2026-011"] == SANCTIONS_HIT  # LOW risk, but a confirmed sanctions match
    assert got["ALT-2026-010"] == CLOSED         # same name as a listed person, verified different DOB
    assert all(v in (READY, CLOSED, SANCTIONS_HIT) for v in got.values())


def test_poisoned_case_file_is_dropped(server):
    inv = outcomes(server)["ALT-2026-007"]
    assert any("flagged [1]" in step for step in inv.steps)


def test_rogue_agent_is_refused_and_false_claim_holds_the_report(server):
    results = outcomes(server, rogue=True)
    assert all(inv.refused >= 1 for inv in results.values())
    assert results["ALT-2026-001"].outcome == HELD
    assert results["ALT-2026-011"].outcome == SANCTIONS_HIT

