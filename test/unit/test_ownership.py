"""Tests for the OFAC 50% ownership rule (policy v1.3.0).

Owners are screened against the FICTIONAL test list: Khalid Testani (born 1975-05-05)
and Jana Demonstrova (born 1985-07-22) are listed people, Northwind Shell Trading LLC
is a listed company. Every other name here is invented and on no list.

Run:  python -m pytest test -v
"""
from __future__ import annotations

import pytest

from src.config.policy_loader import load_policy
from src.config.settings import get_settings
from src.core.orchestrator import Orchestrator
from src.core.ownership import OwnershipChecker
from src.core.sanctions_screener import SanctionsScreener
from src.monitoring.decision_logger import DecisionLogger

POLICY = load_policy("1.3.0")
KHALID = {"name": "Khalid Testani", "dob": "1975-05-05"}
JANA = {"name": "Jana Demonstrova", "dob": "1985-07-22"}


def fictional_list(policy=POLICY) -> SanctionsScreener:
    return SanctionsScreener.from_policy(get_settings().sanctions_test_list_path, policy.sanctions_screening, "test")


@pytest.fixture(scope="module")
def checker() -> OwnershipChecker:
    return OwnershipChecker(fictional_list(), POLICY.sanctions_screening.ownership_rule)


def test_two_blocked_owners_add_up_to_50(checker):
    r = checker.check([{**KHALID, "percent": 30}, {**JANA, "percent": 20}, {"name": "Clara Openfield", "percent": 50}])
    assert r.blocked_share == 50
    assert r.recommendation == "block_and_escalate"


def test_blocked_owner_below_50_goes_to_review(checker):
    r = checker.check([{**KHALID, "percent": 30}, {**JANA, "percent": 19}, {"name": "Clara Openfield", "percent": 51}])
    assert r.blocked_share == 49
    assert r.recommendation == "escalate_for_review"


def test_clean_owners_clear(checker):
    r = checker.check([{"name": "Clara Openfield", "percent": 60}, {"name": "Omar Brightwater", "percent": 40}])
    assert r.recommendation == "clear"
    assert r.blocked_share == 0


def test_blocked_company_counts_in_full_down_a_chain(checker):
    # Khalid owns 60% of Holding A, so Holding A is blocked; its 50% of the target counts as 50%, not 30%.
    holding_a = {"name": "Fictiva Holding A Ltd", "percent": 50, "owners": [{**KHALID, "percent": 60}, {"name": "Clara Openfield", "percent": 40}]}
    r = checker.check([holding_a, {"name": "Omar Brightwater", "percent": 50}])
    assert r.owners[0].status == "blocked"
    assert r.blocked_share == 50
    assert r.recommendation == "block_and_escalate"


def test_listed_company_as_owner(checker):
    r = checker.check([{"name": "Northwind Shell Trading LLC", "percent": 50}, {"name": "Clara Openfield", "percent": 50}])
    assert r.owners[0].status == "blocked"
    assert r.recommendation == "block_and_escalate"


def test_name_only_matches_can_reach_50_so_review(checker):
    # No dates of birth: the matches are only "possible", so review instead of a block.
    r = checker.check([{"name": "Khalid Testani", "percent": 60}, {"name": "Clara Openfield", "percent": 40}])
    assert (r.blocked_share, r.possible_share) == (0, 60)
    assert r.recommendation == "escalate_for_review"


def test_exactly_50_under_the_eu_style_rule_is_not_blocked():
    eu_rule = POLICY.sanctions_screening.ownership_rule.model_copy(update={"inclusive": False})
    r = OwnershipChecker(fictional_list(), eu_rule).check([{**KHALID, "percent": 50}, {"name": "Clara Openfield", "percent": 50}])
    assert r.recommendation == "escalate_for_review"


def test_shares_over_100_are_rejected(checker):
    with pytest.raises(ValueError, match="more than 100%"):
        checker.check([{"name": "Clara Openfield", "percent": 70}, {"name": "Omar Brightwater", "percent": 40}])


def test_orchestrator_blocks_unlisted_company_by_ownership():
    orch = Orchestrator(POLICY, get_settings(), sanctions=fictional_list(), audit=DecisionLogger(None))
    r = orch.handle_sanctions_screen("Fictiva Ventures FZE", owners=[{**KHALID, "percent": 70}, {"name": "Clara Openfield", "percent": 30}])
    assert r["matches"] == []  # the company itself is on no list...
    assert r["recommendation"] == "block_and_escalate"  # ...but it is owned 70% by a listed person
    assert r["ownership"]["owners"][0]["listed_name"] == "Khalid Testani"


def test_rule_off_in_older_policy():
    old = load_policy("1.2.0")
    orch = Orchestrator(old, get_settings(), sanctions=fictional_list(old), audit=DecisionLogger(None))
    r = orch.handle_sanctions_screen("Fictiva Ventures FZE", owners=[{**KHALID, "percent": 70}])
    assert r["recommendation"] == "clear"
    assert r["ownership"]["applied"] is False
