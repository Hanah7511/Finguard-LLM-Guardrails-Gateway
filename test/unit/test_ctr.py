"""Tests for the Currency Transaction Report rule (policy v1.4.0): cash of MORE THAN
10,000 only (31 CFR 1010.311). A large wire is not a CTR case.

Run:  python -m pytest test -v
"""
from __future__ import annotations

from src.config.policy_loader import load_policy
from src.core.risk_assessor import RiskAssessor

POLICY = load_policy("1.4.0")


def assessor(policy=POLICY) -> RiskAssessor:
    return RiskAssessor(policy.risk_assessment, [], policy.version)


def ctr(amount, transaction_type, policy=POLICY) -> dict:
    return assessor(policy).assess({"transaction_amount": amount}, transaction_type=transaction_type).ctr


def test_cash_over_10000_needs_a_ctr():
    r = ctr(12000, "cash")
    assert r["required"] is True
    assert r["citation"] == "31 CFR 1010.311"


def test_exactly_10000_is_not_more_than_10000():
    assert ctr(10000, "cash")["required"] is False


def test_cash_just_under_is_not_a_ctr_but_says_totals_are_not_added():
    r = ctr(9500, "Cash")  # the type is compared case-insensitively
    assert r["required"] is False
    assert "same-day" in r["reason"]


def test_large_wire_is_not_a_ctr():
    r = ctr(95000, "wire")
    assert r["required"] is False
    assert "not cash" in r["reason"]


def test_missing_type_is_said_plainly():
    r = ctr(50000, None)
    assert r["required"] is False
    assert "transaction_type not given" in r["reason"]


def test_ctr_does_not_change_the_risk_score():
    signals = {"structuring_flag": 1, "transaction_amount": 50000}
    cash = assessor().assess(signals, transaction_type="cash")
    wire = assessor().assess(signals, transaction_type="wire")
    assert cash.ctr["required"] and not wire.ctr["required"]
    assert cash.risk_score == wire.risk_score


def test_rule_off_in_older_policy():
    r = ctr(545000, "cash", load_policy("1.3.0"))
    assert (r["applied"], r["required"]) == (False, False)
