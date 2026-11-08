"""The exact check on figures in a claim (policy v1.12.0). No model needed.

Run: python -m pytest test/unit/test_fact_check.py -v
"""
from __future__ import annotations

import pytest

from src.config.policy_loader import load_policy
from src.monitoring.telemetry import Decision
from src.output_guardrails.grounding.base import Grounder, GroundingResult, GroundingVerdict
from src.output_guardrails.grounding.fact_check import unsupported_facts
from src.output_guardrails.pipeline import OutputGuardrailPipeline

LAW = (
    "A report is required for a transaction in currency of more than $10,000. A bank shall file within "
    "30 calendar days and keep a copy for five years from the date of filing. The penalty is not more than 5 years."
)


@pytest.mark.parametrize("claim", [
    "Transactions of more than $10,000 are reported.",
    "A transaction above $10,000 is reported.",            # another way to say "more than"
    "The $10,000 limit applies.",                          # no comparison claimed
    "File within 30 calendar days.",
    "File within thirty calendar days.",                   # number as a word
    "File within 30 days.",                                # less precise, not wrong
    "Keep it for 5 years from the date of filing.",        # 5 and five
    "The penalty is up to 5 years.",
    "A bank must keep good records.",                      # nothing to check
])
def test_claims_with_correct_figures_pass(claim):
    assert unsupported_facts(LAW, claim) == ()


@pytest.mark.parametrize("claim, kind", [
    ("Transactions of $10,000 or more are reported.", "money"),
    ("Transactions of at least $10,000 are reported.", "money"),
    ("Transactions of more than $50,000 are reported.", "money"),
    ("File within 90 calendar days.", "period"),
    ("File within 30 business days.", "period"),
    ("Keep it for ten years.", "period"),
    ("Keep it for 5 months.", "period"),
    ("Keep it for five years from the date of the transaction.", "date_anchor"),
])
def test_claims_with_changed_figures_fail(claim, kind):
    assert unsupported_facts(LAW, claim) == (kind,)


def test_not_more_than_is_not_read_as_more_than():
    assert unsupported_facts("A fine of not more than $500.", "A fine of up to $500.") == ()
    assert unsupported_facts("A fine of not more than $500.", "A fine of more than $500.") == ("money",)


def test_only_the_kind_is_returned_never_the_value():
    assert "50,000" not in " ".join(unsupported_facts(LAW, "More than $50,000."))


# In the pipeline

class SaysGrounded(Grounder):
    method = "stub"

    def __init__(self, verdict=GroundingVerdict.GROUNDED):
        self.verdict = verdict

    def score(self, source, claim):
        return GroundingResult(self.verdict, 0.97, 0.01, 0.02, self.method)


def pipeline(version, verdict=GroundingVerdict.GROUNDED):
    return OutputGuardrailPipeline(load_policy(version), SaysGrounded(verdict), None, None)


def test_an_accepted_claim_with_a_wrong_figure_goes_to_a_person():
    decision, result = pipeline("1.12.0").check_grounding(LAW, "Transactions of $10,000 or more are reported.")
    assert result.verdict is GroundingVerdict.UNGROUNDED and result.failed_fact_checks == ("money",)
    assert decision.decision is Decision.ESCALATE
    assert "failed_fact_checks=money" in decision.reasoning_trace


def test_an_accepted_claim_with_right_figures_stays_accepted():
    _, result = pipeline("1.12.0").check_grounding(LAW, "Transactions of more than $10,000 are reported.")
    assert result.verdict is GroundingVerdict.GROUNDED


def test_a_contradicted_claim_stays_blocked():
    _, result = pipeline("1.12.0", GroundingVerdict.HALLUCINATED).check_grounding(LAW, "More than $50,000.")
    assert result.verdict is GroundingVerdict.HALLUCINATED


def test_the_older_policy_does_not_check():
    _, result = pipeline("1.11.0").check_grounding(LAW, "Transactions of $10,000 or more are reported.")
    assert result.verdict is GroundingVerdict.GROUNDED
