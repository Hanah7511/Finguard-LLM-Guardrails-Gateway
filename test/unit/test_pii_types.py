"""Tests for the personal data added in policy v1.9.0, with the real Presidio model. All values are made up.

Run:  python -m pytest test/unit/test_pii_types.py -v
"""
from __future__ import annotations

import pytest

from src.config.policy_loader import load_policy
from src.input_guardrails.pii.presidio_detector import build_pii_detector

POLICY = load_policy("1.9.0")


@pytest.fixture(scope="module")
def detector():
    return build_pii_detector(POLICY.input_guardrails, "presidio")


@pytest.mark.parametrize("text, expected", [
    ("Passport number N1234567 was presented.", "Passport number [PASSPORT] was presented."),
    ("Emirates ID 784-1990-1234567-1 is on file.", "Emirates ID [EMIRATES_ID] is on file."),
    ("The form lists 784199076543217 for her.", "The form lists [EMIRATES_ID] for her."),
    ("Call +971 50 123 4567 today.", "Call [PHONE] today."),
    ("Call 050-765-4321 today.", "Call [PHONE] today."),
    ("Date of birth is 12 March 1980 on the form.", "Date of birth is [DATE_OF_BIRTH] on the form."),
    ("She was born on 1975-05-05 in Dubai.", "She was born on [DATE_OF_BIRTH] in Dubai."),
])
def test_new_types_are_masked(detector, text, expected):
    assert detector.mask(text).masked_text == expected


@pytest.mark.parametrize("text", [
    "The alert was raised on 2026-11-05.",
    "Review the case by 12 March 2027.",
    "Product code AB123456 was on the invoice.",
    "Reference TXN-2026001 appears on the wire.",
    "The law cited is 31 CFR 1010.311.",
])
def test_ordinary_codes_and_dates_stay_readable(detector, text):
    assert detector.mask(text).masked_text == text


def test_older_policy_does_not_mask_the_new_types():
    older = build_pii_detector(load_policy("1.8.0").input_guardrails, "presidio")
    assert "784-1990-1234567-1" in older.mask("Emirates ID 784-1990-1234567-1 is on file.").masked_text
