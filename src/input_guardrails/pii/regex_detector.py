"""Dependency-free PII detector with checksum validation.

Used when Presidio is not installed, and by the audit logger as a last-line scrubber.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from src.config.policy_schema import CustomPIIRecognizer
from src.input_guardrails.pii.base import PIIDetector, PIIEntity

_CONTEXT_WINDOW = 40


def luhn_valid(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    iban = candidate.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


@dataclass(frozen=True)
class _Rule:
    entity_type: str
    regex: re.Pattern
    score: float
    context: tuple[str, ...] = ()
    validator: Optional[Callable[[str], bool]] = None


BUILTIN_RULES: dict[str, _Rule] = {
    "US_SSN": _Rule(
        "US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b"), 0.85
    ),
    "CREDIT_CARD": _Rule(
        "CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), 0.9, validator=luhn_valid
    ),
    "IBAN_CODE": _Rule(
        "IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"), 0.9, validator=iban_valid
    ),
    "US_BANK_NUMBER": _Rule(
        "US_BANK_NUMBER",
        re.compile(r"\b\d{8,17}\b"),
        0.6,
        context=("account", "acct", "bank", "routing", "checking", "savings"),
    ),
}


class RegexPIIDetector(PIIDetector):
    name = "regex"

    def __init__(
        self,
        entities: Iterable[str],
        custom_recognizers: Iterable[CustomPIIRecognizer] = (),
    ) -> None:
        self._rules: list[_Rule] = [BUILTIN_RULES[e] for e in entities if e in BUILTIN_RULES]
        for rec in custom_recognizers:
            self._rules.append(
                _Rule(rec.entity_type, re.compile(rec.regex), rec.score, tuple(c.lower() for c in rec.context))
            )

    def detect(self, text: str) -> list[PIIEntity]:
        lowered = text.lower()
        found: list[PIIEntity] = []
        for rule in self._rules:
            for m in rule.regex.finditer(text):
                if rule.validator and not rule.validator(m.group()):
                    continue
                if rule.context:
                    window = lowered[max(0, m.start() - _CONTEXT_WINDOW) : m.end() + _CONTEXT_WINDOW]
                    if not any(word in window for word in rule.context):
                        continue
                found.append(PIIEntity(rule.entity_type, m.start(), m.end(), rule.score))
        return found


def build_log_scrubber() -> Callable[[str], str]:
    """Masks SSNs, cards and IBANs anywhere in a log string, plus long digit runs."""
    detector = RegexPIIDetector(["US_SSN", "CREDIT_CARD", "IBAN_CODE"])
    long_digits = re.compile(r"\b\d{9,19}\b")

    def scrub(value: str) -> str:
        return long_digits.sub("[REDACTED_NUMBER]", detector.mask(value).masked_text)

    return scrub
