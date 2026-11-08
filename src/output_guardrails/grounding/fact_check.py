"""Exact checks on the figures in a claim.

The NLI model reads meaning, so it can miss small changes: "$10,000 or more" looks the same
to it as "more than $10,000". The law treats those differently (exactly $10,000 needs no report).
This file checks such facts letter by letter: every one in the claim must also be in the source.

It only says "this figure is not in the source". It cannot tell whether a figure that is there
is about the right thing, so it adds to the model and does not replace it.
"""
from __future__ import annotations

import re

# "one" is left out on purpose: it is usually an ordinary word ("one of", "on one day").
_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "twelve": 12, "thirty": 30, "sixty": 60, "ninety": 90,
}
_NUMBER_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b")

# Longest first, so "not more than" is never read as "more than".
_BEFORE = {
    "not more than": "<=", "no more than": "<=", "not exceeding": "<=", "up to": "<=", "at most": "<=",
    "not less than": ">=", "no less than": ">=", "at least": ">=",
    "in excess of": ">", "more than": ">", "greater than": ">", "exceeding": ">", "exceeds": ">",
    "exceed": ">", "above": ">", "over": ">",
    "less than": "<", "below": "<", "under": "<",
}
_AFTER = {"or more": ">=", "or less": "<="}

_MONEY = re.compile(
    r"(?:\b(?P<before>" + "|".join(sorted(_BEFORE, key=len, reverse=True)) + r")\s+(?:the\s+)?)?"
    r"\$\s?(?P<amount>\d[\d,]*)(?:\s+(?P<after>" + "|".join(_AFTER) + r")\b)?"
)
_PERIOD = re.compile(r"\b(\d+)[\s-]*(calendar day|business day|banking day|day|month|year|percent)s?\b")
_DATE_ANCHOR = re.compile(r"\bfrom the date of (?:the |its |first |initial )?([a-z]+)")


def _clean(text: str) -> str:
    text = " ".join(text.lower().split())
    return _NUMBER_WORD.sub(lambda m: str(_NUMBER_WORDS[m[1]]), text)


def _money(text: str) -> set[tuple[int, str]]:
    """Each amount with its comparison: "more than $10,000" -> (10000, ">"). "" means none is given."""
    return {
        (int(m["amount"].replace(",", "")), _BEFORE.get(m["before"] or "") or _AFTER.get(m["after"] or "") or "")
        for m in _MONEY.finditer(text)
    }


def _periods(text: str) -> set[tuple[int, str]]:
    return {(int(number), unit) for number, unit in _PERIOD.findall(text)}


def _money_supported(fact: tuple[int, str], source: set[tuple[int, str]]) -> bool:
    amount, comparison = fact
    # A claim with no comparison ("the $10,000 threshold") only needs the amount.
    # A source with no comparison says nothing against the claim's.
    return any(a == amount and (not comparison or not c or c == comparison) for a, c in source)


def _period_supported(fact: tuple[int, str], source: set[tuple[int, str]]) -> bool:
    number, unit = fact
    # "30 days" is less precise than "30 calendar days" but not wrong. "30 business days" is wrong.
    return any(n == number and (u == unit or (unit == "day" and u.endswith(" day"))) for n, u in source)


def unsupported_facts(source: str, claim: str) -> tuple[str, ...]:
    """Which kinds of fact in the claim are not in the source: "money", "period", "date_anchor".
    Only the kind is returned, never the value, because these results can reach the audit log."""
    source_text, claim_text = _clean(source), _clean(claim)
    failed = []
    source_money = _money(source_text)
    if not all(_money_supported(f, source_money) for f in _money(claim_text)):
        failed.append("money")
    source_periods = _periods(source_text)
    if not all(_period_supported(f, source_periods) for f in _periods(claim_text)):
        failed.append("period")
    if not set(_DATE_ANCHOR.findall(claim_text)) <= set(_DATE_ANCHOR.findall(source_text)):
        failed.append("date_anchor")
    return tuple(failed)
