"""Normalised phrase matching shared by the injection, safety and indirect-injection checks.

Normalisation defeats cheap evasions: case, Unicode look-alikes (NFKC), zero-width
characters, and punctuation/whitespace between words ("ignore-previous_instructions").
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).lower()
    return " ".join(_NON_WORD.sub(" ", text).split())


class PatternMatcher:
    """Matches phrases anchored at a word start, so "hide transaction" also matches
    "hide transactions" but "aml" does not match "family"."""

    def __init__(self, patterns: Iterable[str]) -> None:
        self._patterns = [(p, " " + normalize(p)) for p in patterns]

    def find_all(self, text: str) -> list[str]:
        haystack = " " + normalize(text)
        return [original for original, needle in self._patterns if needle in haystack]
