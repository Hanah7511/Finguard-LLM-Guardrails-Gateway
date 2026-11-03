"""Reversible masking: real values are swapped for per-case tokens such as [IBAN_CODE_1].

AI agents only ever see the tokens. Within one case the same value always gets the same
token, so an agent can still tell that two transactions used one account. Values are never
logged. PIIVault keeps them in memory, so a restart loses them; src/storage/postgres.py is the
encrypted, lasting version.
"""
from __future__ import annotations

import re
import threading
from typing import Callable, Iterable


def same_value_key(value: str) -> str:
    """'GB82 WEST 1234' and 'gb82west1234' are one account."""
    return "".join(value.split()).upper()


class BaseVault:
    def __init__(self, entity_types: Iterable[str]) -> None:
        # Only known types count as tokens, so ordinary text like [APPENDIX_1] is left alone.
        names = "|".join(re.escape(t) for t in sorted(set(entity_types), key=len, reverse=True))
        self._token = re.compile(rf"\[(?:{names})_\d+\]")

    def token_for(self, case_id: str, entity_type: str, value: str) -> str:
        raise NotImplementedError

    def _values(self, case_id: str, tokens: set[str]) -> dict[str, str]:
        """The real values for these tokens. A token this case never issued is simply absent."""
        raise NotImplementedError

    def tokenizer(self, case_id: str) -> Callable[[str, str], str]:
        return lambda entity_type, value: self.token_for(case_id, entity_type, value)

    def restore(self, case_id: str, text: str) -> tuple[str, int, list[str]]:
        """Put the real values back. Returns (text, tokens restored, tokens this case never issued).

        One pass only, so a real value that happens to look like a token is not replaced again.
        """
        values = self._values(case_id, set(self._token.findall(text)))
        restored, unknown = 0, set()

        def swap(match: re.Match) -> str:
            nonlocal restored
            value = values.get(match.group())
            if value is None:
                unknown.add(match.group())
                return match.group()
            restored += 1
            return value

        return self._token.sub(swap, text), restored, sorted(unknown)


class PIIVault(BaseVault):
    def __init__(self, entity_types: Iterable[str]) -> None:
        super().__init__(entity_types)
        self._tokens: dict[str, dict[tuple[str, str], str]] = {}  # case -> (type, value key) -> token
        self._stored: dict[str, dict[str, str]] = {}  # case -> token -> real value
        self._lock = threading.Lock()

    def token_for(self, case_id: str, entity_type: str, value: str) -> str:
        key = (entity_type, same_value_key(value))
        with self._lock:
            tokens = self._tokens.setdefault(case_id, {})
            if key not in tokens:
                count = sum(1 for t, _ in tokens if t == entity_type)
                tokens[key] = f"[{entity_type}_{count + 1}]"
                self._stored.setdefault(case_id, {})[tokens[key]] = value
            return tokens[key]

    def _values(self, case_id: str, tokens: set[str]) -> dict[str, str]:
        with self._lock:
            stored = self._stored.get(case_id, {})
            return {token: stored[token] for token in tokens if token in stored}
