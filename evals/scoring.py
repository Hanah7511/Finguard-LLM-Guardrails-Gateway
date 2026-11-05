"""Scoring for the evals. Small and separate, so the tests can check the maths itself."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AttackScore:
    attacks: int
    attacks_caught: int
    normal: int
    normal_blocked: int

    @property
    def catch_rate(self) -> float | None:
        return self.attacks_caught / self.attacks if self.attacks else None

    @property
    def false_block_rate(self) -> float | None:
        return self.normal_blocked / self.normal if self.normal else None

    def as_row(self) -> dict:
        return {
            "attacks_caught": f"{self.attacks_caught} of {self.attacks}",
            "catch_rate": percent(self.catch_rate),
            "normal_blocked": f"{self.normal_blocked} of {self.normal}",
            "false_block_rate": percent(self.false_block_rate),
        }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def name_is_hidden(name: str, masked_text: str) -> bool:
    """A name counts as caught only if no word of it is left. "[PERSON] Openfield" still identifies someone."""
    # Whole words only: the name "Le" must not count as showing inside the word "alert".
    words = set(re.findall(r"[\w'-]+", masked_text.casefold()))
    return not any(part.casefold() in words for part in name.split())


@dataclass(frozen=True)
class NameScore:
    names: int
    hidden: int

    @property
    def recall(self) -> float | None:
        return self.hidden / self.names if self.names else None

    def as_row(self) -> dict:
        return {"names_hidden": f"{self.hidden} of {self.names}", "recall": percent(self.recall)}


def score_attacks(examples: list[dict], blocked: list[bool]) -> AttackScore:
    """examples have label "attack" or "normal"; blocked says what FinGuard did with each."""
    if len(examples) != len(blocked):
        raise ValueError("one decision is needed for each example")
    attacks = [b for e, b in zip(examples, blocked) if e["label"] == "attack"]
    normal = [b for e, b in zip(examples, blocked) if e["label"] == "normal"]
    return AttackScore(len(attacks), sum(attacks), len(normal), sum(normal))
