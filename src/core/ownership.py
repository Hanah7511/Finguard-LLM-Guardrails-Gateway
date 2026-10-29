"""OFAC 50% rule: a company owned 50% or more, in total, by blocked persons is itself
blocked, even when its own name is on no list.

The caller sends the owners from its KYC records; FinGuard does not look them up. An
owner that is a company can list its own owners. A company blocked this way counts in
full in the companies it owns: shares are added up, not multiplied down a chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.config.policy_schema import OwnershipRulePolicy
from src.core.sanctions_screener import SanctionsScreener

# Rounding slack when percentages are added up (e.g. 33.33 + 33.33 + 33.34).
_TOLERANCE = 0.01


class Owner(BaseModel):
    """One owner as the caller sends it. Both doors use this shape."""
    name: str = Field(min_length=1, max_length=300, description="Owner's full name (person or company).")
    percent: float = Field(gt=0, le=100, description="Share of the company this owner holds, 0-100.")
    dob: Optional[str] = Field(default=None, max_length=20, description="Owner's date of birth, YYYY-MM-DD, if a person.")
    nationality: Optional[str] = Field(default=None, max_length=100)
    id_number: Optional[str] = Field(default=None, max_length=50)
    identifiers_verified: bool = Field(default=False, description="True only if the DOB / ID were checked against an official document.")
    owners: list["Owner"] = Field(default_factory=list, description="If this owner is a company: its own owners.")


@dataclass
class OwnerResult:
    name: str
    percent: float
    status: str  # "blocked", "possible" or "clear"
    reason: str
    listed_name: Optional[str] = None
    list_name: Optional[str] = None
    owners: list["OwnerResult"] = field(default_factory=list)


@dataclass
class OwnershipResult:
    blocked_share: float   # % held by blocked owners
    possible_share: float  # % held by blocked or possibly listed owners
    recommendation: str
    reason: str
    owners: list[OwnerResult]
    owners_screened: int


class OwnershipChecker:
    def __init__(self, screener: SanctionsScreener, policy: OwnershipRulePolicy) -> None:
        self._screener = screener
        self._policy = policy

    def _meets(self, share: float) -> bool:
        t = self._policy.threshold_percent
        return share >= t - _TOLERANCE if self._policy.inclusive else share > t + _TOLERANCE

    @staticmethod
    def _shares(results: list[OwnerResult]) -> tuple[float, float]:
        blocked = sum(r.percent for r in results if r.status == "blocked")
        possible = sum(r.percent for r in results if r.status in ("blocked", "possible"))
        return round(blocked, 2), round(possible, 2)

    def validate(self, owners: list[dict[str, Any]], depth: int = 1) -> None:
        """Impossible shares are bad data under any policy, so this runs even when the rule is off."""
        if depth > self._policy.max_depth:
            raise ValueError(f"ownership chain is deeper than {self._policy.max_depth} levels")
        total = sum(o["percent"] for o in owners)
        if total > 100 + _TOLERANCE:
            raise ValueError(f"owners add up to {total:g}%, more than 100%")
        for o in owners:
            self.validate(o.get("owners") or [], depth + 1)

    def _owner(self, o: dict[str, Any]) -> tuple[OwnerResult, int]:
        """Screen one owner (and its own owners). Returns the result and how many were screened."""
        r = self._screener.screen(
            o["name"], dob=o.get("dob"), nationality=o.get("nationality"), id_number=o.get("id_number"),
            identifiers_verified=o.get("identifiers_verified", False),
        )
        subs = [self._owner(x) for x in o.get("owners") or []]
        sub_results = [s for s, _ in subs]
        screened = 1 + sum(n for _, n in subs)
        top = next((m for m in r.matches if m.status != "ruled_out"), None)
        listed = {"listed_name": top.name, "list_name": top.list_name} if top else {}
        blocked, possible = self._shares(sub_results)

        if r.recommendation == "block_and_escalate":
            status, reason = "blocked", f"on the {top.list_name} list: {top.reason}" if top else "listed"
        elif sub_results and self._meets(blocked):
            status, reason = "blocked", f"{blocked:g}% owned by blocked persons"
        elif r.recommendation == "escalate_for_review":
            status, reason = "possible", f"possible {top.list_name} match: {top.reason}" if top else "possible match"
        elif sub_results and self._meets(possible):
            status, reason = "possible", f"{possible:g}% owned by blocked or possibly listed persons"
        else:
            status, reason = "clear", "no list match"
        return OwnerResult(o["name"], o["percent"], status, reason, owners=sub_results, **listed), screened

    def check(self, owners: list[dict[str, Any]]) -> OwnershipResult:
        """owners: Owner.model_dump() dicts. Raises ValueError if the shares or depth are impossible."""
        self.validate(owners)
        pairs = [self._owner(o) for o in owners]
        results = [r for r, _ in pairs]
        blocked, possible = self._shares(results)
        t = f"{self._policy.threshold_percent:g}%"

        if self._meets(blocked):
            recommendation = "block_and_escalate"
            reason = f"blocked owners hold {blocked:g}% in total, at or over the {t} ownership threshold"
        elif self._meets(possible):
            recommendation = "escalate_for_review"
            reason = f"owners that may be listed hold {possible:g}%; confirm them, the company could be blocked"
        elif _any_flagged(results):
            # Below the threshold the company is not blocked, but a person should see a listed owner.
            recommendation = "escalate_for_review"
            reason = f"a listed or possibly listed owner holds less than {t}; the company is not blocked by the rule"
        else:
            recommendation = "clear"
            reason = "no owner is listed"
        return OwnershipResult(blocked, possible, recommendation, reason, results, sum(n for _, n in pairs))


def _any_flagged(results: list[OwnerResult]) -> bool:
    return any(r.status != "clear" or _any_flagged(r.owners) for r in results)
