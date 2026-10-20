"""Scores an incoming alert from its risk signals, using weights from the policy.

If another system (e.g. a detection engine) already produced a risk score, pass it
as external_score. The final score is the higher of the two, so FinGuard can raise
the caution level but never lower it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.config.policy_schema import RiskAssessmentPolicy
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch

RECOMMENDATIONS = {"HIGH": "proceed_with_caution", "MEDIUM": "standard_review", "LOW": "proceed"}
_SPECIAL_KEYS = {"transaction_amount", "country"}
_DISAGREEMENT_GAP = 0.30


@dataclass
class RiskAssessment:
    risk_score: float
    risk_level: str
    recommendation: str
    own_score: float
    external_score: Optional[float] = None
    scores_disagree: bool = False
    contributing_signals: dict[str, float] = field(default_factory=dict)
    ignored_signals: list[str] = field(default_factory=list)
    decision: Optional[GuardrailDecision] = None


def _as_unit(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


class RiskAssessor:
    layer = "risk.assessment"

    def __init__(self, policy: RiskAssessmentPolicy, high_risk_jurisdictions: list[str], policy_version: str) -> None:
        self._policy = policy
        self._jurisdictions = {j.strip().lower() for j in high_risk_jurisdictions}
        self._policy_version = policy_version

    def assess(self, context: dict[str, Any], external_score: Optional[float] = None) -> RiskAssessment:
        sw = Stopwatch()
        weights = self._policy.signal_weights
        contributions: dict[str, float] = {}
        ignored: list[str] = []

        for name, raw in context.items():
            if name in _SPECIAL_KEYS:
                continue
            value = _as_unit(raw)
            if name not in weights or value is None:
                ignored.append(name)
            elif value > 0:
                contributions[name] = round(weights[name] * value, 4)

        if self._is_large(context.get("transaction_amount")):
            contributions["large_amount"] = self._policy.large_amount_weight
        country = context.get("country")
        if isinstance(country, str) and country.strip().lower() in self._jurisdictions:
            contributions["high_risk_country"] = self._policy.high_risk_country_weight

        own = round(min(1.0, sum(contributions.values())), 2)
        ext = round(max(0.0, min(1.0, external_score)), 2) if external_score is not None else None
        score = max(own, ext) if ext is not None else own
        disagree = ext is not None and abs(own - ext) >= _DISAGREEMENT_GAP

        if score >= self._policy.high_threshold:
            level = "HIGH"
        elif score >= self._policy.medium_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        decision = GuardrailDecision(
            layer=self.layer,
            decision=Decision.PASS,
            reasoning_trace=(
                f"score={score:.2f} (own={own:.2f}, external={ext}) level={level} "
                f"signals={sorted(contributions)}"
            ),
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata={"contributions": contributions, "ignored_signals": ignored, "scores_disagree": disagree},
        )
        return RiskAssessment(
            score, level, RECOMMENDATIONS[level], own, ext, disagree, contributions, ignored, decision
        )

    def _is_large(self, amount: Any) -> bool:
        return (
            isinstance(amount, (int, float))
            and not isinstance(amount, bool)
            and amount >= self._policy.large_amount_threshold
        )
