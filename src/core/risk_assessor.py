"""Scores an incoming alert from its risk signals, using weights from the policy.

If another system (e.g. a detection engine) already produced a risk score, pass it
as external_score. The final score is the higher of the two, so FinGuard can raise
the caution level but never lower it. It also says whether a Currency Transaction Report is required (cash over the policy
threshold). That is a reporting duty, not a sign of suspicion, so it does not change the score.
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
    ctr: dict[str, Any] = field(default_factory=dict)


def _as_unit(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


class RiskAssessor:
    layer = "risk.assessment"

    def __init__(self, policy: RiskAssessmentPolicy, high_risk_jurisdictions: list[str], policy_version: str, monitored_jurisdictions: tuple[str, ...] = (),) -> None:
        self._policy = policy
        self._jurisdictions = {j.strip().lower() for j in high_risk_jurisdictions}
        self._monitored = {j.strip().lower() for j in monitored_jurisdictions}
        self._policy_version = policy_version

    def assess(
            self, context: dict[str, Any], external_score: Optional[float] = None, transaction_type: Optional[str] = None
        ) -> RiskAssessment:
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
        place = country.strip().lower() if isinstance(country, str) else ""
        if place in self._jurisdictions:
            contributions["high_risk_country"] = self._policy.high_risk_country_weight
        elif place in self._monitored and self._policy.monitored_country_weight > 0:
            contributions["monitored_country"] = self._policy.monitored_country_weight

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
        ctr = self._ctr(context.get("transaction_amount"), transaction_type)

        decision = GuardrailDecision(
            layer=self.layer,
            decision=Decision.PASS,
            reasoning_trace=(
                f"score={score:.2f} (own={own:.2f}, external={ext}) level={level} "
                f"signals={sorted(contributions)} ctr_required={ctr['required']}"
            ),
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata={"contributions": contributions, "ignored_signals": ignored, "scores_disagree": disagree},
        )
        return RiskAssessment(
            score, level, RECOMMENDATIONS[level], own, ext, disagree, contributions, ignored, decision, ctr
        )

    
    def _ctr(self, amount: Any, transaction_type: Optional[str]) -> dict[str, Any]:
        rule = self._policy.ctr_rule
        if not rule.enabled:
            return {"applied": False, "required": False, "reason": "the CTR rule is not in this policy version"}
        out = {"applied": True, "required": False, "citation": rule.citation}
        kind = transaction_type.strip().lower() if isinstance(transaction_type, str) else ""
        limit = f"{rule.threshold:,.0f}"
        if not kind:
            out["reason"] = "transaction_type not given; a CTR applies only to cash"
        elif kind not in {t.lower() for t in rule.cash_types}:
            out["reason"] = f"a {kind} transaction is not cash; a CTR applies only to cash"
        elif not isinstance(amount, (int, float)) or isinstance(amount, bool):
            out["reason"] = "cash transaction without a transaction_amount to compare"
        elif amount > rule.threshold:  # the rule says "more than", so exactly the threshold is not enough
            out.update(required=True, reason=f"cash transaction of {amount:,.0f} is more than {limit}")
        else:
            # 31 CFR 1010.313 adds up one person's cash on the same business day; FinGuard sees one amount.
            out["reason"] = (f"cash transaction of {amount:,.0f} is not more than {limit}; "
                             "same-day cash totals are not added up here")
        return out


    def _is_large(self, amount: Any) -> bool:
        return (
            isinstance(amount, (int, float))
            and not isinstance(amount, bool)
            and amount >= self._policy.large_amount_threshold
        )
