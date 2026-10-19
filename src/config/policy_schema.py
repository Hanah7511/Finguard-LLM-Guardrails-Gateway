"""Validates the guardrail policy YAML itself.

Every model forbids unknown keys, so a typo such as ``minimum_confidance`` fails
at startup instead of silently falling back to a default.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _clean_patterns(patterns: list[str]) -> list[str]:
    cleaned = [p.strip() for p in patterns]
    if any(not p for p in cleaned):
        raise ValueError("patterns must not be empty strings")
    return cleaned


PatternList = Annotated[list[str], AfterValidator(_clean_patterns)]


class PolicyMetadata(StrictModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    effective_date: date
    approved_by: str = Field(min_length=1)
    change_reason: str = Field(min_length=1)
    supersedes: Optional[str] = None


class CustomPIIRecognizer(StrictModel):
    entity_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    regex: str
    score: float = Field(ge=0.0, le=1.0)
    context: list[str] = Field(default_factory=list)

    @field_validator("regex")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return value


class LlamaGuardConfig(StrictModel):
    enabled: bool = False
    model_id: str = "meta-llama/Llama-Guard-3-1B"
    # What to do if the model cannot load or errors: fail closed by default.
    on_error: Literal["block", "allow"] = "block"


class InputGuardrailsPolicy(StrictModel):
    require_consent: bool = True
    max_query_chars: int = Field(default=4000, gt=0)
    injection_patterns: PatternList = Field(min_length=1)
    safety_patterns: PatternList = Field(min_length=1)
    pii_entities: list[str] = Field(min_length=1)
    pii_score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    custom_pii_recognizers: list[CustomPIIRecognizer] = Field(default_factory=list)
    llama_guard: LlamaGuardConfig = LlamaGuardConfig()


class ToolCallGuardrailsPolicy(StrictModel):
    allowlist: dict[str, list[str]] = Field(min_length=1)
    # tool -> parameter -> regex the value must fully match. Listed params are required.
    parameter_rules: dict[str, dict[str, str]] = Field(default_factory=dict)
    max_parameter_chars: int = Field(default=500, gt=0)
    max_result_chars: int = Field(default=20000, gt=0)
    indirect_injection_patterns: PatternList = Field(default_factory=list)

    @field_validator("parameter_rules")
    @classmethod
    def _rules_compile(cls, rules: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        for tool, params in rules.items():
            for name, regex in params.items():
                try:
                    re.compile(regex)
                except re.error as exc:
                    raise ValueError(f"parameter_rules.{tool}.{name}: invalid regex: {exc}") from exc
        return rules

    @model_validator(mode="after")
    def _rules_reference_known_tools(self) -> "ToolCallGuardrailsPolicy":
        known = {tool for tools in self.allowlist.values() for tool in tools}
        unknown = set(self.parameter_rules) - known
        if unknown:
            raise ValueError(f"parameter_rules reference tools not in any allowlist: {sorted(unknown)}")
        return self


class OutputGuardrailsPolicy(StrictModel):
    grounding_method: Literal["nli", "cosine"] = "nli"
    minimum_confidence: float = Field(ge=0.0, le=1.0)
    entailment_threshold: float = Field(ge=0.0, le=1.0)
    contradiction_threshold: float = Field(ge=0.0, le=1.0)
    cosine_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)


class RiskAssessmentPolicy(StrictModel):
    # signal name -> weight; signal values are 0..1 (booleans count as 0/1)
    signal_weights: dict[str, float] = Field(min_length=1)
    large_amount_threshold: float = Field(default=10000.0, gt=0)
    large_amount_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    high_risk_country_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    high_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "RiskAssessmentPolicy":
        if self.medium_threshold >= self.high_threshold:
            raise ValueError("medium_threshold must be below high_threshold")
        if any(w < 0 for w in self.signal_weights.values()):
            raise ValueError("signal weights must be non-negative")
        return self


class SanctionsScreeningPolicy(StrictModel):
    lists: list[str] = Field(min_length=1)
    fuzzy_match_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    high_risk_jurisdictions: list[str] = Field(default_factory=list)


class GuardrailPolicy(StrictModel):
    policy_metadata: PolicyMetadata
    input_guardrails: InputGuardrailsPolicy
    tool_call_guardrails: ToolCallGuardrailsPolicy
    output_guardrails: OutputGuardrailsPolicy
    risk_assessment: RiskAssessmentPolicy
    sanctions_screening: SanctionsScreeningPolicy

    @property
    def version(self) -> str:
        return self.policy_metadata.version
