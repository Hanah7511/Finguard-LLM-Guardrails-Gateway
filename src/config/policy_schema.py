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


class GlinerPolicy(StrictModel):
    # A small model for names and addresses, with Presidio's rules still handling numbers. Off before v1.10.0.
    enabled: bool = False
    model_id: str = "urchade/gliner_multi_pii-v1"
    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    labels: dict[str, str] = Field(default_factory=dict)  # what we ask the model for -> FinGuard's entity type


class InputGuardrailsPolicy(StrictModel):
    require_consent: bool = True
    max_query_chars: int = Field(default=4000, gt=0)
    injection_patterns: PatternList = Field(min_length=1)
    safety_patterns: PatternList = Field(min_length=1)
    pii_entities: list[str] = Field(min_length=1)
    pii_score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Names the caller lists at Tool 1 become tokens and are masked in every later text. Off before v1.6.0.
    party_masking: bool = False
    custom_pii_recognizers: list[CustomPIIRecognizer] = Field(default_factory=list)
    llama_guard: LlamaGuardConfig = LlamaGuardConfig()
    gliner: GlinerPolicy = Field(default_factory=GlinerPolicy)


class ToolCallGuardrailsPolicy(StrictModel):
    allowlist: dict[str, list[str]] = Field(min_length=1)
    # tool -> parameter -> regex the value must fully match. Listed params are required.
    parameter_rules: dict[str, dict[str, str]] = Field(default_factory=dict)
    max_parameter_chars: int = Field(default=500, gt=0)
    # Agents prove who they are with their own secret key (AGENT_KEY_* in .env). Off before v1.4.0.
    require_agent_keys: bool = False
    # Tool 3 masks as per-case tokens ([IBAN_CODE_1]) that Tool 8 can restore. Off before v1.5.0.
    reversible_masking: bool = False
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



class CTRRulePolicy(StrictModel):
    # 31 CFR 1010.311: a Currency Transaction Report for each CASH transaction of MORE THAN
    # the threshold. Off before v1.4.0 so older decisions stay reproducible.
    enabled: bool = False
    threshold: float = Field(default=10000.0, gt=0)
    cash_types: list[str] = Field(default_factory=lambda: ["cash"])
    citation: str = "31 CFR 1010.311"

class RiskAssessmentPolicy(StrictModel):    
    # signal name -> weight; signal values are 0..1 (booleans count as 0/1)
    signal_weights: dict[str, float] = Field(min_length=1)
    large_amount_threshold: float = Field(default=10000.0, gt=0)
    large_amount_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    high_risk_country_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    monitored_country_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    ctr_rule: CTRRulePolicy = Field(default_factory=CTRRulePolicy)


    high_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "RiskAssessmentPolicy":
        if self.medium_threshold >= self.high_threshold:
            raise ValueError("medium_threshold must be below high_threshold")
        if any(w < 0 for w in self.signal_weights.values()):
            raise ValueError("signal weights must be non-negative")
        return self


class OwnershipRulePolicy(StrictModel):
    # OFAC 50% rule. Off before v1.3.0 so older decisions stay reproducible.
    enabled: bool = False
    threshold_percent: float = Field(default=50.0, gt=0, le=100)
    # True: "50% or more" (OFAC). False: "more than 50%" (EU).
    inclusive: bool = True
    max_depth: int = Field(default=5, ge=1, le=10)


class ArabicScriptPolicy(StrictModel):
    # Screen names written in Arabic script. Off before v1.11.0: such names matched nothing.
    enabled: bool = False
    # How close an Arabic name must be to the list's own Arabic name. Stricter than the
    # English-letter threshold: Arabic leaves out short vowels, so one letter counts for more.
    match_threshold: float = Field(default=0.9, ge=0.5, le=1.0)
    # How close two consonant keys must be (Arabic name vs English-letter name). 1.0 = the same key.
    key_threshold: float = Field(default=1.0, ge=0.5, le=1.0)


class SanctionsScreeningPolicy(StrictModel):
    lists: list[str] = Field(min_length=1)
    fuzzy_match_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    high_risk_jurisdictions: list[str] = Field(default_factory=list)
    # FATF grey list: raises country risk to ELEVATED without forcing a review.
    increased_monitoring_jurisdictions: list[str] = Field(default_factory=list)
    # Use date of birth / nationality / ID number to confirm or rule out a name match.
    # Off in v1.0.0 and v1.1.0 so their decisions stay reproducible; on from v1.2.0.
    use_identifiers: bool = False
    # An exact name match is cleared by a different DOB / ID only if they were verified (KYC).
    clear_requires_verified_identifiers: bool = False
    # Tool 4 takes a token such as [PERSON_1] and never shows the agent the name. Off before v1.6.0.
    accept_tokens: bool = False
    ownership_rule: OwnershipRulePolicy = Field(default_factory=OwnershipRulePolicy)
    arabic_script: ArabicScriptPolicy = Field(default_factory=ArabicScriptPolicy)


class FinalizeCheckPolicy(StrictModel):
    # Tool 8, the last gate before human review. Off before v1.5.0.
    enabled: bool = False
    # True before v1.8.0: Tool 8 hands back real values. False: only an analyst can get them.
    return_real_values: bool = True
    # Scan the finished report once more and mask anything that slipped through. Off before v1.6.0.
    recheck_report: bool = False
    allowed_agents: list[str] = Field(default_factory=list)
    required_steps: list[Literal["risk_assessed", "claims_grounded", "no_sanctions_block", "no_refused_calls"]] = Field(
        default_factory=lambda: ["risk_assessed", "claims_grounded", "no_sanctions_block", "no_refused_calls"]
    )


class AnalystReleasePolicy(StrictModel):
    # POST /cases/{case_id}/release: real values go to a named analyst only. Off before v1.8.0.
    enabled: bool = False
    allowed_analysts: list[str] = Field(default_factory=list)  # each has its own ANALYST_KEY_<NAME>


class AccessControlPolicy(StrictModel):
    # Who may call FinGuard, and what a case id may look like. All off before v1.7.0.
    require_app_key: bool = False  # the REST door needs X-API-Key on everything except /health
    require_agent_key_on_case_tools: bool = False  # Tools 1, 3, 4, 5 need an agent key, as Tools 2 and 8 do
    require_case_id: bool = False  # tools that change a case refuse a call without one
    case_id_pattern: str = "^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$"

    @field_validator("case_id_pattern")
    @classmethod
    def _pattern_compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid case_id_pattern: {exc}") from exc
        return value


class GuardrailPolicy(StrictModel):
    policy_metadata: PolicyMetadata
    input_guardrails: InputGuardrailsPolicy
    tool_call_guardrails: ToolCallGuardrailsPolicy
    output_guardrails: OutputGuardrailsPolicy
    risk_assessment: RiskAssessmentPolicy
    sanctions_screening: SanctionsScreeningPolicy
    finalize_check: FinalizeCheckPolicy = Field(default_factory=FinalizeCheckPolicy)
    access_control: AccessControlPolicy = Field(default_factory=AccessControlPolicy)
    analyst_release: AnalystReleasePolicy = Field(default_factory=AnalystReleasePolicy)

    @model_validator(mode="after")
    def _finalize_needs_keys_and_tokens(self) -> "GuardrailPolicy":
        cfg, tools = self.finalize_check, self.tool_call_guardrails
        if cfg.enabled:
            # Tool 8 hands back real values, so it must know who is calling and have tokens to restore.
            if not (tools.require_agent_keys and tools.reversible_masking):
                raise ValueError("finalize_check needs require_agent_keys and reversible_masking")
            unknown = set(cfg.allowed_agents) - set(tools.allowlist)
            if unknown or not cfg.allowed_agents or not cfg.required_steps:
                raise ValueError(f"finalize_check needs known allowed_agents and required_steps; unknown: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _release_rules(self) -> "GuardrailPolicy":
        release, final = self.analyst_release, self.finalize_check
        if release.enabled and not (final.enabled and release.allowed_analysts):
            raise ValueError("analyst_release needs finalize_check and at least one allowed analyst")
        if not final.return_real_values and not (release.enabled and final.recheck_report):
            raise ValueError("return_real_values: false needs analyst_release and recheck_report")
        return self

    @model_validator(mode="after")
    def _case_tools_need_agent_keys(self) -> "GuardrailPolicy":
        if self.access_control.require_agent_key_on_case_tools and not self.tool_call_guardrails.require_agent_keys:
            raise ValueError("require_agent_key_on_case_tools needs require_agent_keys")
        return self

    @model_validator(mode="after")
    def _names_need_tokens(self) -> "GuardrailPolicy":
        uses_tokens = self.input_guardrails.party_masking or self.sanctions_screening.accept_tokens
        if uses_tokens and not self.tool_call_guardrails.reversible_masking:
            raise ValueError("party_masking and accept_tokens need reversible_masking")
        return self

    @property
    def version(self) -> str:
        return self.policy_metadata.version
