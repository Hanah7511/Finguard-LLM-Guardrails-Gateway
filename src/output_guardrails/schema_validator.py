"""Validates the shape of the LLM response with Pydantic."""
from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class ComplianceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    explanation: str = Field(min_length=10)
    cited_regulation: str = Field(min_length=3)
    confidence_score: float = Field(ge=0.0, le=1.0)

    @field_validator("risk_level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


class SchemaValidator:
    layer = "output.schema_validation"

    def __init__(self, policy_version: str) -> None:
        self._policy_version = policy_version

    def validate(self, raw: Union[str, dict]) -> tuple[GuardrailDecision, Optional[ComplianceResponse]]:
        sw = Stopwatch()
        try:
            data = raw if isinstance(raw, dict) else json.loads(_strip_fence(raw))
            parsed = ComplianceResponse.model_validate(data)
        except json.JSONDecodeError as exc:
            return self._reject(f"not_valid_json: {exc.msg}", sw), None
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors())
            return self._reject(f"schema_errors: {errors}", sw), None
        decision = GuardrailDecision(
            layer=self.layer,
            decision=Decision.PASS,
            reasoning_trace="all_required_fields_valid",
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
        )
        return decision, parsed

    def _reject(self, trace: str, sw: Stopwatch) -> GuardrailDecision:
        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.REJECT,
            reasoning_trace=trace,
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
        )


def _strip_fence(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text
