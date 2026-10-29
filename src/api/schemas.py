"""Request shapes for the FastAPI door. A request that does not fit is rejected (422)
before FinGuard sees it. Rule values (lengths, thresholds) stay in the YAML policy.

Unknown fields are ignored, so an upstream engine can post a whole alert as it is.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.core.ownership import Owner


class CaseRequest(BaseModel):
    # Links every audit record for one case across projects.
    case_id: Optional[str] = Field(default=None, max_length=100, examples=["CASE-2026-006"])


class ValidateRequest(CaseRequest):
    text: str = Field(min_length=1, examples=["Customer John Smith, email john@example.com, asks about CTR rules"])
    consent: Optional[bool] = None
    enforce_consent: bool = False


class ChatRequest(CaseRequest):
    query: str = Field(min_length=1, examples=["When must a bank file a currency transaction report?"])
    consent: Optional[bool] = None
    customer_id: Optional[str] = Field(default=None, max_length=100)


class AssessRiskRequest(CaseRequest):
    query: str = Field(min_length=1, examples=["Review ALT-2026-006: pass-through activity on a trading company account"])
    signals: dict[str, Any] = Field(
        examples=[{"pass_through_flag": 1, "velocity_score": 0.6, "transaction_amount": 250000, "country": "United Arab Emirates"}]
    )
    external_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, examples=[0.78])


class SanctionsScreenRequest(CaseRequest):
    entity_name: str = Field(min_length=1, max_length=300, examples=["Viktor Sampleov"])
    country: Optional[str] = Field(default=None, max_length=100, examples=["Canada"])
    dob: Optional[str] = Field(default=None, max_length=20, examples=["1990-01-01"])
    nationality: Optional[str] = Field(default=None, max_length=100)
    id_number: Optional[str] = Field(default=None, max_length=50)
    # True only if the DOB / ID were checked against an official document (KYC).
    identifiers_verified: bool = False
    # For a company: its owners from KYC records, for the 50% ownership rule.
    owners: Optional[list[Owner]] = Field(default=None, examples=[[{"name": "Khalid Testani", "percent": 30, "dob": "1975-05-05"}]])
