"""The FastAPI door, for people and apps. The MCP door (for AI agents) is separate.

Each route checks the request shape (schemas.py), calls one orchestrator handle_*
method and returns its answer. No guardrail logic lives here, so both doors always
apply exactly the same rules.

Run:   uvicorn src.api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import AssessRiskRequest, ChatRequest, SanctionsScreenRequest, ValidateRequest
from src.core.orchestrator import Orchestrator, get_orchestrator
from src.llm.client import LLMUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the orchestrator once at startup (policy check, law store, sanctions lists),
    # so a bad policy file stops the server here instead of failing the first request.
    get_orchestrator()
    yield


app = FastAPI(
    title="FinGuard",
    description="Guardrails gateway for AML / financial-crime LLM work.",
    version="0.3.0",
    lifespan=lifespan,
)

# Routes are plain `def`, not `async def`: the checks run models and CPU work, and
# FastAPI runs plain functions in a thread pool so one slow request does not block others.


@app.get("/health")
def health(orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Is the server up, which policy is active, and is screening real or test."""
    return {
        "status": "ok",
        "policy_version": orch.version,
        "sanctions_list_source": orch.sanctions.source,
        "llm_configured": orch.llm is not None,
    }


@app.get("/policy")
def policy(orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    return orch.handle_get_policy_info()


@app.post("/validate")
def validate(req: ValidateRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Input guards only, no LLM call. For apps that call their own LLM."""
    return orch.handle_validate(req.text, req.consent, req.enforce_consent, req.case_id)


@app.post("/chat")
def chat(req: ChatRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Full pipeline: input guards -> law lookup -> LLM -> output guards."""
    try:
        return orch.handle_chat(req.query, req.consent, req.customer_id, req.case_id)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/assess-risk")
def assess_risk(req: AssessRiskRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Tool 1 over HTTP. An upstream engine posts its alert here."""
    return orch.handle_assess_risk(req.query, req.signals, req.external_risk_score, req.case_id)


@app.post("/sanctions-screen")
def sanctions_screen(req: SanctionsScreenRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Tool 4 over HTTP. Screen one person or company."""
    return orch.handle_sanctions_screen(
        req.entity_name, req.country, req.case_id, req.dob, req.nationality, req.id_number
    )
