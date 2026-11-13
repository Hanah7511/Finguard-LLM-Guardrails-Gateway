"""The FastAPI door, for people and apps. The MCP door (for AI agents) is separate.

Each route checks the request shape (schemas.py), calls one orchestrator handle_*
method and returns its answer. No guardrail logic lives here, so both doors always
apply exactly the same rules.

Run:   uvicorn src.api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from src.api.schemas import AssessRiskRequest, ChatRequest, ReleaseRequest, SanctionsScreenRequest, ValidateRequest
from src.core.orchestrator import AccessDenied, Orchestrator, get_orchestrator
from src.llm.client import LLMUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the orchestrator once at startup (policy check, law store, sanctions lists),
    # so a bad policy file stops the server here instead of failing the first request.
    orch = get_orchestrator()
    orch.require_app_key_is_set()
    orch.require_analyst_key_is_set()
    orch.load_models()  # now, so the first user does not wait and a missing model stops the door here
    yield


app = FastAPI(
    title="FinGuard",
    description="Guardrails gateway for AML / financial-crime LLM work.",
    version="0.3.0",
    lifespan=lifespan,
)

# Routes are plain `def`, not `async def`: the checks run models and CPU work, and
# FastAPI runs plain functions in a thread pool so one slow request does not block others.


def require_app_key(
    x_api_key: Optional[str] = Header(default=None), orch: Orchestrator = Depends(get_orchestrator)
) -> None:
    try:
        orch.check_app_key(x_api_key)
    except AccessDenied as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def require_case_id(orch: Orchestrator, case_id: Optional[str], required: bool = True) -> None:
    try:
        orch.check_case_id(case_id, required)
    except AccessDenied as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# /health stays open, so a load balancer or Docker can check the server without a key.
APP_KEY = [Depends(require_app_key)]


@app.get("/health")
def health(orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Is the server up, which policy is active, and is screening real or test."""
    return {
        "status": "ok",
        "policy_version": orch.version,
        "sanctions_list_source": orch.sanctions.source,
        "llm_configured": orch.llm is not None,
    }


@app.get("/policy", dependencies=APP_KEY)
def policy(orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    return orch.handle_get_policy_info()


@app.post("/validate", dependencies=APP_KEY)
def validate(req: ValidateRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Input guards only, no LLM call. For apps that call their own LLM."""
    require_case_id(orch, req.case_id, required=False)
    return orch.handle_validate(req.text, req.consent, req.enforce_consent, req.case_id)


@app.post("/chat", dependencies=APP_KEY)
def chat(req: ChatRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Full pipeline: input guards -> law lookup -> LLM -> output guards."""
    require_case_id(orch, req.case_id, required=False)
    try:
        return orch.handle_chat(req.query, req.consent, req.customer_id, req.case_id)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/assess-risk", dependencies=APP_KEY)
def assess_risk(req: AssessRiskRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Tool 1 over HTTP. An upstream engine posts its alert here."""
    require_case_id(orch, req.case_id)
    parties = [p.model_dump() for p in req.parties] if req.parties else None
    try:
        return orch.handle_assess_risk(
            req.query, req.signals, req.external_risk_score, req.case_id, req.transaction_type, parties
        )
    except ValueError as exc:  # e.g. parties without a case_id
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/cases/{case_id}/release", dependencies=APP_KEY)
def release(
    case_id: str,
    req: ReleaseRequest,
    response: Response,
    x_analyst_key: Optional[str] = Header(default=None),
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict[str, Any]:
    """Real values, for a named analyst only. Deliberately not an MCP tool, so no agent can ask."""
    response.headers["Cache-Control"] = "no-store"  # real values must not sit in a cache
    require_case_id(orch, case_id)
    try:
        answer = orch.handle_release(case_id, req.report_text, x_analyst_key)
    except AccessDenied as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if not answer["released"]:
        raise HTTPException(status_code=409, detail={"reason": answer["reason"], "missing": answer["missing"]})
    return answer


@app.post("/sanctions-screen", dependencies=APP_KEY)
def sanctions_screen(req: SanctionsScreenRequest, orch: Orchestrator = Depends(get_orchestrator)) -> dict[str, Any]:
    """Tool 4 over HTTP. Screen one person or company, and a company's owners if given."""
    require_case_id(orch, req.case_id)
    owners = [o.model_dump() for o in req.owners] if req.owners else None
    try:
        return orch.handle_sanctions_screen(
            req.entity_name, req.country, req.case_id, req.dob, req.nationality, req.id_number, owners,
            req.identifiers_verified,
        )
    except ValueError as exc:  # e.g. owners adding up to more than 100%
        raise HTTPException(status_code=422, detail=str(exc)) from exc
