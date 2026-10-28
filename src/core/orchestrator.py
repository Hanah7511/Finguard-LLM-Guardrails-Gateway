
"""THE BRAIN. FastAPI routes and MCP tools both call this file, never the guard modules.

One Orchestrator is built at startup and owns every guardrail component. Each
handle_* method runs only the checks that request needs, writes one audit record
(PII-free), and returns a plain dict, so both doors share exactly the same logic.
"""
from __future__ import annotations

import logging
import uuid
from collections import Counter
from functools import lru_cache
from typing import Any, Optional

from src.config.policy_loader import get_policy
from src.config.policy_schema import GuardrailPolicy, SanctionsScreeningPolicy
from src.config.settings import Settings, get_settings
from src.core.guardrail_pipeline import GuardrailPipeline
from src.core.risk_assessor import RiskAssessor
from src.core.sanctions_screener import SanctionsScreener
from src.input_guardrails.pii.base import PIIDetector
from src.input_guardrails.pii.presidio_detector import build_pii_detector
from src.input_guardrails.pii.regex_detector import build_log_scrubber
from src.input_guardrails.pipeline import InputGuardrailPipeline
from src.input_guardrails.safety_classifier import SafetyModel
from src.llm.client import LLMClient, LLMUnavailableError, build_llm_client
from src.monitoring.cost_tracker import CostTracker
from src.monitoring.decision_logger import DecisionLogger
from src.monitoring.telemetry import Decision, GuardrailDecision
from src.output_guardrails.grounding.base import Grounder, GroundingVerdict
from src.output_guardrails.grounding.cosine_grounding import CosineGrounder
from src.output_guardrails.grounding.nli_grounding import NLIGrounder
from src.output_guardrails.pipeline import OutputGuardrailPipeline
from src.tool_call_guardrails.pipeline import ToolCallGuardrailPipeline
from src.vector_store.retriever import Embedder, RegulationRetriever, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

_GROUNDING_REASONS = {
    GroundingVerdict.GROUNDED: "claim_supported_by_source",
    GroundingVerdict.HALLUCINATED: "claim_contradicts_source_document",
    GroundingVerdict.UNGROUNDED: "source_does_not_address_claim",
}
_SANCTIONS_DECISIONS = {
    "clear": Decision.PASS,
    "escalate_for_review": Decision.ESCALATE,
    "block_and_escalate": Decision.BLOCK,
}


def new_request_id() -> str:
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


class Orchestrator:
    def __init__(
        self,
        policy: GuardrailPolicy,
        settings: Settings,
        *,
        pii_detector: Optional[PIIDetector] = None,
        retriever: Optional[RegulationRetriever] = None,
        grounder: Optional[Grounder] = None,
        sanctions: Optional[SanctionsScreener] = None,
        llm: Optional[LLMClient] = None,
        audit: Optional[DecisionLogger] = None,
        safety_model: Optional[SafetyModel] = None,
    ) -> None:
        # Every argument after `settings` is optional so tests can plug in fakes.
        self.policy = policy
        self.version = policy.version
        embedder = SentenceTransformerEmbedder(settings.embedding_model)

        self.pii_detector = pii_detector or build_pii_detector(
            policy.input_guardrails, settings.pii_backend, settings.spacy_model
        )
        self.retriever = retriever or RegulationRetriever.from_json(settings.regulations_path, embedder, extra_paths=[settings.local_regulations_path])
        self.grounder = grounder or self._build_grounder(settings, embedder)

        self.input = InputGuardrailPipeline(policy, self.pii_detector, safety_model)
        self.tools = ToolCallGuardrailPipeline(policy, self.pii_detector)
        self.output = OutputGuardrailPipeline(policy, self.grounder, self.retriever, self.pii_detector)

        s = policy.sanctions_screening
        self.risk = RiskAssessor(
            policy.risk_assessment, s.high_risk_jurisdictions, self.version, tuple(s.increased_monitoring_jurisdictions),
        )
        self.sanctions = sanctions or self._load_sanctions(settings, s)

        self.cost_tracker = CostTracker()
        self.audit = audit or DecisionLogger(settings.audit_log_path, build_log_scrubber())
        self.llm = llm or build_llm_client(
            settings.openai_api_key, settings.llm_model, settings.llm_temperature, settings.llm_timeout_seconds
        )
        self.chat_pipeline = GuardrailPipeline(
            self.input,
            self.output,
            self.tools,
            self.retriever,
            self.llm,
            self.cost_tracker,
            policy.output_guardrails.max_retries,
            self.version,
            settings.retrieval_top_k,
        )

    @staticmethod
    def _load_sanctions(settings: Settings, cfg: SanctionsScreeningPolicy) -> SanctionsScreener:
        path, source = settings.sanctions_list_path, "official"
        if not path.exists():
            # Screening against fictional names must never pass silently as real screening.
            logger.warning(
                "Real sanctions lists not found at %s; using the FICTIONAL test list. "
                "Run scripts/build_sanctions_lists.py.", path
            )
            path, source = settings.sanctions_test_list_path, "test"
        return SanctionsScreener.from_json(
            path, cfg.lists, cfg.fuzzy_match_threshold, cfg.high_risk_jurisdictions, source, cfg.use_identifiers, tuple(cfg.increased_monitoring_jurisdictions),
        )

    def _build_grounder(self, settings: Settings, embedder: Embedder) -> Grounder:
        cfg = self.policy.output_guardrails
        if cfg.grounding_method == "cosine":
            return CosineGrounder(embedder, cfg.cosine_threshold)
        return NLIGrounder(cfg.entailment_threshold, cfg.contradiction_threshold, model_name=settings.nli_model)

    def _audit(self, event: str, decisions: list[GuardrailDecision], case_id: Optional[str], **extra: Any) -> str:
        request_id = new_request_id()
        self.audit.log(
            {
                "request_id": request_id,
                "event": event,
                "case_id": case_id,
                "policy_version": self.version,
                "decisions": [d.to_dict() for d in decisions],
                **extra,
            }
        )
        return request_id


    # ── FastAPI door ────────────────────────────────────────────────

    def handle_chat(
        self, query: str, consent: Optional[bool], customer_id: Optional[str] = None, case_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Full pipeline: input guards -> retrieval -> LLM -> output guards."""
        try:
            outcome = self.chat_pipeline.run_full_pipeline(query, consent)
        except LLMUnavailableError:
            self._audit("chat", [], case_id, customer_id=customer_id, status="LLM_UNAVAILABLE")
            raise
        request_id = self._audit(
            "chat",
            outcome.decisions,
            case_id,
            customer_id=customer_id,
            status=outcome.status,
            cost=CostTracker.combine(outcome.costs),
        )
        return {
            "request_id" : request_id,
            "status" : outcome.status,
            "response" : outcome.response_text,
            "message" : outcome.message,
            "blocked_by" : outcome.blocked_by,
            "risk_level" : outcome.risk_level,
            "cited_regulation" : outcome.cited_regulation,
            "grounding_verdict" : outcome.grounding_verdict,
            "confidence_score" : outcome.confidence_score,
            "pii_masked" : outcome.pii_masked,
            "policy_version" : self.version,

        }

    def handle_validate(
        self, text: str, consent: Optional[bool] = None, enforce_consent: bool = False, case_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Input guards only, no LLM call. For apps that call their own LLM."""
        result = self.input.run(text, consent, enforce_consent=enforce_consent)
        request_id = self._audit("validate", result.decisions, case_id, allowed=result.allowed)
        return {
            "request_id": request_id,
            "allowed": result.allowed,
            "masked_text": result.processed_text if result.allowed else None,
            "blocked_by": result.blocked_by,
            "message": result.message,
            "pii_found": result.pii_entities,
            "policy_version": self.version,
        }

    # ── MCP door (and reusable by any caller) ───────────────────────

    def handle_assess_risk(
        self,
        query: str,
        signals: dict[str, Any],
        external_risk_score: Optional[float] = None,
        case_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tool 1. Quick pattern scan of the query, then weighted risk scoring."""
        decisions = []
        for check in (self.input.injection, self.input.safety_patterns):
            decision = check.check(query)
            decisions.append(decision)
            if decision.stops_pipeline:
                request_id = self._audit("assess_risk", decisions, case_id, risk_level="HIGH")
                return {
                    "request_id": request_id,
                    "risk_score": 1.0,
                    "risk_level": "HIGH",
                    "recommendation": "block",
                    "reason": f"{decision.layer}: {decision.reasoning_trace}",
                    "policy_version": self.version,
                }

        assessment = self.risk.assess(signals, external_risk_score)
        decisions.append(assessment.decision)
        request_id = self._audit("assess_risk", decisions, case_id, risk_level=assessment.risk_level)
        return {
            "request_id": request_id,
            "risk_score": assessment.risk_score,
            "risk_level": assessment.risk_level,
            "recommendation": assessment.recommendation,
            "own_score": assessment.own_score,
            "external_score": assessment.external_score,
            "scores_disagree": assessment.scores_disagree,
            "risk_signals": sorted(assessment.contributing_signals),
            "ignored_signals": assessment.ignored_signals,
            "policy_version": self.version,
        }

    def handle_validate_tool_call(
        self, agent_name: str, tool_name: str, parameters: dict[str, Any], case_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Tool 2. Allowlist, then parameter checks, before the agent's tool runs."""
        v = self.tools.validate_call(agent_name, tool_name, parameters)
        request_id = self._audit(
            "validate_tool_call", v.decisions, case_id, agent=agent_name, tool=tool_name, approved=v.approved
        )
        return {
            "request_id": request_id,
            "approved": v.approved,
            "reason": v.reason,
            "agents_allowed_for_tool": v.agents_allowed_for_tool,
            "policy_version": self.version,
        }

    def handle_pii_check(self, text: str, case_id: Optional[str] = None) -> dict[str, Any]:
        """Tool 3. Mask PII in a tool result and scan it for hidden instructions."""
        r = self.tools.validate_result(text)
        problems = [d.layer for d in r.decisions if d.decision is Decision.BLOCK]
        request_id = self._audit("pii_check", r.decisions, case_id, safe_to_use=r.safe_to_use)
        return {
            "request_id": request_id,
            "masked_text": r.masked_text,
            "pii_found": r.pii_found,
            "safe_to_use": r.safe_to_use,
            "problems": problems,
            "policy_version": self.version,
        }

    def handle_sanctions_screen(
        self, entity_name: str, country: Optional[str] = None, case_id: Optional[str] = None, dob: Optional[str] = None, nationality: Optional[str] = None, id_number: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tool 4. Exact + fuzzy screening against the configured lists."""
        r = self.sanctions.screen(entity_name, country, dob, nationality, id_number)
        supplied = [k for k, v in (("dob", dob), ("nationality", nationality), ("id", id_number)) if v]
        statuses = Counter(m.status for m in r.matches if m.status)
        decision = GuardrailDecision(
            layer="compliance.sanctions",
            decision=_SANCTIONS_DECISIONS[r.recommendation],
            # The screened name is personal data: record the outcome, never the name.
            reasoning_trace=(
                f"matches={len(r.matches)} exact={r.exact_match} country_risk={r.country_risk} "
                f"lists={','.join(r.checked_lists)} "
                 # Which identifiers were supplied, never their values.
                f"identifiers={','.join(supplied) or 'none'} statuses={dict(statuses)}"
            ),
            policy_version=self.version,
        )
        request_id = self._audit("sanctions_screen", [decision], case_id, recommendation=r.recommendation)
        return {
            "request_id": request_id,
            "match_found": r.match_found,
            "exact_match": r.exact_match,
            "checked_lists": r.checked_lists,
            "list_source": self.sanctions.source,
            "matches": [
                {"listed_name": m.name, "list": m.list_name, "program": m.program, "score": m.score, "status": m.status, "reason": m.reason,}
                for m in r.matches
            ],
            "country_risk": r.country_risk,
            "recommendation": r.recommendation,
            "confidence": r.confidence,
            "policy_version": self.version,
        }

    def handle_validate_grounding(self, claims: list[dict[str, str]], case_id: Optional[str] = None) -> dict[str, Any]:
        """Tool 5. Each claim is checked against FinGuard's OWN copy of the cited law."""
        decisions, results = [], []
        for item in claims:
            decision, grounding, source_id = self.output.check_grounding_for_citation(item["citation"], item["claim"])
            decisions.append(decision)
            if grounding is None:
                verdict, reason = GroundingVerdict.UNGROUNDED, "cited_regulation_not_in_trusted_store"
            else:
                verdict, reason = grounding.verdict, _GROUNDING_REASONS[grounding.verdict]
            results.append(
                {
                    "citation": item["citation"],
                    "verdict": verdict.value,
                    "reason": reason,
                    "source_id": source_id,
                    "entailment_score": grounding.entailment_score if grounding else None,
                    "contradiction_score": grounding.contradiction_score if grounding else None,
                    "neutral_score": grounding.neutral_score if grounding else None,
                }
            )

        # One wrong claim is enough to stop the whole report.
        verdicts = {r["verdict"] for r in results}
        if GroundingVerdict.HALLUCINATED.value in verdicts:
            overall = GroundingVerdict.HALLUCINATED
        elif GroundingVerdict.UNGROUNDED.value in verdicts or not results:
            overall = GroundingVerdict.UNGROUNDED
        else:
            overall = GroundingVerdict.GROUNDED

        request_id = self._audit("validate_grounding", decisions, case_id, verdict=overall.value)
        return {
            "request_id": request_id,
            "verdict": overall.value,
            "safe_to_finalize": overall is GroundingVerdict.GROUNDED,
            "claims": results,
            "policy_version": self.version,
        }

    def handle_scan_documents(self, documents: list[str], case_id: Optional[str] = None) -> dict[str, Any]:
        """Tool 7. Find hidden instructions in retrieved (RAG) documents."""
        scan = self.tools.scan_documents(documents)
        request_id = self._audit("scan_documents", [scan.decision], case_id, flagged=len(scan.flagged_indices))
        return {
            "request_id": request_id,
            "safe_to_use": not scan.flagged_indices,
            "clean_indices": scan.clean_indices,
            "flagged_indices": scan.flagged_indices,
            "policy_version": self.version,
        }
    
    def handle_get_policy_info(self) -> dict[str, Any]:
        """Tool 6 and GET /policy. Which rules are active, for audit traceability."""
        meta = self.policy.policy_metadata
        out = self.policy.output_guardrails
        return {
            "active_policy_version": meta.version,
            "effective_date": meta.effective_date.isoformat(),
            "approved_by": meta.approved_by,
            "change_reason": meta.change_reason,
            "supersedes": meta.supersedes,
            "thresholds": {
                "grounding_method": out.grounding_method,
                "minimum_confidence": out.minimum_confidence,
                "entailment_threshold": out.entailment_threshold,
                "contradiction_threshold": out.contradiction_threshold,
            },
        }

@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    """The single shared Orchestrator, built on first use."""
    return Orchestrator(get_policy(), get_settings())

