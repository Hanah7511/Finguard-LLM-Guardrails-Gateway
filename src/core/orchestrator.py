
"""THE BRAIN. FastAPI routes and MCP tools both call this file, never the guard modules.

One Orchestrator is built at startup and owns every guardrail component. Each
handle_* method runs only the checks that request needs, writes one audit record
(PII-free), and returns a plain dict, so both doors share exactly the same logic.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import logging
import uuid
from collections import Counter
from functools import lru_cache
from typing import Any, Mapping, Optional

from src.config.policy_loader import get_policy
from src.config.policy_schema import GuardrailPolicy, SanctionsScreeningPolicy
from src.config.settings import Settings, get_settings
from src.core.case_tracker import CaseTracker
from src.core.guardrail_pipeline import GuardrailPipeline
from src.core.ownership import OwnerResult, OwnershipChecker
from src.core.pii_vault import NAME_TYPES, PIIVault
from src.core.risk_assessor import RiskAssessor
from src.core.sanctions_screener import SanctionsScreener
from src.input_guardrails.pii.base import PIIDetector
from src.input_guardrails.pii.presidio_detector import build_pii_detector
from src.input_guardrails.pii.regex_detector import build_log_scrubber
from src.input_guardrails.attack_classifier import AttackClassifier
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
from src.tool_call_guardrails.agent_identity import MIN_KEY_CHARS, analyst_key_env_name
from src.tool_call_guardrails.agent_identity import AgentIdentity
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
_SEVERITY = list(_SANCTIONS_DECISIONS)  # clear < escalate_for_review < block_and_escalate


def _owner_out(o: OwnerResult) -> dict[str, Any]:
    return {
        "name": o.name, "percent": o.percent, "status": o.status, "reason": o.reason,
        "listed_name": o.listed_name, "list": o.list_name, "owners": [_owner_out(x) for x in o.owners],
    }


def _hide_owner_names(owner: dict[str, Any], from_tokens: dict[str, str]) -> dict[str, Any]:
    return {
        **owner, "name": from_tokens.get(owner["name"], "[HIDDEN]"), "listed_name": None,
        "owners": [_hide_owner_names(o, from_tokens) for o in owner["owners"]],
    }


def _report_hash(report_text: str) -> str:
    """A fingerprint of the masked report, so a release can prove it is what Tool 8 approved."""
    return hashlib.sha256(report_text.encode()).hexdigest()


class AccessDenied(Exception):
    """A call refused at the door. It says why, and never contains a key."""


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
        agent_keys: Optional[Mapping[str, str]] = None,
        analyst_keys: Optional[Mapping[str, str]] = None,
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

        self.input = InputGuardrailPipeline(
            policy, self.pii_detector, safety_model, self._build_attack_classifier(policy, settings, embedder)
        )
        self.tools = ToolCallGuardrailPipeline(policy, self.pii_detector, self._build_identity(policy, agent_keys))
        self.analysts = self._build_analysts(policy, analyst_keys)
        self.output = OutputGuardrailPipeline(policy, self.grounder, self.retriever, self.pii_detector)
        i = policy.input_guardrails
        self.vault, self.cases, stored_audit = self._build_storage(
            settings,
            [*i.pii_entities, *(r.entity_type for r in i.custom_pii_recognizers), *(NAME_TYPES if i.party_masking else ())],
        )

        s = policy.sanctions_screening
        self.risk = RiskAssessor(
            policy.risk_assessment, s.high_risk_jurisdictions, self.version, tuple(s.increased_monitoring_jurisdictions),
        )
        self.sanctions = sanctions or self._load_sanctions(settings, s)
        self.ownership = OwnershipChecker(self.sanctions, s.ownership_rule)

        api_key = settings.finguard_api_key.get_secret_value() if settings.finguard_api_key else ""
        self._app_key = api_key if len(api_key) >= MIN_KEY_CHARS else None  # too short counts as no key

        self.cost_tracker = CostTracker()
        self.audit = audit or stored_audit or DecisionLogger(settings.audit_log_path, build_log_scrubber())
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
    def _build_storage(settings: Settings, entity_types: list[str]):
        """(vault, case record, audit log). The audit log is None in memory mode: it stays a file."""
        if settings.storage_backend == "postgres":
            from src.storage.postgres import build_postgres_storage

            return build_postgres_storage(settings, entity_types, build_log_scrubber())
        logger.warning("STORAGE_BACKEND=memory: a restart loses tokens and case records, and the two doors share nothing.")
        return PIIVault(entity_types), CaseTracker(), None

    def load_models(self) -> None:
        """Loads the models now, when a door starts, so the first user does not wait for them."""
        self.pii_detector.load()
        if self.input.attack_classifier:
            self.input.attack_classifier.score("warm up")  # reads the weights and loads MiniLM

    @staticmethod
    def _build_attack_classifier(policy: GuardrailPolicy, settings: Settings, embedder: Embedder) -> Optional[AttackClassifier]:
        """Reuses the MiniLM that law search already loads, so memory does not double. None when off."""
        cfg = policy.input_guardrails.attack_classifier
        if not cfg.enabled:
            return None
        return AttackClassifier(
            settings.attack_classifier_path, embedder, settings.embedding_model, cfg.block_at, cfg.escalate_at, policy.version
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
        return SanctionsScreener.from_policy(path, cfg, source)

    @staticmethod
    def _build_analysts(policy: GuardrailPolicy, analyst_keys: Optional[Mapping[str, str]]) -> Optional[AgentIdentity]:
        """Who may release real values. Tests pass their own keys; otherwise ANALYST_KEY_* from .env."""
        cfg = policy.analyst_release
        if not cfg.enabled:
            return None
        if analyst_keys is not None:
            return AgentIdentity(analyst_keys, policy.version)
        return AgentIdentity.from_env(cfg.allowed_analysts, policy.version, key_name=analyst_key_env_name)

    @staticmethod
    def _build_identity(policy: GuardrailPolicy, agent_keys: Optional[Mapping[str, str]]) -> Optional[AgentIdentity]:
        """Tests and the fake agent pass their own keys; otherwise read AGENT_KEY_* from .env."""
        if agent_keys is not None:
            return AgentIdentity(agent_keys, policy.version)
        cfg = policy.tool_call_guardrails
        if cfg.require_agent_keys:
            return AgentIdentity.from_env(cfg.allowlist, policy.version)
        return None


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

    def require_app_key_is_set(self) -> None:
        """Called when the REST door starts: without a key it could only refuse everyone."""
        if self.policy.access_control.require_app_key and not self._app_key:
            raise RuntimeError(f"Policy {self.version} needs FINGUARD_API_KEY, of at least {MIN_KEY_CHARS} characters")

    def check_app_key(self, sent: Optional[str]) -> None:
        """REST door. Raises AccessDenied. Constant-time compare, and the key is never logged."""
        if not self.policy.access_control.require_app_key:
            return
        if not (self._app_key and sent and hmac.compare_digest(sent.encode(), self._app_key.encode())):
            logger.warning("REST call refused: missing or wrong X-API-Key")
            raise AccessDenied("invalid_or_missing_api_key")

    def check_case_id(self, case_id: Optional[str], required: bool = True) -> None:
        """Raises AccessDenied. The value itself is not logged: it may not be a case id at all."""
        cfg = self.policy.access_control
        if not cfg.require_case_id:
            return
        if not case_id:
            if required:
                raise AccessDenied("case_id_required")
        elif not re.fullmatch(cfg.case_id_pattern, case_id):
            raise AccessDenied("case_id_bad_format")

    def check_agent(self, tool: str, case_id: Optional[str], agent_name: Optional[str], agent_key: Optional[str]) -> None:
        """MCP door, Tools 1, 3, 4 and 5. Raises AccessDenied. A refusal here is NOT added to the
        case record: otherwise anyone sending a wrong key could spoil someone else's case for good."""
        self.check_case_id(case_id)
        if not self.policy.access_control.require_agent_key_on_case_tools:
            return
        who = self.tools.identity.check(agent_name or "", agent_key)
        if who.stops_pipeline:
            self._audit("access_denied", [who], case_id, tool=tool, agent=agent_name)
            raise AccessDenied(who.reasoning_trace)

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
        transaction_type: Optional[str] = None,
        parties: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Tool 1. Quick pattern scan of the query, then weighted risk scoring and the CTR rule.
        Parties the caller lists become tokens. Raises ValueError for parties without a case_id."""
        if parties and not case_id and self.policy.input_guardrails.party_masking:
            raise ValueError("parties need a case_id: their tokens belong to a case")
        decisions = []
        for check in self.input.query_checks:
            decision = check.check(query)
            decisions.append(decision)
            if decision.stops_pipeline:
                self.cases.call_refused(case_id)
                request_id = self._audit("assess_risk", decisions, case_id, risk_level="HIGH")
                return {
                    "request_id": request_id,
                    "risk_score": 1.0,
                    "risk_level": "HIGH",
                    "recommendation": "escalate_for_review" if decision.decision == Decision.ESCALATE else "block",
                    "reason": f"{decision.layer}: {decision.reasoning_trace}",
                    "policy_version": self.version,
                }

        assessment = self.risk.assess(signals, external_risk_score, transaction_type)
        decisions.append(assessment.decision)
        self.cases.risk_assessed(case_id)
        party_tokens = self._register_parties(case_id, parties)
        request_id = self._audit(
            "assess_risk", decisions, case_id, risk_level=assessment.risk_level,
            ctr_required=assessment.ctr["required"], parties_masked=len(party_tokens),  # a count, never the names
        )
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
            "ctr": assessment.ctr,
            "parties": party_tokens,
            "policy_version": self.version,
        }

    def _register_parties(self, case_id: Optional[str], parties: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Store each party's names in the vault and return only their tokens."""
        if not parties or not self.policy.input_guardrails.party_masking:
            return []
        registered = []
        for party in parties:
            entity = "ORGANIZATION" if party.get("type") == "company" else "PERSON"
            registered.append({
                "token": self.vault.token_for(case_id, entity, party["name"]),
                "type": party.get("type", "person"),
                "aliases": [self.vault.token_for(case_id, entity, a) for a in party.get("aliases") or []],
            })
        return registered

    def handle_validate_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        parameters: dict[str, Any],
        case_id: Optional[str] = None,
        agent_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Tool 2. Agent identity (if the policy requires keys), allowlist, then parameter checks."""
        v = self.tools.validate_call(agent_name, tool_name, parameters, agent_key)
        if not v.approved:
            self.cases.call_refused(case_id)
        # The key itself is never logged or returned; only who it belongs to.
        request_id = self._audit(
            "validate_tool_call", v.decisions, case_id,
            agent=agent_name, verified_agent=v.verified_agent, tool=tool_name, approved=v.approved,
        )
        return {
            "request_id": request_id,
            "approved": v.approved,
            "reason": v.reason,
            "verified_agent": v.verified_agent,
            "agents_allowed_for_tool": v.agents_allowed_for_tool,
            "policy_version": self.version,
        }
    

    def handle_pii_check(self, text: str, case_id: Optional[str] = None) -> dict[str, Any]:
        """Tool 3. Mask PII in a tool result and scan it for hidden instructions.
        Tokens need a case to belong to: without a case_id the values are masked for good."""
        self.cases.seen(case_id)
        reversible = bool(self.policy.tool_call_guardrails.reversible_masking and case_id)
        known_found: list[str] = []
        if reversible and self.policy.input_guardrails.party_masking:
            text, known_found = self.vault.mask_known(case_id, text)
        r = self.tools.validate_result(text, self.vault.tokenizer(case_id) if reversible else None)
        problems = [d.layer for d in r.decisions if d.decision is Decision.BLOCK]
        request_id = self._audit("pii_check", r.decisions, case_id, safe_to_use=r.safe_to_use)
        return {
            "request_id": request_id,
            "masked_text": r.masked_text,
            "reversible": reversible,
            "pii_found": sorted({*r.pii_found, *(t[1:].rsplit("_", 1)[0] for t in known_found)}),
            "safe_to_use": r.safe_to_use,
            "problems": problems,
            "policy_version": self.version,
        }

    def handle_sanctions_screen(
        self, entity_name: str, country: Optional[str] = None, case_id: Optional[str] = None, dob: Optional[str] = None, nationality: Optional[str] = None, id_number: Optional[str] = None,
        owners: Optional[list[dict[str, Any]]] = None, identifiers_verified: bool = False,
    ) -> dict[str, Any]:
        """Tool 4. Exact + fuzzy screening against the configured lists, then the
        ownership rule when owners are given. Raises ValueError for impossible shares or a token
        this case never issued. If a token was sent, the answer never shows a name."""
        from_tokens: dict[str, str] = {}
        entity_name = self._name_from_token(case_id, entity_name, from_tokens)
        if owners:
            owners = self._owners_from_tokens(case_id, owners, from_tokens)
        r = self.sanctions.screen(entity_name, country, dob, nationality, id_number, identifiers_verified)
        recommendation, ownership, trace = r.recommendation, None, ""
        if owners:
            self.ownership.validate(owners)
        if owners and self.policy.sanctions_screening.ownership_rule.enabled:
            o = self.ownership.check(owners)
            # The stricter of the company's own result and its ownership result wins.
            recommendation = max(r.recommendation, o.recommendation, key=_SEVERITY.index)
            ownership = {
                "applied": True,
                "blocked_share": o.blocked_share,
                "possible_share": o.possible_share,
                "recommendation": o.recommendation,
                "reason": o.reason,
                "owners": [_owner_out(x) for x in o.owners],
            }
            trace = f" owners_screened={o.owners_screened} blocked_share={o.blocked_share:g} possible_share={o.possible_share:g}"
        elif owners:
            ownership = {"applied": False, "reason": f"the ownership rule is off in policy {self.version}"}
        supplied = [k for k, v in (("dob", dob), ("nationality", nationality), ("id", id_number)) if v]
        statuses = Counter(m.status for m in r.matches if m.status)
        decision = GuardrailDecision(
            layer="compliance.sanctions",
            decision=_SANCTIONS_DECISIONS[recommendation],
            # The screened name is personal data: record the outcome, never the name.
            reasoning_trace=(
                f"matches={len(r.matches)} exact={r.exact_match} country_risk={r.country_risk} "
                f"lists={','.join(r.checked_lists)} "
                 # Which identifiers were supplied, never their values.
                f"identifiers={','.join(supplied) or 'none'} verified={identifiers_verified} statuses={dict(statuses)}{trace}"
            ),
            policy_version=self.version,
        )
        if recommendation == "block_and_escalate":
            self.cases.sanctions_blocked(case_id)
        request_id = self._audit("sanctions_screen", [decision], case_id, recommendation=recommendation)
        if from_tokens and ownership and ownership.get("owners"):
            ownership["owners"] = [_hide_owner_names(o, from_tokens) for o in ownership["owners"]]
        return {
            "request_id": request_id,
            "match_found": r.match_found,
            "exact_match": r.exact_match,
            "checked_lists": r.checked_lists,
            "list_source": self.sanctions.source,
            "matches": [
                {"listed_name": None if from_tokens else m.name, "list": m.list_name, "program": m.program, "score": m.score, "matched_by": m.matched_by, "status": m.status, "reason": m.reason}
                for m in r.matches
            ],
            "country_risk": r.country_risk,
            "recommendation": recommendation,
            # Confidence is for the name screen of the entity itself, not its owners.
            "confidence": r.confidence,
            "ownership": ownership,
            "policy_version": self.version,
        }

    def _name_from_token(self, case_id: Optional[str], name: str, from_tokens: dict[str, str]) -> str:
        """The real name behind a token, remembered so the answer can hide it. Other text is returned as it is."""
        if not (self.policy.sanctions_screening.accept_tokens and self.vault.is_token(name)):
            return name
        value = self.vault.value_of(case_id, name) if case_id else None
        if value is None:
            raise ValueError(f"{name} is not a token of this case")
        from_tokens[value] = name
        return value

    def _owners_from_tokens(self, case_id: Optional[str], owners: list[dict[str, Any]], from_tokens: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {**o, "name": self._name_from_token(case_id, o["name"], from_tokens),
             "owners": self._owners_from_tokens(case_id, o.get("owners") or [], from_tokens)}
            for o in owners
        ]

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

        self.cases.grounding_checked(case_id, overall is GroundingVerdict.GROUNDED)
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
    
    def handle_finalize_check(
        self, agent_name: str, report_text: str, case_id: str, agent_key: Optional[str] = None
    ) -> dict[str, Any]:
        """Tool 8. The last gate: checks FinGuard's own record of the case. From v1.8.0 an approved
        report stays masked and only an analyst can get the real values; before that they came back
        here. A refused report stays masked, and an unverified caller gets no report at all."""
        cfg = self.policy.finalize_check
        decisions: list[GuardrailDecision] = []
        report, missing, unknown, restored, verified, caught = None, [], [], 0, None, 0
        if not cfg.enabled:
            reason = f"finalize_check is off in policy {self.version}"
        else:
            who = self.tools.identity.check(agent_name, agent_key)
            decisions.append(who)
            verified = who.metadata["verified_agent"]
            if who.stops_pipeline:
                reason = who.reasoning_trace
            elif agent_name not in cfg.allowed_agents:
                reason = "agent_not_allowed_to_finalize"
            else:
                missing = self.cases.missing(case_id, cfg.required_steps)
                if cfg.recheck_report:
                    report_text, caught = self._recheck_report(case_id, report_text)
                final_text, restored, unknown = self.vault.restore(case_id, report_text)
                if unknown:  # a token this case never issued: the report would go out with a blank in it
                    missing.append("unknown_tokens")
                if missing:
                    reason, restored = "required_steps_missing", 0
                    report = self.pii_detector.mask(report_text, self.vault.tokenizer(case_id)).masked_text
                else:
                    reason = "all_required_steps_done"
                    if cfg.return_real_values:
                        report = final_text
                    else:  # the masked report only: real values now go to an analyst, never to an agent
                        report, restored = report_text, 0
                        self.cases.report_approved(case_id, _report_hash(report_text))
        approved = reason == "all_required_steps_done"
        decisions.append(GuardrailDecision(
            layer="compliance.finalize",
            decision=Decision.PASS if approved else Decision.BLOCK,
            # Step names and a count only: the report and its values never reach the audit trail.
            reasoning_trace=f"{reason} missing={','.join(missing) or 'none'} tokens_restored={restored} values_caught={caught}",
            policy_version=self.version,
        ))
        request_id = self._audit(
            "finalize_check", decisions, case_id, agent=agent_name, verified_agent=verified, approved=approved
        )
        return {
            "request_id": request_id,
            "approved": approved,
            "reason": reason,
            "missing": missing,
            "unknown_tokens": unknown,
            "tokens_restored": restored,
            "values_caught": caught,  # real values the writer left in the report: a sign something slipped through
            "analyst_release_needed": approved and not cfg.return_real_values,
            "report_text": report,
            "policy_version": self.version,
        }

    def require_analyst_key_is_set(self) -> None:
        """Called when the REST door starts: without a key no report could ever be released."""
        cfg = self.policy.analyst_release
        if cfg.enabled and not any(self.analysts.has_key(a) for a in cfg.allowed_analysts):
            raise RuntimeError(f"Policy {self.version} needs a key for one of: {', '.join(cfg.allowed_analysts)} (ANALYST_KEY_<NAME>)")

    def handle_release(self, case_id: str, report_text: str, analyst_key: Optional[str]) -> dict[str, Any]:
        """REST only, and never an MCP tool: no agent can even ask for real values. Only the exact
        report Tool 8 approved can be released, and the case is checked again first, because it may
        have changed since. Raises AccessDenied for a bad key; the audit log records who asked."""
        if not self.policy.analyst_release.enabled:
            raise AccessDenied("release_is_off")
        self.check_case_id(case_id)
        analyst = self.analysts.owner(analyst_key)
        if analyst is None:
            refused = GuardrailDecision(
                layer="compliance.release", decision=Decision.BLOCK,
                reasoning_trace="invalid_analyst_key", policy_version=self.version,
            )
            self._audit("release", [refused], case_id, released=False)
            raise AccessDenied("invalid_analyst_key")
        missing = self.cases.missing(case_id, self.policy.finalize_check.required_steps)
        if _report_hash(report_text) not in self.cases.approved_reports(case_id):
            missing.append("report_not_approved")
        final_text, restored, unknown = self.vault.restore(case_id, report_text)
        if unknown:
            missing.append("unknown_tokens")
        released = not missing
        if released:
            self.cases.report_released(case_id, analyst)
        decision = GuardrailDecision(
            layer="compliance.release",
            decision=Decision.PASS if released else Decision.BLOCK,
            # Counts and step names only: the report and its values never reach the audit trail.
            reasoning_trace=f"analyst={analyst} missing={','.join(missing) or 'none'} tokens_restored={restored if released else 0}",
            policy_version=self.version,
        )
        # Audited before the answer exists: if this write fails, no real values are returned.
        request_id = self._audit("release", [decision], case_id, analyst=analyst, released=released)
        return {
            "request_id": request_id,
            "released": released,
            "reason": "released" if released else "not_releasable",
            "missing": missing,
            "tokens_restored": restored if released else 0,
            "report_text": final_text if released else None,
            "policy_version": self.version,
        }

    def _recheck_report(self, case_id: str, report_text: str) -> tuple[str, int]:
        """Mask anything still readable in the report. This does not refuse the report: a wrong
        guess by the detector must not hold up a good one, so the count is reported instead."""
        text, known = self.vault.mask_known(case_id, report_text)
        result = self.pii_detector.mask(text, self.vault.tokenizer(case_id))
        return result.masked_text, len(known) + len(result.entities)

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

