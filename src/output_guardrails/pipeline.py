"""Runs output checks cheapest-first: schema -> PII leak -> grounding -> confidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from src.config.policy_schema import GuardrailPolicy
from src.input_guardrails.pii.base import PIIDetector
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch
from src.output_guardrails.confidence_gate import ConfidenceGate
from src.output_guardrails.grounding.base import Grounder, GroundingResult, GroundingVerdict
from src.output_guardrails.schema_validator import ComplianceResponse, SchemaValidator
from src.vector_store.retriever import RegulationRetriever


class OutputStatus(str, Enum):
    APPROVED = "APPROVED"
    SCHEMA_REJECTED = "SCHEMA_REJECTED"  # caller should retry the LLM
    BLOCKED = "BLOCKED"                  # hallucination: never shown to the user
    ESCALATED = "ESCALATED"              # needs analyst review


@dataclass
class OutputPipelineResult:
    status: OutputStatus
    decisions: list[GuardrailDecision] = field(default_factory=list)
    response: Optional[ComplianceResponse] = None
    grounding: Optional[GroundingResult] = None
    source_doc_id: Optional[str] = None


class OutputGuardrailPipeline:
    def __init__(
        self,
        policy: GuardrailPolicy,
        grounder: Grounder,
        retriever: RegulationRetriever,
        pii_detector: PIIDetector,
    ) -> None:
        self._version = policy.version
        self.schema = SchemaValidator(self._version)
        self.grounder = grounder
        self.retriever = retriever
        self.pii_detector = pii_detector
        self.confidence = ConfidenceGate(policy.output_guardrails.minimum_confidence, self._version)

    def run(self, raw_output: Union[str, dict]) -> OutputPipelineResult:
        decisions: list[GuardrailDecision] = []

        schema_decision, response = self.schema.validate(raw_output)
        decisions.append(schema_decision)
        if response is None:
            return OutputPipelineResult(OutputStatus.SCHEMA_REJECTED, decisions)

        # The LLM only saw masked input, but must not emit PII it invented or inferred.
        sw = Stopwatch()
        pii = self.pii_detector.mask(response.explanation)
        decisions.append(
            GuardrailDecision(
                layer="output.pii",
                decision=Decision.MASK if pii.found else Decision.PASS,
                reasoning_trace=f"masked: {', '.join(pii.entity_types)}" if pii.found else "no_pii_found",
                policy_version=self._version,
                latency_ms=sw.ms(),
            )
        )
        if pii.found:
            response = response.model_copy(update={"explanation": pii.masked_text})

        grounding_decision, grounding, source_id = self.check_grounding_for_citation(
            response.cited_regulation, response.explanation
        )
        decisions.append(grounding_decision)
        if grounding_decision.decision is Decision.BLOCK:
            return OutputPipelineResult(OutputStatus.BLOCKED, decisions, response, grounding, source_id)
        if grounding_decision.decision is Decision.ESCALATE:
            return OutputPipelineResult(OutputStatus.ESCALATED, decisions, response, grounding, source_id)

        confidence_decision = self.confidence.check(response.confidence_score)
        decisions.append(confidence_decision)
        status = OutputStatus.APPROVED if confidence_decision.decision is Decision.APPROVE else OutputStatus.ESCALATED
        return OutputPipelineResult(status, decisions, response, grounding, source_id)

    def check_grounding_for_citation(
        self, citation: str, claim: str
    ) -> tuple[GuardrailDecision, Optional[GroundingResult], Optional[str]]:
        sw = Stopwatch()
        layer = f"output.{self.grounder.method}_grounding"
        source = self.retriever.find_by_citation(citation)
        if source is None:
            return (
                GuardrailDecision(
                    layer=layer,
                    decision=Decision.ESCALATE,
                    reasoning_trace="cited_regulation_not_in_corpus",
                    policy_version=self._version,
                    latency_ms=sw.ms(),
                ),
                None,
                None,
            )
        decision, result = self.check_grounding(source.grounding_text, claim, layer=layer, source_id=source.doc_id)
        return decision, result, source.doc_id

    def check_grounding(
        self, source: str, claim: str, *, layer: Optional[str] = None, source_id: str = "provided"
    ) -> tuple[GuardrailDecision, GroundingResult]:
        sw = Stopwatch()
        result = self.grounder.score(source, claim)
        mapping = {
            GroundingVerdict.GROUNDED: Decision.PASS,
            GroundingVerdict.HALLUCINATED: Decision.BLOCK,
            GroundingVerdict.UNGROUNDED: Decision.ESCALATE,
        }
        decision = GuardrailDecision(
            layer=layer or f"output.{self.grounder.method}_grounding",
            decision=mapping[result.verdict],
            reasoning_trace=(
                f"verdict={result.verdict.value}, entailment={result.entailment_score:.2f}, "
                f"contradiction={result.contradiction_score:.2f}, source={source_id}"
            ),
            policy_version=self._version,
            latency_ms=sw.ms(),
            metadata=result.to_dict(),
        )
        return decision, result
