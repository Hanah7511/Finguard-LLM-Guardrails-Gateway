"""THE COORDINATOR: input guardrails -> retrieval -> LLM -> output guardrails, in order."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.input_guardrails.pipeline import InputGuardrailPipeline
from src.llm.client import LLMClient, LLMUnavailableError
from src.llm.prompts import SCHEMA_RETRY_NOTE, SYSTEM_PROMPT, build_user_prompt
from src.monitoring.cost_tracker import CostRecord, CostTracker
from src.monitoring.telemetry import Decision, GuardrailDecision
from src.output_guardrails.pipeline import OutputGuardrailPipeline, OutputStatus
from src.tool_call_guardrails.pipeline import ToolCallGuardrailPipeline
from src.vector_store.retriever import RegulationRetriever

ESCALATION_MESSAGE = "This requires analyst review."
HALLUCINATION_MESSAGE = (
    "The generated answer contradicted the cited regulation and was withheld. "
    "This requires analyst review."
)


@dataclass
class ChatOutcome:
    status: str  # APPROVED | BLOCKED | ESCALATED
    decisions: list[GuardrailDecision] = field(default_factory=list)
    costs: list[CostRecord] = field(default_factory=list)
    message: Optional[str] = None
    blocked_by: Optional[str] = None
    response_text: Optional[str] = None
    risk_level: Optional[str] = None
    cited_regulation: Optional[str] = None
    grounding_verdict: Optional[str] = None
    confidence_score: Optional[float] = None
    pii_masked: bool = False
    pii_entities: list[str] = field(default_factory=list)


class GuardrailPipeline:
    def __init__(
        self,
        input_pipeline: InputGuardrailPipeline,
        output_pipeline: OutputGuardrailPipeline,
        tool_pipeline: ToolCallGuardrailPipeline,
        retriever: RegulationRetriever,
        llm: Optional[LLMClient],
        cost_tracker: CostTracker,
        max_retries: int,
        policy_version: str,
        top_k: int = 3,
    ) -> None:
        self.input = input_pipeline
        self.output = output_pipeline
        self.tools = tool_pipeline
        self.retriever = retriever
        self.llm = llm
        self.cost_tracker = cost_tracker
        self.max_retries = max_retries
        self.policy_version = policy_version
        self.top_k = top_k

    def run_full_pipeline(self, query: str, consent: Optional[bool]) -> ChatOutcome:
        # 1. Input guardrails (cheapest first, stop on first block)
        inp = self.input.run(query, consent)
        outcome = ChatOutcome(
            status="BLOCKED",
            decisions=list(inp.decisions),
            pii_masked=bool(inp.pii_entities),
            pii_entities=inp.pii_entities,
        )
        if not inp.allowed:
            outcome.message, outcome.blocked_by = inp.message, inp.blocked_by
            return outcome

        if self.llm is None:
            raise LLMUnavailableError("No LLM configured; set OPENAI_API_KEY")

        # 2. Retrieve regulations and drop any carrying hidden instructions
        hits = self.retriever.search(inp.processed_text, k=self.top_k)
        docs = [doc for doc, _ in hits]
        scan = self.tools.scan_documents([d.text for d in docs])
        outcome.decisions.append(scan.decision)
        docs = [docs[i] for i in scan.clean_indices]

        # 3. LLM call, retried only when the response shape is wrong
        user_prompt = build_user_prompt(inp.processed_text, docs)
        result = None
        for attempt in range(self.max_retries + 1):
            completion = self.llm.complete(SYSTEM_PROMPT, user_prompt)
            outcome.costs.append(
                self.cost_tracker.record(completion.model, completion.prompt_tokens, completion.completion_tokens)
            )
            # 4. Output guardrails
            result = self.output.run(completion.text)
            outcome.decisions.extend(result.decisions)
            if result.status is not OutputStatus.SCHEMA_REJECTED:
                break
            reason = result.decisions[0].reasoning_trace
            user_prompt = f"{build_user_prompt(inp.processed_text, docs)}\n\n{SCHEMA_RETRY_NOTE.format(reason=reason)}"

        if result.status is OutputStatus.SCHEMA_REJECTED:
            outcome.decisions.append(
                GuardrailDecision(
                    layer="output.retry_budget",
                    decision=Decision.ESCALATE,
                    reasoning_trace=f"schema_rejected_after_{self.max_retries + 1}_attempts",
                    policy_version=self.policy_version,
                )
            )
            outcome.status, outcome.message = "ESCALATED", ESCALATION_MESSAGE
            return outcome

        response = result.response
        outcome.risk_level = response.risk_level
        outcome.cited_regulation = response.cited_regulation
        outcome.confidence_score = response.confidence_score
        outcome.grounding_verdict = result.grounding.verdict.value if result.grounding else "UNGROUNDED"

        if result.status is OutputStatus.APPROVED:
            outcome.status = "APPROVED"
            outcome.response_text = response.explanation
        elif result.status is OutputStatus.BLOCKED:
            outcome.status, outcome.message = "BLOCKED", HALLUCINATION_MESSAGE
            outcome.blocked_by = result.decisions[-1].layer
        else:
            # Escalated answers are withheld from the end user; analysts see them in review.
            outcome.status, outcome.message = "ESCALATED", ESCALATION_MESSAGE
        return outcome
