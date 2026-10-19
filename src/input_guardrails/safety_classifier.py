"""Financial-harm detection in two layers.

Layer 1: policy patterns (fast, exact intent phrases).
Layer 2: Llama Guard (understands meaning; catches paraphrased evasion). Optional,
         enabled in the policy once its latency is acceptable on your hardware.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Protocol

from src.config.policy_schema import LlamaGuardConfig
from src.input_guardrails.patterns import PatternMatcher
from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch

logger = logging.getLogger(__name__)


class PatternSafetyClassifier:
    layer = "input.safety_patterns"

    def __init__(self, patterns: Iterable[str], policy_version: str) -> None:
        self._matcher = PatternMatcher(patterns)
        self._policy_version = policy_version

    def check(self, text: str) -> GuardrailDecision:
        sw = Stopwatch()
        matches = self._matcher.find_all(text)
        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.BLOCK if matches else Decision.PASS,
            reasoning_trace=(
                f"financial_harm_pattern: {', '.join(matches)}" if matches else "no_patterns_found"
            ),
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata={"patterns_matched": matches} if matches else {},
        )


class SafetyModel(Protocol):
    def classify(self, text: str) -> tuple[bool, Optional[str]]:
        """Return (is_safe, violated_category)."""


# Appended to Llama Guard's standard categories so it reasons about AML evasion.
FINANCIAL_HARM_CATEGORY = (
    "S15: Financial Crime Evasion. Requests for help evading anti-money-laundering "
    "controls, structuring transactions to stay under reporting thresholds, bypassing "
    "KYC or sanctions screening, or concealing the source or movement of funds."
)


class HuggingFaceLlamaGuard:
    """Runs Llama Guard locally via transformers. Loaded on first use."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, torch_dtype="auto")
        self._model.eval()

    def classify(self, text: str) -> tuple[bool, Optional[str]]:
        import torch

        if self._model is None:
            self._load()
        conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        input_ids = self._tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            categories={"S15": FINANCIAL_HARM_CATEGORY},
        )
        with torch.no_grad():
            output = self._model.generate(input_ids, max_new_tokens=20, do_sample=False)
        verdict = self._tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        lines = [ln.strip() for ln in verdict.strip().splitlines() if ln.strip()]
        if lines and lines[0].lower() == "safe":
            return True, None
        return False, lines[1] if len(lines) > 1 else "unspecified"


class LlamaGuardClassifier:
    layer = "input.llama_guard"

    def __init__(
        self, config: LlamaGuardConfig, policy_version: str, model: Optional[SafetyModel] = None
    ) -> None:
        self._config = config
        self._policy_version = policy_version
        self._model = model
        if config.enabled and model is None:
            self._model = HuggingFaceLlamaGuard(config.model_id)

    def check(self, text: str) -> GuardrailDecision:
        sw = Stopwatch()
        if not self._config.enabled:
            return self._decision(Decision.SKIP, "disabled_in_policy", sw)
        try:
            is_safe, category = self._model.classify(text)
        except Exception as exc:  # model load/inference failure
            logger.exception("Llama Guard failed")
            fail_closed = self._config.on_error == "block"
            return self._decision(
                Decision.BLOCK if fail_closed else Decision.PASS,
                f"classifier_error ({type(exc).__name__}); on_error={self._config.on_error}",
                sw,
            )
        if is_safe:
            return self._decision(Decision.PASS, "classified_safe", sw)
        return self._decision(Decision.BLOCK, f"classified_unsafe: {category}", sw, {"category": category})

    def _decision(self, decision: Decision, trace: str, sw: Stopwatch, metadata: dict | None = None):
        return GuardrailDecision(
            layer=self.layer,
            decision=decision,
            reasoning_trace=trace,
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            metadata=metadata or {},
        )
