"""The attack classifier, with a fake embedder and a tiny made-up model, so MiniLM is never loaded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config.settings import get_settings
from src.input_guardrails.attack_classifier import AttackClassifier
from src.monitoring.telemetry import Decision

NAME = "fake-embedder"


class FakeEmbedder:
    """Gives each known text two fixed numbers instead of running MiniLM."""

    VECTORS = {"attack text": [3.0, 0.0], "unsure text": [0.0, 0.0], "normal text": [-3.0, 0.0]}

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([self.VECTORS[t] for t in texts])


def write_model(folder: Path, weights=(1.0, 0.0), embedder=NAME) -> Path:
    """score = sigmoid(first number): 3 -> 0.95, 0 -> 0.50, -3 -> 0.05."""
    path = folder / "model.json"
    path.write_text(json.dumps({"embedder": embedder, "weights": list(weights), "bias": 0.0}), encoding="utf-8")
    return path


def make(path: Path, embedder=None) -> AttackClassifier:
    return AttackClassifier(path, embedder or FakeEmbedder(), NAME, block_at=0.7, escalate_at=0.35, policy_version="test")


@pytest.mark.parametrize("text, expected", [
    ("attack text", Decision.BLOCK),
    ("unsure text", Decision.ESCALATE),
    ("normal text", Decision.PASS),
])
def test_each_score_band_gives_its_outcome(tmp_path, text, expected):
    decision = make(write_model(tmp_path)).check(text)
    assert decision.decision == expected
    assert text not in decision.reasoning_trace  # the audit log gets the score, never the text


@pytest.mark.parametrize("broken", ["missing file", "other embedder", "wrong number of weights"])
def test_a_broken_model_blocks_instead_of_letting_everything_through(tmp_path, broken):
    if broken == "missing file":
        path = tmp_path / "nothing_here.json"
    elif broken == "other embedder":
        path = write_model(tmp_path, embedder="some-other-model")
    else:
        path = write_model(tmp_path, weights=(1.0, 0.0, 0.0))
    decision = make(path).check("normal text")
    assert decision.decision == Decision.BLOCK
    assert decision.reasoning_trace.startswith("classifier_error")


def test_the_model_file_is_read_only_once(tmp_path):
    path = write_model(tmp_path)
    classifier = make(path)
    classifier.check("normal text")
    path.unlink()  # if it tried to read the file again, this would now fail
    assert classifier.check("attack text").decision == Decision.BLOCK
    assert classifier.check("normal text").decision == Decision.PASS


@pytest.mark.parametrize("block_at, escalate_at", [(0.5, 0.5), (0.3, 0.6), (1.5, 0.3)])
def test_cut_offs_in_the_wrong_order_are_refused(tmp_path, block_at, escalate_at):
    with pytest.raises(ValueError):
        AttackClassifier(write_model(tmp_path), FakeEmbedder(), NAME, block_at, escalate_at, "test")


def test_the_real_saved_model_fits_finguards_embedder():
    """No MiniLM needed: only checks the committed file matches what FinGuard will feed it."""
    path = Path("data/classifiers/attack_classifier.json")
    model = json.loads(path.read_text(encoding="utf-8"))
    assert model["embedder"] == get_settings().embedding_model
    assert len(model["weights"]) == 384  # all-MiniLM-L6-v2 gives 384 numbers per text


# ── Step 7b: plugged into FinGuard ──────────────────────────────────

from pydantic import ValidationError  # noqa: E402

from src.config.policy_loader import load_policy  # noqa: E402
from src.config.policy_schema import AttackClassifierPolicy  # noqa: E402
from src.core.orchestrator import Orchestrator  # noqa: E402
from src.core.sanctions_screener import SanctionsScreener  # noqa: E402
from src.input_guardrails.pii.base import PIIDetector  # noqa: E402
from src.input_guardrails.pipeline import InputGuardrailPipeline  # noqa: E402
from src.monitoring.decision_logger import DecisionLogger  # noqa: E402
from src.monitoring.telemetry import GuardrailDecision  # noqa: E402

POLICY = load_policy("1.13.0")


class FixedClassifier:
    """Always gives the same decision, so the pipeline can be tested without a model."""

    layer = AttackClassifier.layer

    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def check(self, text: str) -> GuardrailDecision:
        return GuardrailDecision(self.layer, self.decision, "attack_score=fixed", "test")


class CountingPII(PIIDetector):
    """Finds nothing, but counts calls: proves whether masking ran."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, text):
        self.calls += 1
        return []


def run(decision: Decision):
    pii = CountingPII()
    pipeline = InputGuardrailPipeline(POLICY, pii, attack_classifier=FixedClassifier(decision))
    return pipeline.run("Summarise this alert.", consent=True, run_llama_guard=False), pii


def test_policy_1_13_turns_it_on_and_older_policies_keep_it_off():
    assert POLICY.input_guardrails.attack_classifier.enabled
    assert not load_policy("1.12.0").input_guardrails.attack_classifier.enabled


def test_the_policy_refuses_cut_offs_in_the_wrong_order():
    with pytest.raises(ValidationError):
        AttackClassifierPolicy(enabled=True, block_at=0.5, escalate_at=0.6)


def test_escalate_stops_the_request_before_masking_and_asks_for_a_person():
    result, pii = run(Decision.ESCALATE)
    assert not result.allowed
    assert result.blocked_by == "input.attack_classifier"
    assert "human review" in result.message
    assert pii.calls == 0  # stopped before PII masking


def test_block_stops_the_request_with_the_block_message():
    result, _ = run(Decision.BLOCK)
    assert not result.allowed
    assert result.message.startswith("Request blocked")


def test_pass_lets_the_request_carry_on_to_masking():
    result, pii = run(Decision.PASS)
    assert result.allowed
    assert pii.calls == 1


def orchestrator() -> Orchestrator:
    settings = get_settings()
    test_list = SanctionsScreener.from_policy(settings.sanctions_test_list_path, POLICY.sanctions_screening, "test")
    return Orchestrator(POLICY, settings, sanctions=test_list, audit=DecisionLogger(None))


def test_tool_1_sends_an_unsure_query_to_review():
    orch = orchestrator()
    assert isinstance(orch.input.attack_classifier, AttackClassifier)  # built from the policy
    orch.input.attack_classifier = FixedClassifier(Decision.ESCALATE)
    result = orch.handle_assess_risk("Summarise this alert.", signals={})
    assert result["recommendation"] == "escalate_for_review"


def test_a_door_warms_up_the_classifier_as_it_starts():
    scored = []
    orch = orchestrator()
    orch.input.attack_classifier = type("Counts", (), {"score": lambda self, text: scored.append(text)})()
    orch.load_models()
    assert len(scored) == 1  # MiniLM and the weights are loaded before the first user arrives
