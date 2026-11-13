"""A small trained model that scores how likely a request is an attack.

MiniLM turns the text into 384 numbers. The saved weights turn those into one score:
    score = sigmoid(weights . numbers + bias)
Trained by scripts/train_attack_classifier.py. Only numpy is needed here, not scikit-learn.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch
from src.vector_store.retriever import Embedder


class AttackClassifier:
    layer = "input.attack_classifier"

    def __init__(
        self,
        model_path: Path,
        embedder: Embedder,
        embedder_name: str,
        block_at: float,
        escalate_at: float,
        policy_version: str,
    ) -> None:
        if not 0 <= escalate_at < block_at <= 1:
            raise ValueError("need 0 <= escalate_at < block_at <= 1")
        self._model_path = model_path
        self._embedder = embedder
        self._embedder_name = embedder_name
        self._block_at = block_at
        self._escalate_at = escalate_at
        self._version = policy_version
        self._weights: Optional[np.ndarray] = None
        self._bias = 0.0
        self._lock = threading.Lock()

    def load(self) -> None:
        """Reads the saved model once. Raises if it is missing or was made for another embedder."""
        with self._lock:  # two requests at once must not both load it
            if self._weights is not None:
                return
            model = json.loads(self._model_path.read_text(encoding="utf-8"))
            if model["embedder"] != self._embedder_name:
                raise ValueError("the model was trained with a different embedder")
            self._bias = float(model["bias"])
            self._weights = np.asarray(model["weights"], dtype="float64")

    def score(self, text: str) -> float:
        """0 = surely normal, 1 = surely an attack."""
        self.load()
        numbers = np.asarray(self._embedder.encode([text])[0], dtype="float64")
        if numbers.shape != self._weights.shape:
            raise ValueError("the embedder gives a different number of numbers than the model has weights")
        return float(1 / (1 + np.exp(-(numbers @ self._weights + self._bias))))

    def check(self, text: str) -> GuardrailDecision:
        sw = Stopwatch()
        try:
            score = self.score(text)
        except Exception as exc:  # a broken classifier must never let everything through
            return GuardrailDecision(
                layer=self.layer,
                decision=Decision.BLOCK,
                reasoning_trace=f"classifier_error: {type(exc).__name__}; failed_closed",
                policy_version=self._version,
                latency_ms=sw.ms(),
            )
        if score >= self._block_at:
            decision = Decision.BLOCK
        elif score >= self._escalate_at:
            decision = Decision.ESCALATE
        else:
            decision = Decision.PASS
        return GuardrailDecision(
            layer=self.layer,
            decision=decision,
            # the score and cut-offs only, never the text: this goes to the audit log
            reasoning_trace=f"attack_score={score:.2f} block_at={self._block_at} escalate_at={self._escalate_at}",
            policy_version=self._version,
            latency_ms=sw.ms(),
            metadata={"attack_score": round(score, 3)},
        )
