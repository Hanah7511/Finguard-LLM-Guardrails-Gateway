"""Tracks token usage, cost and time per LLM call. Prices come from data/llm_pricing.json."""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

from src.monitoring.metrics import percentile


def load_pricing(path: Path) -> dict[str, tuple[float, float]]:
    """USD per 1M tokens (input, output) for each model. A broken file stops the door from starting:
    a wrong price table would make every cost wrong without anyone noticing."""
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    pricing = {}
    for name, price in models.items():
        rates = (float(price["input"]), float(price["output"]))
        if min(rates) < 0:
            raise ValueError(f"{path.name}: negative price for {name}")
        pricing[name] = rates
    return pricing


@dataclass
class CostRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    priced: bool  # False when the model has no entry in the pricing table

    def to_dict(self) -> dict:
        return asdict(self)


class CostTracker:
    def __init__(self, pricing: Optional[Mapping[str, tuple[float, float]]] = None, window: int = 1000) -> None:
        self._pricing = dict(pricing or {})  # no table: every call counts as unpriced, never as free
        self._lock = threading.Lock()
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.request_count = 0
        self.unpriced_calls = 0
        self._llm_ms: deque = deque(maxlen=window)  # only the last `window` call times

    def record(
        self, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: Optional[float] = None
    ) -> CostRecord:
        rates = self._pricing.get(model) or self._match_prefix(model)
        cost = 0.0
        if rates:
            cost = (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000
        rec = CostRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
            priced=rates is not None,
        )
        with self._lock:
            self.total_tokens += rec.total_tokens
            self.total_cost_usd += rec.cost_usd
            self.request_count += 1
            if not rec.priced:
                self.unpriced_calls += 1
            if latency_ms is not None:
                self._llm_ms.append(latency_ms)
        return rec

    def totals(self) -> dict:
        """Every LLM call since this door started. cost_is_complete is False when some call had no
        price: cost_usd is then too low, not exact."""
        with self._lock:
            timings = list(self._llm_ms)
            return {
                "calls": self.request_count,
                "unpriced_calls": self.unpriced_calls,
                "cost_is_complete": self.unpriced_calls == 0,
                "tokens": self.total_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
                "p50_ms": percentile(timings, 50),
                "p95_ms": percentile(timings, 95),
            }

    def _match_prefix(self, model: str) -> tuple[float, float] | None:
        # "gpt-4o-2024-08-06" -> "gpt-4o"; longest prefix wins.
        for name in sorted(self._pricing, key=len, reverse=True):
            if model.startswith(name + "-"):
                return self._pricing[name]
        return None

    @staticmethod
    def combine(records: list[CostRecord]) -> dict:
        return {
            "llm_calls": len(records),
            "tokens": sum(r.total_tokens for r in records),
            "cost_usd": round(sum(r.cost_usd for r in records), 6),
        }
