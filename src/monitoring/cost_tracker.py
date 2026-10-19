"""Tracks token usage and cost per request."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

# USD per 1M tokens (input, output). Check against your provider's current price sheet.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4": (30.00, 60.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


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
    def __init__(self, pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self._pricing = dict(pricing or DEFAULT_PRICING)
        self._lock = threading.Lock()
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.request_count = 0

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> CostRecord:
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
        return rec

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
