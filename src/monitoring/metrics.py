"""What FinGuard decided since this door started: counts and timings only, never text, ids or names.

Kept in memory: a restart sets the counts back to zero, and each door (REST, MCP) counts only its own requests.
"""
from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from src.monitoring.telemetry import Decision, GuardrailDecision

BLOCKING = frozenset({Decision.BLOCK, Decision.REJECT})

# The text format Prometheus reads (version 0.0.4).
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    """Nearest rank: sort the values and take the one p% of the way along. None when there are none."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(math.ceil(p / 100 * len(ordered)) - 1, 0)]


class Metrics:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.since = datetime.now(timezone.utc).isoformat()
        self._outcomes: Counter = Counter()                          # allowed / blocked / held_for_review / failed
        self._events: Counter = Counter()                            # event name -> requests
        self._layers: dict[str, Counter] = defaultdict(Counter)      # layer -> decision -> count
        # only the last `window` timings per layer, so memory stays the same however long the door runs
        self._timings: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        # whole requests, from arrival to answer, per handler (assess_risk, chat, ...)
        self._handled: Counter = Counter()
        self._request_ms: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def record(self, event: str, decisions: list[GuardrailDecision], failed: bool = False) -> None:
        """One request: counted once as blocked, held for a person, allowed, or failed."""
        found = {Decision(d.decision) for d in decisions}
        if failed:
            outcome = "failed"
        elif found & BLOCKING:
            outcome = "blocked"
        elif Decision.ESCALATE in found:
            outcome = "held_for_review"
        else:
            outcome = "allowed"
        with self._lock:
            self._outcomes[outcome] += 1
            self._events[event] += 1
            for d in decisions:
                self._layers[d.layer][Decision(d.decision).value] += 1
                if d.latency_ms:
                    self._timings[d.layer].append(d.latency_ms)

    def record_time(self, handler: str, ms: float) -> None:
        """How long one whole request took. Refusals and errors count too: they are part of the real wait."""
        with self._lock:
            self._handled[handler] += 1
            self._request_ms[handler].append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = sum(self._outcomes.values())
            held = self._outcomes["held_for_review"]
            layers = {}
            for layer, counts in sorted(self._layers.items()):
                timings = list(self._timings[layer]) if layer in self._timings else []
                layers[layer] = {**dict(sorted(counts.items())),
                                 "p50_ms": percentile(timings, 50), "p95_ms": percentile(timings, 95)}
            request_time = {
                handler: {"count": count,
                          "p50_ms": percentile(self._request_ms[handler], 50),
                          "p95_ms": percentile(self._request_ms[handler], 95)}
                for handler, count in sorted(self._handled.items())
            }
            return {
                "since": self.since,
                "note": "Counts since this door started. A restart sets them back to zero, "
                        "and each door counts only its own requests.",
                "requests": {
                    "total": total,
                    **{k: self._outcomes[k] for k in ("allowed", "blocked", "held_for_review", "failed")},
                    "held_rate": round(held / total, 4) if total else None,
                },
                "by_event": dict(sorted(self._events.items())),
                "request_time_ms": request_time,
                "layers": layers,
            }


# ── Prometheus text format ─────────────────────────────────────────

def _label(value: Any) -> str:
    """A label value inside "...": a backslash, quote or new line must not break the line."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _line(name: str, value: float, **labels: Any) -> str:
    inside = ",".join(f'{k}="{_label(v)}"' for k, v in labels.items())
    return f"{name}{{{inside}}} {value}" if inside else f"{name} {value}"


def to_prometheus(metrics: dict[str, Any]) -> str:
    """The /metrics numbers in the format Prometheus reads: one number per line, labels in {}.
    Times are in seconds (the Prometheus convention). p50/p95 are gauges: they cover the last
    timings of this door only, so they cannot be added up across several FinGuard copies."""
    out: list[str] = []

    def family(name: str, kind: str, help_text: str, lines: list[str]) -> None:
        if lines:
            out.extend([f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *lines])

    def quantiles(name: str, p50: Optional[float], p95: Optional[float], **labels: Any) -> list[str]:
        return [_line(name, ms / 1000, **labels, quantile=q) for q, ms in (("0.5", p50), ("0.95", p95)) if ms is not None]

    requests, llm = metrics["requests"], metrics["llm"]
    family("finguard_info", "gauge", "The active guardrail policy.",
           [_line("finguard_info", 1, policy_version=metrics["policy_version"])])
    family("finguard_requests_total", "counter", "Requests since this door started, by outcome.",
           [_line("finguard_requests_total", requests[k], outcome=k)
            for k in ("allowed", "blocked", "held_for_review", "failed")])
    if requests["held_rate"] is not None:
        family("finguard_held_rate", "gauge", "Share of requests held for a person.",
               [_line("finguard_held_rate", requests["held_rate"])])
    family("finguard_events_total", "counter", "Requests since this door started, by kind.",
           [_line("finguard_events_total", n, event=e) for e, n in metrics["by_event"].items()])
    family("finguard_request_duration_seconds", "gauge", "Whole request time, p50 and p95 of the last requests.",
           [line for h, t in metrics["request_time_ms"].items()
            for line in quantiles("finguard_request_duration_seconds", t["p50_ms"], t["p95_ms"], handler=h)])
    family("finguard_layer_decisions_total", "counter", "Guardrail decisions, by layer and decision.",
           [_line("finguard_layer_decisions_total", n, layer=layer, decision=d)
            for layer, counts in metrics["layers"].items()
            for d, n in counts.items() if d not in ("p50_ms", "p95_ms")])
    family("finguard_layer_duration_seconds", "gauge", "Time per guardrail layer, p50 and p95 of the last checks.",
           [line for layer, counts in metrics["layers"].items()
            for line in quantiles("finguard_layer_duration_seconds", counts["p50_ms"], counts["p95_ms"], layer=layer)])
    family("finguard_llm_calls_total", "counter", "LLM calls since this door started.",
           [_line("finguard_llm_calls_total", llm["calls"])])
    family("finguard_llm_unpriced_calls_total", "counter", "LLM calls with no price: the cost total is then too low.",
           [_line("finguard_llm_unpriced_calls_total", llm["unpriced_calls"])])
    family("finguard_llm_tokens_total", "counter", "LLM tokens, prompt and answer together.",
           [_line("finguard_llm_tokens_total", llm["tokens"])])
    family("finguard_llm_cost_usd_total", "counter", "LLM cost in US dollars, from data/llm_pricing.json.",
           [_line("finguard_llm_cost_usd_total", llm["cost_usd"])])
    family("finguard_llm_duration_seconds", "gauge", "LLM call time, p50 and p95 of the last calls.",
           quantiles("finguard_llm_duration_seconds", llm["p50_ms"], llm["p95_ms"]))
    return "\n".join(out) + "\n"
