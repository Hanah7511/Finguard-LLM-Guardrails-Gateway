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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = sum(self._outcomes.values())
            held = self._outcomes["held_for_review"]
            layers = {}
            for layer, counts in sorted(self._layers.items()):
                timings = list(self._timings[layer]) if layer in self._timings else []
                layers[layer] = {**dict(sorted(counts.items())),
                                 "p50_ms": percentile(timings, 50), "p95_ms": percentile(timings, 95)}
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
                "layers": layers,
            }
