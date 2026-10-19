"""Telemetry primitives. Every guardrail check emits one GuardrailDecision."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Decision(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    MASK = "MASK"
    REJECT = "REJECT"
    APPROVE = "APPROVE"
    ESCALATE = "ESCALATE"
    SKIP = "SKIP"


STOPPING_DECISIONS = frozenset({Decision.BLOCK, Decision.REJECT, Decision.ESCALATE})


@dataclass
class GuardrailDecision:
    layer: str
    decision: Decision
    reasoning_trace: str
    policy_version: str
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def stops_pipeline(self) -> bool:
        return self.decision in STOPPING_DECISIONS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


class Stopwatch:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 3)
