"""Append-only JSONL audit trail of guardrail decisions.

Callers only ever pass masked text and entity *types*. As defence in depth, every
free-text field is run through a PII scrubber before it is written, so a bug
upstream still cannot put a raw SSN or card number into the log.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("finguard.audit")

# Identifier fields that are safe and must not be altered by the scrubber.
_UNSCRUBBED_KEYS = frozenset(
    {"request_id", "timestamp", "policy_version", "layer", "decision", "event", "channel"}
)


class DecisionLogger:
    def __init__(self, path: Optional[Path], scrubber: Optional[Callable[[str], str]] = None) -> None:
        self._path = Path(path) if path else None
        self._scrub = scrubber or (lambda s: s)
        self._lock = threading.Lock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
        safe = self._scrub_value(record)
        line = json.dumps(safe, default=str, ensure_ascii=False)
        if self._path:
            with self._lock, self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        logger.info(line)
        return safe

    def _scrub_value(self, value: Any, key: Optional[str] = None) -> Any:
        if isinstance(value, dict):
            return {k: self._scrub_value(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub_value(v, key) for v in value]
        if isinstance(value, str) and key not in _UNSCRUBBED_KEYS:
            return self._scrub(value)
        return value
