"""FinGuard's own record of what happened in each case, read by Tool 8.

Tool 8 trusts this record, not what an agent says it did. Calls without a case_id cannot be
tracked. A sanctions block or a refused call stays on the record; a report can be fixed, so
for grounding only the latest check counts. In memory for now (PostgreSQL in Step 7): a
restart forgets every case, so Tool 8 then refuses them.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class CaseRecord:
    risk_assessed: bool = False
    claims_grounded: bool = False
    sanctions_blocked: bool = False
    refused_calls: int = 0


class CaseTracker:
    def __init__(self) -> None:
        self._cases: dict[str, CaseRecord] = {}
        self._lock = threading.Lock()

    def _update(self, case_id: Optional[str], **changes) -> None:
        if not case_id:
            return
        with self._lock:
            record = self._cases.setdefault(case_id, CaseRecord())
            for name, value in changes.items():
                setattr(record, name, value)

    def seen(self, case_id: Optional[str]) -> None:
        self._update(case_id)

    def risk_assessed(self, case_id: Optional[str]) -> None:
        self._update(case_id, risk_assessed=True)

    def grounding_checked(self, case_id: Optional[str], grounded: bool) -> None:
        self._update(case_id, claims_grounded=grounded)

    def sanctions_blocked(self, case_id: Optional[str]) -> None:
        self._update(case_id, sanctions_blocked=True)

    def call_refused(self, case_id: Optional[str]) -> None:
        if not case_id:
            return
        with self._lock:
            self._cases.setdefault(case_id, CaseRecord()).refused_calls += 1

    def missing(self, case_id: str, required_steps: Iterable[str]) -> list[str]:
        """Required steps this case has not met. A case FinGuard has never seen fails outright."""
        with self._lock:
            record = self._cases.get(case_id)
            if record is None:
                return ["case_not_found"]
            met = {
                "risk_assessed": record.risk_assessed,
                "claims_grounded": record.claims_grounded,
                "no_sanctions_block": not record.sanctions_blocked,
                "no_refused_calls": record.refused_calls == 0,
            }
        return [step for step in required_steps if not met[step]]
