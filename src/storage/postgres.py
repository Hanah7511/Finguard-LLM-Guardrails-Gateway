"""The PostgreSQL versions of the vault, the case record and the audit log.

They behave exactly like the in-memory ones, but both doors share them and a restart loses
nothing. The doors log in as a limited user that can only read and add rows. Any database
failure raises an error, so a decision is never made or returned without its record.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config.settings import Settings
from src.core.case_tracker import CaseRecord, steps_missing
from src.core.pii_vault import BaseVault, same_value_key
from src.monitoring.decision_logger import DecisionLogger
from src.storage.crypto import KEY_VERSION, VaultCipher


def create_storage_engine(url: str) -> Engine:
    # pool_pre_ping: reconnects after a database restart. hide_parameters: no values in error messages.
    # connect_timeout: fail in seconds when the database is down (the driver's default wait is minutes).
    return create_engine(url, pool_pre_ping=True, hide_parameters=True, connect_args={"connect_timeout": 5})


class PostgresVault(BaseVault):
    def __init__(self, engine: Engine, cipher: VaultCipher, entity_types: Iterable[str]) -> None:
        super().__init__(entity_types)
        self._engine = engine
        self._cipher = cipher

    def token_for(self, case_id: str, entity_type: str, value: str) -> str:
        value_hash = self._cipher.value_hash(case_id, entity_type, same_value_key(value))
        with self._engine.begin() as db:
            # One writer per case at a time, so two doors cannot hand out the same number.
            db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:case_id))"), {"case_id": case_id})
            found = db.execute(
                text("SELECT token FROM pii_vault WHERE case_id = :case_id AND entity_type = :entity_type AND value_hash = :value_hash"),
                {"case_id": case_id, "entity_type": entity_type, "value_hash": value_hash},
            ).scalar_one_or_none()
            if found:
                return found
            count = db.execute(
                text("SELECT count(*) FROM pii_vault WHERE case_id = :case_id AND entity_type = :entity_type"),
                {"case_id": case_id, "entity_type": entity_type},
            ).scalar_one()
            token = f"[{entity_type}_{count + 1}]"
            nonce, ciphertext = self._cipher.encrypt(case_id, token, value)
            db.execute(
                text(
                    "INSERT INTO pii_vault (case_id, token, entity_type, value_hash, ciphertext, nonce, key_version) "
                    "VALUES (:case_id, :token, :entity_type, :value_hash, :ciphertext, :nonce, :key_version)"
                ),
                {
                    "case_id": case_id, "token": token, "entity_type": entity_type, "value_hash": value_hash,
                    "ciphertext": ciphertext, "nonce": nonce, "key_version": KEY_VERSION,
                },
            )
            return token

    def _values(self, case_id: str, tokens: set[str]) -> dict[str, str]:
        with self._engine.connect() as db:
            rows = db.execute(
                text("SELECT token, nonce, ciphertext FROM pii_vault WHERE case_id = :case_id AND token = ANY(:tokens)"),
                {"case_id": case_id, "tokens": sorted(tokens)},
            ).all()
        return {r.token: self._cipher.decrypt(case_id, r.token, bytes(r.nonce), bytes(r.ciphertext)) for r in rows}


    def known_values(self, case_id: str, entity_types: Iterable[str]) -> dict[str, str]:
        with self._engine.connect() as db:
            rows = db.execute(
                text("SELECT token, nonce, ciphertext FROM pii_vault WHERE case_id = :case_id AND entity_type = ANY(:types)"),
                {"case_id": case_id, "types": list(entity_types)},
            ).all()
        return {self._cipher.decrypt(case_id, r.token, bytes(r.nonce), bytes(r.ciphertext)): r.token for r in rows}


class PostgresCaseTracker:
    """Adds one row per event and rebuilds the case by replaying them in order."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _add(self, case_id: Optional[str], event: str, detail: Optional[dict[str, Any]] = None) -> None:
        if not case_id:
            return
        with self._engine.begin() as db:
            db.execute(
                text("INSERT INTO case_events (case_id, event, detail) VALUES (:case_id, :event, CAST(:detail AS JSONB))"),
                {"case_id": case_id, "event": event, "detail": json.dumps(detail or {})},
            )

    def seen(self, case_id: Optional[str]) -> None:
        self._add(case_id, "seen")

    def risk_assessed(self, case_id: Optional[str]) -> None:
        self._add(case_id, "risk_assessed")

    def grounding_checked(self, case_id: Optional[str], grounded: bool) -> None:
        self._add(case_id, "grounding_checked", {"grounded": grounded})

    def sanctions_blocked(self, case_id: Optional[str]) -> None:
        self._add(case_id, "sanctions_blocked")

    def call_refused(self, case_id: Optional[str]) -> None:
        self._add(case_id, "call_refused")

    def report_approved(self, case_id: str, report_hash: str) -> None:
        self._add(case_id, "report_approved", {"report_hash": report_hash})

    def report_released(self, case_id: str, analyst: str) -> None:
        self._add(case_id, "report_released", {"analyst": analyst})

    def approved_reports(self, case_id: str) -> set[str]:
        with self._engine.connect() as db:
            rows = db.execute(
                text("SELECT detail FROM case_events WHERE case_id = :case_id AND event = 'report_approved'"),
                {"case_id": case_id},
            ).all()
        return {row.detail["report_hash"] for row in rows}

    def missing(self, case_id: str, required_steps: Iterable[str]) -> list[str]:
        with self._engine.connect() as db:
            rows = db.execute(
                text("SELECT event, detail FROM case_events WHERE case_id = :case_id ORDER BY id"),
                {"case_id": case_id},
            ).all()
        if not rows:
            return ["case_not_found"]
        record = CaseRecord()
        for row in rows:
            if row.event == "risk_assessed":
                record.risk_assessed = True
            elif row.event == "grounding_checked":  # only the latest check counts
                record.claims_grounded = row.detail["grounded"]
            elif row.event == "sanctions_blocked":
                record.sanctions_blocked = True
            elif row.event == "call_refused":
                record.refused_calls += 1
        return steps_missing(record, required_steps)


class PostgresDecisionLogger(DecisionLogger):
    """Same scrubbing as the file log. If the write fails the error reaches the caller."""

    def __init__(self, engine: Engine, scrubber: Optional[Callable[[str], str]] = None) -> None:
        super().__init__(None, scrubber)
        self._engine = engine

    def _write(self, record: dict[str, Any], line: str) -> None:
        # JSONB refuses the NUL character, and hostile text must not be able to stop the audit log.
        safe_json = line.replace("\\u0000", "\\ufffd")
        with self._engine.begin() as db:
            db.execute(
                text(
                    "INSERT INTO audit_log (request_id, event, case_id, policy_version, record) "
                    "VALUES (:request_id, :event, :case_id, :policy_version, CAST(:record AS JSONB))"
                ),
                {
                    "request_id": record.get("request_id") or "none",
                    "event": record.get("event") or "unknown",
                    "case_id": record.get("case_id"),
                    "policy_version": record.get("policy_version"),
                    "record": safe_json,
                },
            )


def build_postgres_storage(
    settings: Settings, entity_types: Iterable[str], scrubber: Optional[Callable[[str], str]] = None
) -> tuple[PostgresVault, PostgresCaseTracker, PostgresDecisionLogger]:
    """Refuses to start without the database, its tables or the key: never falls back to memory."""
    if not settings.database_url:
        raise RuntimeError("STORAGE_BACKEND=postgres needs DATABASE_URL")
    if not settings.vault_master_key:
        raise RuntimeError("STORAGE_BACKEND=postgres needs VAULT_MASTER_KEY")
    cipher = VaultCipher(settings.vault_master_key.get_secret_value())
    engine = create_storage_engine(settings.database_url.get_secret_value())
    try:
        with engine.connect() as db:
            db.execute(text("SELECT 1 FROM pii_vault, case_events, audit_log LIMIT 0"))
    except Exception as exc:
        # The type only: a connection error can quote the address, which holds the password.
        raise RuntimeError(
            f"Cannot use the database ({type(exc).__name__}). Is it running, and has the migrate step finished?"
        ) from None
    return PostgresVault(engine, cipher, entity_types), PostgresCaseTracker(engine), PostgresDecisionLogger(engine, scrubber)
