"""Tests against a real PostgreSQL that has had `alembic upgrade head` run on it.

Skipped unless both addresses are set, so plain `pytest` still needs no database:
  TEST_OWNER_DATABASE_URL  logs in as finguard_owner
  TEST_APP_DATABASE_URL    logs in as finguard_app, the limited user the doors use

Audit rows can never be deleted, so every test uses its own new case id. All values are made up.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import uuid

import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.config.policy_loader import load_policy
from src.config.settings import get_settings
from src.core.orchestrator import Orchestrator
from src.core.sanctions_screener import SanctionsScreener
from src.output_guardrails.grounding.base import Grounder, GroundingResult, GroundingVerdict
from src.storage.crypto import VaultCipher, VaultDecryptError
from src.storage.postgres import PostgresDecisionLogger, PostgresVault, build_postgres_storage, create_storage_engine

OWNER_URL = os.environ.get("TEST_OWNER_DATABASE_URL")
APP_URL = os.environ.get("TEST_APP_DATABASE_URL")
pytestmark = pytest.mark.skipif(not (OWNER_URL and APP_URL), reason="TEST_*_DATABASE_URL not set")

POLICY = load_policy("1.5.0")
MASTER_KEY = base64.urlsafe_b64encode(b"t" * 32).decode()
KEYS = {"investigation_agent": "test-key-investigation-0001", "summarization_agent": "test-key-summarization-0003"}
IBAN = "GB82 WEST 1234 5698 7654 32"
EMAIL = "jane.doe@example.com"
RAW = f"Deposit from IBAN {IBAN}, contact {EMAIL}."
CLAIMS = [{"claim": "Banks must report suspicious transactions.", "citation": "31 CFR 1020.320(a)"}]


class FakeGrounder(Grounder):
    method = "fake"

    def score(self, source: str, claim: str) -> GroundingResult:
        return GroundingResult(GroundingVerdict.GROUNDED, 1.0, 0.0, 0.0, self.method)


def postgres_settings(**changes):
    values = {"storage_backend": "postgres", "database_url": SecretStr(APP_URL), "vault_master_key": SecretStr(MASTER_KEY)}
    return get_settings().model_copy(update={**values, **changes})


def door() -> Orchestrator:
    """A new Orchestrator knows nothing except what is in the database, like a door after a restart."""
    settings = postgres_settings()
    test_list = SanctionsScreener.from_policy(settings.sanctions_test_list_path, POLICY.sanctions_screening, "test")
    return Orchestrator(POLICY, settings, sanctions=test_list, grounder=FakeGrounder(), agent_keys=KEYS)


def new_case() -> str:
    return f"TEST-{uuid.uuid4()}"


def do_required_steps(o: Orchestrator, case: str) -> str:
    o.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id=case)
    masked = o.handle_pii_check(RAW, case)["masked_text"]
    o.handle_validate_grounding(CLAIMS, case)
    return masked


def finalize(o: Orchestrator, report: str, case: str) -> dict:
    return o.handle_finalize_check("summarization_agent", report, case, KEYS["summarization_agent"])


@pytest.fixture(scope="module")
def owner():
    engine = create_storage_engine(OWNER_URL)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def app():
    engine = create_storage_engine(APP_URL)
    yield engine
    engine.dispose()


def vault_on(engine) -> PostgresVault:
    return PostgresVault(engine, VaultCipher(MASTER_KEY), ["IBAN_CODE", "EMAIL_ADDRESS"])


# The two problems 7B exists to fix

def test_a_case_survives_a_restart():
    case = new_case()
    masked = do_required_steps(door(), case)
    r = finalize(door(), masked, case)
    assert r["approved"] is True
    assert r["report_text"] == RAW


def test_both_doors_share_one_case():
    case = new_case()
    rest, mcp = door(), door()
    rest.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id=case)
    masked = mcp.handle_pii_check(RAW, case)["masked_text"]
    rest.handle_validate_grounding(CLAIMS, case)
    assert finalize(mcp, masked, case)["approved"] is True


def test_the_same_value_keeps_its_token_after_a_restart():
    case = new_case()
    first = door().handle_pii_check(f"From {IBAN}", case)["masked_text"]
    second = door().handle_pii_check(f"To {IBAN.replace(' ', '')}", case)["masked_text"]
    assert first == "From [IBAN_CODE_1]" and second == "To [IBAN_CODE_1]"


# Tool 8 gives the same answers as the in-memory version

def test_case_never_seen_is_refused():
    r = finalize(door(), "Report", new_case())
    assert (r["approved"], r["missing"]) == (False, ["case_not_found"])


def test_skipped_steps_are_listed():
    case, o = new_case(), door()
    masked = o.handle_pii_check(RAW, case)["masked_text"]
    r = finalize(o, masked, case)
    assert r["missing"] == ["risk_assessed", "claims_grounded"]
    assert IBAN not in json.dumps(r)


def test_only_the_latest_grounding_check_counts():
    case, o = new_case(), door()
    masked = do_required_steps(o, case)
    o.cases.grounding_checked(case, False)
    assert finalize(o, masked, case)["missing"] == ["claims_grounded"]
    o.cases.grounding_checked(case, True)
    assert finalize(o, masked, case)["approved"] is True


def test_a_sanctions_block_and_a_refused_call_stay_on_the_record():
    case, o = new_case(), door()
    masked = do_required_steps(o, case)
    o.handle_sanctions_screen("Khalid Testani", case_id=case, dob="1975-05-05")
    o.handle_sanctions_screen("Nobody Matchingname", case_id=case)
    o.handle_validate_tool_call("summarization_agent", "transaction_lookup", {}, case, KEYS["summarization_agent"])
    assert finalize(door(), masked, case)["missing"] == ["no_sanctions_block", "no_refused_calls"]


def test_a_token_from_another_case_is_refused():
    a, b, o = new_case(), new_case(), door()
    do_required_steps(o, a)
    o.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id=b)
    o.handle_validate_grounding(CLAIMS, b)
    r = finalize(o, "Report on [IBAN_CODE_1]", b)
    assert r["missing"] == ["unknown_tokens"]
    assert IBAN not in json.dumps(r)


# Two doors at the same moment

def test_many_writers_get_one_token_for_one_value(app):
    case, vaults = new_case(), [vault_on(app), vault_on(app)]
    tokens, errors, barrier = [], [], threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            tokens.append(vaults[i % 2].token_for(case, "IBAN_CODE", IBAN))
        except Exception as exc:  # reported below: an exception in a thread would otherwise vanish
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert set(tokens) == {"[IBAN_CODE_1]"}


def test_many_writers_get_different_numbers_for_different_values(app):
    case, vaults = new_case(), [vault_on(app), vault_on(app)]
    tokens, errors, barrier = [], [], threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            tokens.append(vaults[i % 2].token_for(case, "IBAN_CODE", f"GB00VALUE{i}"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sorted(tokens) == sorted(f"[IBAN_CODE_{n}]" for n in range(1, 9))


# What is actually stored

def test_the_database_holds_no_readable_value(owner):
    case, o = new_case(), door()
    o.handle_pii_check(RAW, case)
    with owner.connect() as db:
        rows = db.execute(text("SELECT * FROM pii_vault WHERE case_id = :c"), {"c": case}).all()
    assert len(rows) == 2
    stored = b"".join(bytes(r.ciphertext) + bytes(r.nonce) + bytes(r.value_hash) + r.token.encode() for r in rows)
    assert IBAN.encode() not in stored and EMAIL.encode() not in stored
    assert all(r.key_version == 1 for r in rows)


def test_equal_values_in_two_cases_cannot_be_matched(owner, app):
    a, b, vault = new_case(), new_case(), vault_on(app)
    vault.token_for(a, "IBAN_CODE", IBAN)
    vault.token_for(b, "IBAN_CODE", IBAN)
    with owner.connect() as db:
        hashes = db.execute(text("SELECT value_hash FROM pii_vault WHERE case_id IN (:a, :b)"), {"a": a, "b": b}).scalars().all()
    assert len(hashes) == 2 and bytes(hashes[0]) != bytes(hashes[1])


def test_a_row_copied_to_another_case_will_not_decrypt(owner, app):
    a, b, vault = new_case(), new_case(), vault_on(app)
    vault.token_for(a, "IBAN_CODE", IBAN)
    with owner.begin() as db:
        db.execute(
            text(
                "INSERT INTO pii_vault (case_id, token, entity_type, value_hash, ciphertext, nonce) "
                "SELECT :b, token, entity_type, value_hash, ciphertext, nonce FROM pii_vault WHERE case_id = :a"
            ),
            {"a": a, "b": b},
        )
    with pytest.raises(VaultDecryptError):
        vault.restore(b, "[IBAN_CODE_1]")


def test_a_changed_row_will_not_decrypt(owner, app):
    case, vault = new_case(), vault_on(app)
    vault.token_for(case, "IBAN_CODE", IBAN)
    with owner.begin() as db:
        db.execute(text("UPDATE pii_vault SET ciphertext = ciphertext || '\\x00'::bytea WHERE case_id = :c"), {"c": case})
    with pytest.raises(VaultDecryptError):
        vault.restore(case, "[IBAN_CODE_1]")


def test_the_wrong_master_key_will_not_decrypt(app):
    case = new_case()
    vault_on(app).token_for(case, "IBAN_CODE", IBAN)
    wrong = PostgresVault(app, VaultCipher(base64.urlsafe_b64encode(b"w" * 32).decode()), ["IBAN_CODE"])
    with pytest.raises(VaultDecryptError):
        wrong.restore(case, "[IBAN_CODE_1]")


def test_real_values_and_keys_never_reach_the_audit_log(owner):
    case, o = new_case(), door()
    assert finalize(o, do_required_steps(o, case), case)["approved"] is True
    with owner.connect() as db:
        rows = db.execute(text("SELECT event, record FROM audit_log WHERE case_id = :c ORDER BY id"), {"c": case}).all()
    logged = json.dumps([r.record for r in rows])
    assert rows[-1].event == "finalize_check"
    assert rows[-1].record["verified_agent"] == "summarization_agent"
    assert IBAN not in logged and EMAIL not in logged
    assert KEYS["summarization_agent"] not in logged


def test_the_nul_character_cannot_stop_the_audit_log(owner, app):
    request_id = str(uuid.uuid4())
    PostgresDecisionLogger(app).log({"request_id": request_id, "event": "test", "note": "bad\x00text"})
    with owner.connect() as db:
        record = db.execute(text("SELECT record FROM audit_log WHERE request_id = :r"), {"r": request_id}).scalar_one()
    assert record["note"] == "bad�text"


# The rules in the database

@pytest.mark.parametrize("statement", [
    "UPDATE audit_log SET event = 'changed'",
    "DELETE FROM audit_log",
    "TRUNCATE audit_log",
    "UPDATE case_events SET event = 'changed'",
    "DELETE FROM case_events",
    "TRUNCATE case_events",
])
def test_the_owner_cannot_change_history(owner, statement):
    with owner.begin() as db:  # needs at least one row, or the row trigger has nothing to fire on
        db.execute(text("INSERT INTO audit_log (request_id, event, record) VALUES ('setup', 'test', '{}')"))
        db.execute(text("INSERT INTO case_events (case_id, event) VALUES ('setup', 'test')"))
    with pytest.raises(DBAPIError, match="append-only"), owner.begin() as db:
        db.execute(text(statement))


@pytest.mark.parametrize("statement", [
    "UPDATE audit_log SET event = 'changed'",
    "DELETE FROM audit_log",
    "TRUNCATE audit_log",
    "DROP TABLE audit_log",
    "UPDATE case_events SET event = 'changed'",
    "DELETE FROM case_events",
    "UPDATE pii_vault SET token = 'changed'",
    "DELETE FROM pii_vault",
    "DROP TABLE pii_vault",
    "CREATE TABLE sneaky (id INTEGER)",
])
def test_the_app_user_can_only_read_and_add(app, statement):
    with pytest.raises(DBAPIError, match="permission denied|must be owner"), app.begin() as db:
        db.execute(text(statement))


# Refusing to run without its safety

@pytest.mark.parametrize("changes, message", [
    ({"database_url": None}, "DATABASE_URL"),
    ({"vault_master_key": None}, "VAULT_MASTER_KEY"),
])
def test_refuses_to_start_when_a_setting_is_missing(changes, message):
    with pytest.raises(RuntimeError, match=message):
        build_postgres_storage(postgres_settings(**changes), ["IBAN_CODE"])


def test_refuses_to_start_with_a_bad_key():
    with pytest.raises(ValueError, match="VAULT_MASTER_KEY"):
        build_postgres_storage(postgres_settings(vault_master_key=SecretStr("short")), ["IBAN_CODE"])


def test_refuses_to_start_without_the_database_and_hides_the_password():
    url = "postgresql+psycopg://finguard_app:do-not-show-me@127.0.0.1:1/finguard"
    with pytest.raises(RuntimeError, match="Cannot use the database") as err:
        build_postgres_storage(postgres_settings(database_url=SecretStr(url)), ["IBAN_CODE"])
    assert "do-not-show-me" not in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_a_failed_audit_write_fails_the_request():
    case, o = new_case(), door()
    o.audit = PostgresDecisionLogger(create_storage_engine("postgresql+psycopg://x:y@127.0.0.1:1/none"))
    with pytest.raises(DBAPIError):
        o.handle_assess_risk("Review this alert", {"structuring_flag": 1}, case_id=case)
