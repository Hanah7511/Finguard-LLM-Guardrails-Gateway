
"""Create the vault, the case record, the audit log and the limited app user."""
import os

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None


def upgrade() -> None:
    # Real values, encrypted by FinGuard before they get here.
    op.execute("""
        CREATE TABLE pii_vault (
            case_id      TEXT        NOT NULL,
            token        TEXT        NOT NULL,
            entity_type  TEXT        NOT NULL,
            value_hash   BYTEA       NOT NULL,
            ciphertext   BYTEA       NOT NULL,
            nonce        BYTEA       NOT NULL,
            key_version  INTEGER     NOT NULL DEFAULT 1,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (case_id, token),
            UNIQUE (case_id, entity_type, value_hash)
        )
    """)

    # One new row per event. Tool 8 reads these to decide if a case is complete.
    op.execute("""
        CREATE TABLE case_events (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            case_id     TEXT        NOT NULL,
            event       TEXT        NOT NULL,
            detail      JSONB       NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX case_events_case_id ON case_events (case_id, id)")

    # Every guardrail decision. Outcomes only, never names or identifier values.
    op.execute("""
        CREATE TABLE audit_log (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            request_id      TEXT        NOT NULL,
            event           TEXT        NOT NULL,
            case_id         TEXT,
            policy_version  TEXT,
            logged_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            record          JSONB       NOT NULL
        )
    """)
    op.execute("CREATE INDEX audit_log_case_id ON audit_log (case_id, id)")
    op.execute("CREATE INDEX audit_log_request_id ON audit_log (request_id)")

    # History can only grow: the database itself refuses edits and deletes.
    op.execute("""
        CREATE FUNCTION refuse_change() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION USING MESSAGE = TG_TABLE_NAME || ' is append-only';
        END;
        $$ LANGUAGE plpgsql
    """)
    for table in ("case_events", "audit_log"):
        op.execute(f"""
            CREATE TRIGGER {table}_no_change BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION refuse_change()
        """)
        op.execute(f"""
            CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table}
            FOR EACH STATEMENT EXECUTE FUNCTION refuse_change()
        """)

    # The doors log in as this user: they can read and add rows, and nothing else.
    # The password is sent as a bound value, so it stays out of the SQL text and error messages.
    op.execute(
        sa.text("SELECT set_config('finguard.app_password', :password, true)")
        .bindparams(password=os.environ["APP_DB_PASSWORD"])
    )
    op.execute("""
        DO $$
        BEGIN
            EXECUTE 'CREATE ROLE finguard_app LOGIN PASSWORD '
                    || quote_literal(current_setting('finguard.app_password'));
        END;
        $$
    """)
    op.execute("GRANT SELECT, INSERT ON pii_vault, case_events, audit_log TO finguard_app")


def downgrade() -> None:
    op.execute("DROP TABLE audit_log, case_events, pii_vault")
    op.execute("DROP FUNCTION refuse_change()")
    op.execute("DROP ROLE finguard_app")

