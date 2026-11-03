
"""How Alembic connects. The address holds the owner password, so it is only read from the environment."""
import os

from alembic import context
from sqlalchemy import create_engine

url = os.environ["MIGRATION_DATABASE_URL"]

if context.is_offline_mode():
    # Only writes the SQL out, without connecting. Used to review a change.
    context.configure(url=url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    # hide_parameters keeps the app password out of error messages.
    engine = create_engine(url, hide_parameters=True)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
