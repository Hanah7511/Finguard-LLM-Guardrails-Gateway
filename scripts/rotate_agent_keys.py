"""Replaces the three agent keys in .env with new random ones. Never prints a key.

Run:  python scripts/rotate_agent_keys.py
Then restart any running door (Docker: docker compose restart), so it reads the new keys.
"""
import re
import secrets
from pathlib import Path

ENV = Path(".env")
KEYS = ("AGENT_KEY_INVESTIGATION", "AGENT_KEY_COMPLIANCE", "AGENT_KEY_SUMMARIZATION")

text = ENV.read_text(encoding="utf-8")
for name in KEYS:
    # 32 random bytes from the operating system's secure generator: about 43 letters and digits
    text, found = re.subn(rf"(?m)^{name}=.*$", f"{name}={secrets.token_urlsafe(32)}", text)
    if found != 1:
        raise SystemExit(f"{name}: expected exactly 1 line in .env, found {found}. Nothing was changed.")
ENV.write_text(text, encoding="utf-8")
print(f"rotated {len(KEYS)} agent keys in .env (values not shown)")
