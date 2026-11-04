
"""Which agent is really calling: worked out from its secret key, not from the name it claims.

Each agent has its own key in .env, named after the agent:
investigation_agent -> AGENT_KEY_INVESTIGATION. FinGuard finds whose key was sent and
refuses the call if that is not the agent named in the request. Keys are compared in
constant time and never logged or returned; only agent names reach the audit trail.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
from collections import Counter
from typing import Iterable, Mapping, Optional

from dotenv import dotenv_values

from src.monitoring.telemetry import Decision, GuardrailDecision, Stopwatch

logger = logging.getLogger(__name__)

# Shorter keys are too easy to guess, so they are not accepted at all.
MIN_KEY_CHARS = 16


def key_env_name(agent: str) -> str:
    """investigation_agent -> AGENT_KEY_INVESTIGATION"""
    return "AGENT_KEY_" + re.sub(r"_AGENT$", "", agent.upper())


def analyst_key_env_name(analyst: str) -> str:
    """primary -> ANALYST_KEY_PRIMARY"""
    return "ANALYST_KEY_" + analyst.upper()


class AgentIdentity:
    layer = "tool.identity"

    def __init__(self, keys: Mapping[str, str], policy_version: str) -> None:
        """keys: agent name -> secret key."""
        usable = {}
        for agent, key in keys.items():
            if key and len(key) >= MIN_KEY_CHARS:
                usable[agent] = key.encode()
            elif key:
                logger.warning("Key for %s is shorter than %d characters; it is ignored.", agent, MIN_KEY_CHARS)
        shared = [k for k, n in Counter(usable.values()).items() if n > 1]
        if shared:
            # Two agents with one key would make "who called?" unanswerable.
            raise ValueError("two agents share the same key; give every agent its own key")
        self._keys = usable
        self._policy_version = policy_version

    @classmethod
    def from_env(cls, agents: Iterable[str], policy_version: str, env_file: str = ".env", key_name=key_env_name) -> "AgentIdentity":
        """Read AGENT_KEY_* for each agent from the environment, else from .env."""
        file_values = dotenv_values(env_file) if os.path.exists(env_file) else {}
        keys, missing = {}, []
        for agent in agents:
            name = key_name(agent)
            key = os.environ.get(name) or file_values.get(name)
            if key:
                keys[agent] = key
            else:
                missing.append(name)
        if missing:
            logger.warning("No agent key set for %s; calls from those agents will be refused.", ", ".join(missing))
        return cls(keys, policy_version)

    def has_key(self, name: str) -> bool:
        return name in self._keys

    def owner(self, key: Optional[str]) -> Optional[str]:
        """The agent this key belongs to, or None. Checks every key so timing shows nothing."""
        if not key:
            return None
        sent = key.encode()
        found = None
        for agent, real in self._keys.items():
            if hmac.compare_digest(sent, real):
                found = agent
        return found

    def check(self, claimed_agent: str, key: Optional[str]) -> GuardrailDecision:
        sw = Stopwatch()
        owner = self.owner(key)
        if not key:
            reason = "missing_agent_key"
        elif owner is None:
            reason = "unknown_agent_key"
        elif owner != claimed_agent:
            reason = "key_belongs_to_another_agent"
        else:
            reason = "agent_key_verified"
        ok = reason == "agent_key_verified"
        return GuardrailDecision(
            layer=self.layer,
            decision=Decision.PASS if ok else Decision.BLOCK,
            reasoning_trace=reason,
            policy_version=self._policy_version,
            latency_ms=sw.ms(),
            # Names only, never the key. verified_agent shows who really called, even when refused.
            metadata={"claimed_agent": claimed_agent, "verified_agent": owner},
        )
