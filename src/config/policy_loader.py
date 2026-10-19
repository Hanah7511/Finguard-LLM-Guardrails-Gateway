"""Loads and validates the versioned guardrail policy. Every module reads policy from here."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from src.config.policy_schema import GuardrailPolicy
from src.config.settings import get_settings

_FILENAME = re.compile(r"^v(\d+\.\d+\.\d+)_guardrail_policy\.ya?ml$")


class PolicyError(RuntimeError):
    """Raised when a policy file is missing or invalid. Startup must fail on this."""


def policy_path(version: str, policy_dir: Optional[Path] = None) -> Path:
    directory = Path(policy_dir or get_settings().policy_dir)
    for suffix in (".yaml", ".yml"):
        candidate = directory / f"v{version}_guardrail_policy{suffix}"
        if candidate.exists():
            return candidate
    raise PolicyError(f"No policy file for version {version} in {directory}")


def load_policy(version: Optional[str] = None, policy_dir: Optional[Path] = None) -> GuardrailPolicy:
    version = version or get_settings().active_policy_version
    path = policy_path(version, policy_dir)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"{path.name}: invalid YAML: {exc}") from exc
    try:
        policy = GuardrailPolicy.model_validate(raw)
    except ValidationError as exc:
        raise PolicyError(f"{path.name}: policy failed validation:\n{exc}") from exc
    if policy.version != version:
        raise PolicyError(
            f"{path.name}: policy_metadata.version is {policy.version}, filename says {version}"
        )
    return policy


def list_policy_versions(policy_dir: Optional[Path] = None) -> list[str]:
    directory = Path(policy_dir or get_settings().policy_dir)
    versions = [m.group(1) for p in directory.iterdir() if (m := _FILENAME.match(p.name))]
    return sorted(versions, key=lambda v: tuple(int(x) for x in v.split(".")))


@lru_cache(maxsize=1)
def get_policy() -> GuardrailPolicy:
    """The active policy, loaded once at startup. Restart the service to pick up edits."""
    return load_policy()
