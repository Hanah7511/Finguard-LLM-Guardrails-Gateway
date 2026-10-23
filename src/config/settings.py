"""Environment-driven settings (.env). Policy *values* live in YAML, not here."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 30.0

    # Policy
    active_policy_version: str = "1.2.0"
    policy_dir: Path = PROJECT_ROOT / "src" / "config" / "policies"

    # Data
    regulations_path: Path = PROJECT_ROOT / "data" / "law_store" / "regulations.json"
    # Optional, gitignored: FATF/UAE/CBUAE text built by scripts/build_local_law_store.py
    local_regulations_path: Path = PROJECT_ROOT / "data" / "law_store" / "regulations_local.json"
     # Real lists, built by scripts/build_sanctions_lists.py (gitignored)
    sanctions_list_path: Path = PROJECT_ROOT / "data" / "sanctions" / "sanctions_list.json"
    # FICTIONAL list, used only when the real lists have not been built
    sanctions_test_list_path: Path = PROJECT_ROOT / "data" / "sanctions" / "test_sanctions_list.json"
    audit_log_path: Path = PROJECT_ROOT / "logs" / "audit.jsonl"

    # Models (loaded lazily, only when first needed)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    spacy_model: str = "en_core_web_sm"
    pii_backend: Literal["auto", "presidio", "regex"] = "auto"
    retrieval_top_k: int = 3

    # Servers
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    mcp_transport: Literal["stdio", "streamable-http", "sse"] = "stdio"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
