"""Proves that both doors share a case and that it survives a restart. All values are made up.

Run:  python scripts/check_restart.py start
      docker compose down
      docker compose up -d
      python scripts/check_restart.py finish <the case id that start printed>
"""
import json
import os
import sys
import time
import urllib.request

import anyio
from dotenv import dotenv_values
from mcp import Client

MCP_URL = "http://127.0.0.1:8001/mcp"
RAW = "Deposit from IBAN GB82 WEST 1234 5698 7654 32."
CLAIM = {
    "claim": "Every bank shall file a report of any suspicious transaction relevant to a possible violation of law or regulation.",
    "citation": "31 CFR 1020.320(a)",
}


def secret(name: str) -> str:
    """Sent, never printed."""
    return os.environ.get(name) or dotenv_values(".env").get(name)


def signed(agent: str, key_name: str) -> dict:
    return {"agent_name": agent, "agent_key": secret(key_name)}


def rest_assess_risk(case_id: str) -> None:
    body = json.dumps({"query": "Review this alert", "signals": {"structuring_flag": 1}, "case_id": case_id}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8000/assess-risk", data=body, headers={"Content-Type": "application/json", "X-API-Key": secret("FINGUARD_API_KEY")}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print("REST door, risk assessed:", json.load(response)["risk_level"])


async def start() -> None:
    case_id = f"DEMO-{int(time.time())}"
    rest_assess_risk(case_id)
    async with Client(MCP_URL) as client:
        masked = await client.call_tool("finguard_pii_and_compliance_check", {"text": RAW, "case_id": case_id, **signed("investigation_agent", "AGENT_KEY_INVESTIGATION")})
        print("MCP door, masked text:  ", masked.structured_content["masked_text"])
        print("MCP door, checking the claim (the first time downloads the model, so this can take minutes)...")
        grounding = await client.call_tool("finguard_validate_grounding", {"claims": [CLAIM], "case_id": case_id, **signed("summarization_agent", "AGENT_KEY_SUMMARIZATION")})
        print("MCP door, safe to finalize:", grounding.structured_content["safe_to_finalize"])
    print("\nCase id:", case_id)


async def finish(case_id: str) -> None:
    async with Client(MCP_URL) as client:
        result = await client.call_tool("finguard_finalize_check", {
            "report_text": "Deposit from IBAN [IBAN_CODE_1].",
            "case_id": case_id,
            **signed("summarization_agent", "AGENT_KEY_SUMMARIZATION"),
        })
    r = result.structured_content
    print("approved:       ", r["approved"])
    print("missing:        ", r["missing"])
    print("tokens restored:", r["tokens_restored"])
    print("report:         ", r["report_text"])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "start":
        anyio.run(start)
    elif len(sys.argv) == 3 and sys.argv[1] == "finish":
        anyio.run(finish, sys.argv[2])
    else:
        print(__doc__)
