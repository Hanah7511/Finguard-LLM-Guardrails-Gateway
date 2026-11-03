"""Quick check that both running doors answer. Start them first: docker compose up -d

Run:  python scripts/check_doors.py
"""
import json
import urllib.request

import anyio
from mcp import Client


async def check_mcp() -> None:
    async with Client("http://127.0.0.1:8001/mcp") as client:
        tools = await client.list_tools()
        print("MCP door:", len(tools.tools), "tools")


with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as response:
    print("REST door:", json.load(response))
anyio.run(check_mcp)