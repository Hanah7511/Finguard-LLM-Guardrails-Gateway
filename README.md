# FinGuard LLM Guardrails Gateway

> **LLMOps · Safety Infrastructure**  
> A composable, policy-driven safety middleware for LLM applications in financial services.  
> Two entry points — FastAPI for humans, MCP for AI agents — one set of guardrail rules.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![MCP](https://img.shields.io/badge/MCP_Server-8_Tools-blue?style=flat)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Audit_+_Encrypted_Vault-336791?style=flat&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![CI](https://github.com/Hanah7511/Finguard-LLM-Guardrails-Gateway/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-In_Progress-orange?style=flat)

---

## What is FinGuard?

LLM-powered systems in financial services face risks that standard application security doesn't address:

- A chatbot handling customer queries can be tricked into leaking account data
- An AI agent filing a Suspicious Activity Report (SAR) can hallucinate a regulation and produce a non-compliant document
- An agent with tool access can be manipulated into calling a privileged API it shouldn't touch

FinGuard sits between any LLM application and the outside world, enforcing a layered set of checks before input reaches the model and before output reaches the user or downstream system.

> **Independence by design.** FinGuard never imports from any downstream project. It communicates only through its two doors — an HTTP API and an MCP server — using a fixed contract. Any project can plug in by writing its own YAML configuration. No code changes required.

---

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Composability** | PII detection and grounding sit behind ABC interfaces (`PIIDetector`, `Grounder`). Swap Presidio for a regex detector, NLI grounding for cosine similarity — without changing orchestration logic. |
| **Observability-first** | Every guardrail decision emits a `GuardrailDecision` object: `layer`, `decision`, `reasoning_trace`, `policy_version`, `latency_ms`, `metadata`, `timestamp`. Each audit record adds `request_id` and `case_id`. Nothing is silent. |
| **Policy-as-versioned-code** | Rules live in YAML with `approved_by` and `change_reason` fields. A regulator can reconstruct which policy was active for any decision in the audit trail. |
| **Defense-in-depth** | Six independent guardrail layers. Uncertain outputs route to human review. No single failure exposes the system. |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ENTRY POINTS                               │
├──────────────────────────┬──────────────────────────────────────────┤
│   FastAPI  (humans/apps) │         MCP Server  (AI agents)          │
│   POST /chat             │         finguard_assess_risk()           │
│   POST /validate         │         finguard_validate_tool_call()    │
│   GET  /policy           │         finguard_pii_and_compliance_check│
│   GET  /health           │         finguard_sanctions_screen()      │
│   GET  /metrics          │         finguard_validate_grounding()    │
│                          │         finguard_get_policy_info()       │
│                          │         finguard_scan_documents()        │
│                          │         finguard_finalize_check()        │
└──────────────┬───────────┴──────────────────┬──────────────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
         ┌────────────────────────────────────┐
         │          ORCHESTRATOR              │
         │     (one handle_* per job)         │
         └────────────────┬───────────────────┘
                          │
   ┌──────────┬──────────┼──────────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼          ▼          ▼
 Input    Tool-Call   Output     Risk     Sanctions    Law
Pipeline  Pipeline   Pipeline  Assessor  Screener     Store
```

---

## Guardrail Layers (6)

### Layer 1 — Input Guardrails (OWASP LLM01)
All requests are scanned before reaching the model. Pipeline runs in performance order:
- **Prompt injection detection** (normalised against case, Unicode and punctuation tricks)
- **Safety pattern matching**
- **PII detection and masking** (Presidio, with a checksum-validated regex fallback)

> **Reversible PII masking (from policy v1.5.0).** Real values are held in FinGuard under a per-case token (`[US_SSN_1]`), encrypted with AES-256-GCM in PostgreSQL. Memory mode remains for tests and quick local runs. Agents work only with tokens. Real values come from one place, because a SAR filed with FinCEN needs them: `POST /cases/{case_id}/release`, for a named analyst, and only for the exact report Tool 8 approved. It is REST only, so no agent can even ask. From policy v1.6.0, names are masked too, in two ways: people and companies the caller lists at Tool 1 are matched exactly, and Presidio looks for other people's names. Detection is not perfect: a name that is neither listed nor found stays visible, and ordinary words are sometimes masked by mistake (it once masked "Customer IBAN"). Treat masking as one layer of protection, and keep human review. How often it misses has not been measured yet (Step 8).

### Layer 2 — Tool-Call Guardrails (OWASP LLM06)
Every agent tool call passes through an allowlist check before execution, and every tool result is PII-scanned on the way out. Agent identity comes from a **per-agent secret key** (from policy v1.4.0), not the agent name, which can be forged.

### Layer 3 — Output Guardrails (OWASP LLM09)
LLM responses validated in three stages:
1. Pydantic schema validation
2. NLI-based grounding check — agent sends claim + citation only; FinGuard fetches authoritative law text from its own trusted store
3. Confidence gate — routes uncertain outputs to human review

### Layer 4 — Safety Guardrails
Two-layer pipeline: fast regex pattern matching (Layer 1) + optional Llama Guard 1B for nuanced harm classification (Layer 2). Each runs independently.

### Layer 5 — Privacy Guardrails
PII (bank and card numbers, IBANs, SSNs, emails) masked before it reaches the LLM, and scrubbed again from all audit logs. Consent flag enforced per request.

### Layer 6 — Compliance Guardrails
NLI grounding against official law texts (US BSA regulations; FATF Recommendations, UAE Federal Decree-Law No. 10 of 2025 and CBUAE guidance kept locally, see [Data Sources](#data-sources--regulatory-basis)). Sanctions screening against the UN, UAE Local Terrorist List and OFAC SDN lists, with FATF black- and grey-list country risk. Versioned YAML policies with `approved_by` and `change_reason`. Append-only, PII-scrubbed audit trail (JSONL today, PostgreSQL planned) with policy version on every decision record.

---

## MCP Tools (8)

*Tools 1–8 are live in `src/mcp_server/server.py`.* From policy v1.7.0, tools that change a case need a `case_id` in the set format, plus `agent_name` and `agent_key`. A call refused for a bad key is audited, but not added to the case record. Each tool's description is written for the agent: when to call it and what to do with the answer. From policy v1.4.0, Tool 2 requires the agent's secret key (`agent_key`) and works out the caller from it, so a self-reported name is not enough. In finintel the tool wrapper will add the key in code, so the LLM never sees it. That also keeps Tool 8's real values away from the LLM: without a key it cannot ask for them, and the wrapper sends the approved report to human review instead of back to the model.

| # | Tool | Description |
|---|------|-------------|
| 01 | `finguard_assess_risk()` | Risk triage before any investigation starts. Scores alert signals against YAML weights. Returns: `LOW / MEDIUM / HIGH`. With `transaction_type`, also says if a Currency Transaction Report is required (cash of more than $10,000; a reporting duty, so it does not change the score). With `parties` and a `case_id`, stores the case's people and companies in the vault and returns only their tokens. |
| 02 | `finguard_validate_tool_call()` | Authorizes or blocks an agent's tool request. Checks the agent's secret key, then the YAML allowlist, then the parameters. No LLM call — fast. |
| 03 | `finguard_pii_and_compliance_check()` | PII detection and reversible masking. With a `case_id`, values become per-case tokens (`[IBAN_CODE_1]`); the same value keeps its token within a case. Uses Presidio + regex fallback. Real values stored in FinGuard, not returned. |
| 04 | `finguard_sanctions_screen()` | Screens a person or company against the UN, UAE Local Terrorist List and OFAC SDN lists (exact + fuzzy names, aliases). Date of birth / ID number confirm or rule out a name match; a name alone goes to review, and an exact name match is only cleared on identifiers marked `identifiers_verified` (KYC). For a company, `owners` applies the OFAC 50% rule (sanctioned owners' shares added up, through chains of companies). Country risk from the FATF black and grey lists. Accepts a token such as `[PERSON_1]`: FinGuard looks up the name itself, and the answer shows no names. |
| 05 | `finguard_validate_grounding()` | NLI hallucination check. Agent sends claim + citation reference only — FinGuard fetches the authoritative law text. Agent cannot supply a fake source. |
| 06 | `finguard_get_policy_info()` | Returns active policy version, last change, `approved_by`, and applicable rules. Also available via `GET /policy`. |
| 07 | `finguard_scan_documents()` | Scans RAG-retrieved documents for indirect prompt injection — hidden instructions embedded in regulatory text or case files. |
| 08 | `finguard_finalize_check()` | Final gate before human review. Needs the agent's key. Checks FinGuard's own record of the `case_id` against the steps required in the YAML: risk assessed, claims grounded, no sanctions block, no refused calls. Approved: the report stays masked, ready for an analyst. Refused: `missing` says why and the report stays masked. Masks anything still readable in the report and returns the count in `values_caught`. This does not refuse the report. |

---

## Orchestrator — handle_* Pattern

The orchestrator loads all workers once at startup (PII masker, NLI checker, sanctions screener, law store, pipelines, audit log). Both doors share the same orchestrator — no logic is duplicated.

| Function | Door | Workers used |
|----------|------|--------------|
| `handle_chat()` | API `/chat` | All: input → law lookup → GPT → output guards |
| `handle_validate()` | API `/validate` | Input guards only — no GPT, no cost |
| `handle_assess_risk()` | MCP Tool 1 | Injection + safety scan → risk assessor |
| `handle_validate_tool_call()` | MCP Tool 2 | Agent key → allowlist → parameter check |
| `handle_pii_check()` | MCP Tool 3 | PII masker → hidden instruction scanner |
| `handle_sanctions_screen()` | MCP Tool 4 | Sanctions screener only |
| `handle_validate_grounding()` | MCP Tool 5 | Law store + NLI checker, per claim |
| `handle_get_policy_info()` | MCP Tool 6 + `/policy` | YAML rulebook reader |
| `handle_scan_documents()` | MCP Tool 7 | Hidden instruction scanner |
| `handle_finalize_check()` | MCP Tool 8 | Agent key → case record → approve the masked report |
| `handle_release()` | REST only | App key + analyst key → case record → same report → real values |

> Every `handle_*` follows the same 4 steps: **(1)** Call the right workers → **(2)** Collect decisions (PASS / BLOCK / MASK) → **(3)** Write one audit record with `case_id`, no private data → **(4)** Return a simple answer with reason.

---

## Complete AML Investigation Flow

End-to-end flow when FinGuard is used by `finintel_sentinel_ai` (LangGraph multi-agent SAR investigation system):

```
1. finrisk engine → CRITICAL alert
   └─ signals: structuring_flag, velocity_score, sanctions_country_flag, device_risk_score

2. Tool 1: finguard_assess_risk(case_id, alert)
   └─ returns risk_level: HIGH · all subsequent calls carry the same case_id

3. Investigation Agent (wrapped calls)
   ├─ [wrapper] Tool 2: validate_tool_call → customer_history APPROVED
   └─ [wrapper] Tool 3: pii_check → SSN → [US_SSN_1]

4. Compliance Agent (wrapped calls)
   ├─ [wrapper] Tool 2: validate_tool_call → sanctions_check APPROVED
   ├─ Tool 4: sanctions_screen (real name from finintel DB)
   └─ Tool 7: scan_documents (RAG results scanned for injection)

5. Summarization Agent → SAR draft
   └─ Tool 5: validate_grounding(claims_list, citations)
      └─ FinGuard fetches real law text → NLI check → GROUNDED or BLOCKED

6. Tool 8: finguard_finalize_check(case_id)
   └─ ✅ risk assessed · ✅ claims grounded · ✅ no sanctions block
      ✅ no refused calls → [US_SSN_1] → 123-45-6789

7. The masked report goes to a human analyst
   └─ Analyst calls the release endpoint with their own key · SAR filed with FinCEN
```

---

## Security Model

Five properties enforced by construction — not by agent cooperation:

| Fix | Property |
|-----|----------|
| **FIX 1** | **Agent cannot skip guardrails.** In finintel, every tool is wrapped: `validate_tool_call()` → run tool → `pii_check()`. The agent calls the wrapper, never the tool directly. |
| **FIX 2** | **All calls linked by `case_id`.** Every tool call carries a `case_id`. Tool 8 reads FinGuard's own record of the case before approving finalization. |
| **FIX 3** | **Agent identity from secret key, not self-report.** A compromised agent cannot impersonate a higher-privileged one by changing its `agent_name` field. |
| **FIX 4** | **Agent cannot supply its own grounding source.** Tool 5 accepts claim + citation reference only. FinGuard fetches the law text — a hallucinating agent cannot provide a fake source that confirms its own false claim. |
| **FIX 5** | **Reversible masking for regulatory compliance.** SARs must contain real SSN/TIN. FinGuard masks values as tokens for all AI steps, and only a named analyst can get them back, through a REST endpoint that no agent can reach. Both properties hold simultaneously. |

---

## Folder Structure

```
finguard-llm-guardrails-gateway/
├── src/
│   ├── config/
│   │   └── policies/
│   │       ├── v1.0.0_guardrail_policy.yaml   # versioned policy (approved_by, change_reason)
│   │       ├── v1.1.0_guardrail_policy.yaml
│   │       ├── v1.2.0_guardrail_policy.yaml
│   │       ├── v1.3.0_guardrail_policy.yaml
│   │       ├── v1.4.0_guardrail_policy.yaml
│   │       ├── v1.5.0_guardrail_policy.yaml
│   │       ├── v1.6.0_guardrail_policy.yaml
│   │       ├── v1.7.0_guardrail_policy.yaml
│   │       ├── v1.8.0_guardrail_policy.yaml
│   │       ├── v1.9.0_guardrail_policy.yaml
│   │       ├── v1.10.0_guardrail_policy.yaml
│   │       ├── v1.11.0_guardrail_policy.yaml
│   │       ├── v1.12.0_guardrail_policy.yaml  # active
│   │       └── CHANGELOG.md
│   ├── core/
│   │   ├── orchestrator.py          # THE BRAIN — one handle_* per job
│   │   ├── guardrail_pipeline.py
│   │   ├── case_tracker.py          # FinGuard's own record of each case, read by Tool 8
│   │   ├── ownership.py             # OFAC 50% ownership rule
│   │   ├── pii_vault.py             # per-case tokens and the real values behind them
│   │   ├── risk_assessor.py
│   │   └── sanctions_screener.py
│   ├── input_guardrails/
│   │   ├── patterns.py
│   │   ├── injection_detector.py
│   │   ├── safety_classifier.py
│   │   ├── pii/
│   │   │   ├── base.py
│   │   │   ├── regex_detector.py
│   │   │   └── presidio_detector.py
│   │   └── pipeline.py
│   ├── tool_call_guardrails/
│   │   ├── agent_identity.py
│   │   ├── tool_allowlist.py
│   │   ├── pre_execution_gate.py
│   │   ├── post_execution_gate.py
│   │   ├── indirect_injection_scanner.py
│   │   └── pipeline.py
│   ├── output_guardrails/
│   │   ├── schema_validator.py
│   │   ├── grounding/
│   │   │   ├── base.py
│   │   │   ├── cosine_grounding.py
│   │   │   └── nli_grounding.py
│   │   ├── confidence_gate.py
│   │   └── pipeline.py
│   ├── llm/
│   │   ├── client.py
│   │   └── prompts.py
│   ├── vector_store/
│   │   └── retriever.py             # FinGuard's own law store: citation lookup → FAISS (NumPy fallback)
│   ├── mcp_server/                  # MCP door
│   │   └── server.py                # 8 tools, one per handle_*
│   ├── api/                         # FastAPI door
│   │   ├── main.py                  # 6 endpoints, one per handle_*
│   │   └── schemas.py               # request shapes (422 on bad input)
│   ├── storage/
│   │   ├── crypto.py                # AES-256-GCM for the vault
│   │   └── postgres.py              # database vault, case record and audit log
│   └── monitoring/
│       ├── telemetry.py
│       ├── decision_logger.py
│       └── cost_tracker.py
├── test/
│   ├── integration/                 # real PostgreSQL tests, skipped unless a database address is set
│   └── unit/                        # cost-free tests: test sanctions list, no LLM, no audit file
├── data/
│   ├── law_store/                   # official law texts, split by section
│   ├── sanctions/                   # sanctions lists (fictional test list committed)
│   └── sample_alerts/               # fictional alerts: the upstream-engine contract
├── scripts/
│   ├── build_law_store.py           # US regulations from eCFR / govinfo
│   ├── build_local_law_store.py     # FATF, UAE and CBUAE texts from local PDFs
│   ├── build_sanctions_lists.py     # UN, UAE Local Terrorist List, OFAC SDN
│   ├── run_sample_alerts.py         # runs the sample alerts, compares with expected
│   ├── fake_agent.py                # scripted stand-in for finintel's 3 agents (MCP client)
│   ├── check_doors.py               # checks that both running doors answer
│   └── check_restart.py             # proves a case is shared and survives a restart
├── migrations/                      # numbered table changes, run by Alembic
├── alembic.ini
├── .github/workflows/ci.yml         # runs the tests and builds the image on every push
├── .dockerignore                    # keeps secrets and private data out of the image
├── Dockerfile                       # one image for both doors (Python 3.12, CPU PyTorch, non-root)
├── docker-compose.yml               # starts the REST and MCP containers
├── requirements.txt                 # exact versions needed to run
└── requirements-dev.txt             # adds test and data-building tools
```

---

## Data Sources & Regulatory Basis

FinGuard checks against official sources, not text an agent supplies. Build scripts download or extract them into one format.

| Data | Source | Built by | On GitHub? |
|------|--------|----------|------------|
| US regulations (31 CFR 1010.100, .230, .311, .313, .314, .410, 1020.320; 31 U.S.C. 5324) | eCFR API, govinfo US Code | `scripts/build_law_store.py` | ✅ public domain |
| FATF Recommendations (R.1, 6, 10, 11, 12, 16, 19, 20, 24) | fatf-gafi.org PDF | `scripts/build_local_law_store.py` | ❌ copyrighted — kept local |
| UAE Federal Decree-Law No. 10 of 2025 (Articles 1–42) | official PDF | `scripts/build_local_law_store.py` | ❌ kept local |
| CBUAE Guidance on Transaction Monitoring and Sanctions Screening | CBUAE Rulebook PDF | `scripts/build_local_law_store.py` | ❌ kept local |
| UN Security Council Consolidated List | scsanctions.un.org | `scripts/build_sanctions_lists.py` | ❌ large, changes daily — rebuild |
| UAE Local Terrorist List | Executive Office (EOCN), uaeiec.gov.ae | `scripts/build_sanctions_lists.py` | ❌ rebuild |
| OFAC SDN List | sanctionslistservice.ofac.treas.gov | `scripts/build_sanctions_lists.py` | ❌ rebuild |
| Test sanctions list | **fictional** names, same format | hand-written | ✅ |
| FATF black and grey lists (19 June 2026) | fatf-gafi.org | in the policy YAML | ✅ |

Without the real sanctions file, FinGuard falls back to the fictional test list, logs a warning and marks every answer `"list_source": "test"`, so test screening is never mistaken for real screening.

**Requirement → feature** (from the sources above):

| Requirement | Source | FinGuard feature |
|-------------|--------|------------------|
| Screen against the UN Consolidated List and the UAE Local Terrorist List | CBUAE TM & Sanctions Screening Guidance, §1.3, §3.2 | Both lists built from the official files; screened from policy v1.2.0 |
| Name screening before onboarding and at least daily | CBUAE TM & Sanctions Screening Guidance, §3.4.1 | Rebuild script for the lists; daily scheduling planned |
| Use date of birth / nationality to tell a true match from a false match | CBUAE TM & Sanctions Screening Guidance, §3.4.1 | Identifier matching: confirmed / possible / ruled out, with a reason |
| Freeze without delay for designated persons | FATF Recommendation 6 | Confirmed match → `block_and_escalate` |
| Enhanced due diligence for countries the FATF calls for | FATF Recommendation 19 | Black list → review; grey list → higher risk score |
| Entities owned 50% or more by blocked persons are blocked too | OFAC Revised Guidance on Entity Property and Interests in Property (2014) | Tool 4 `owners`: ownership rule, threshold in the policy YAML |
| Identify beneficial owners of legal-entity customers | 31 CFR 1010.230, FATF Recommendation 10 | KYC/CDD tool (planned) |
| Report cash transactions over $10,000 | 31 CFR 1010.311 | Cash-only CTR rule in Tool 1. Checks one amount at a time: same-day totals (31 CFR 1010.313) are not added up |
| Suspicious activity reports must be accurate | 31 CFR 1020.320 | Tool 5: each claim checked against FinGuard's own copy of the cited law |

---

## Setup

FinGuard runs today with Docker or plain Python. Items marked *planned* are not implemented yet. A fresh clone starts on the fictional sanctions list and the public law store; the real list and licensed texts are built locally and never committed.

### 1. Clone & install

```bash
git clone https://github.com/Hanah7511/Finguard-LLM-Guardrails-Gateway
cd Finguard-LLM-Guardrails-Gateway
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scriptsctivate
pip install -r requirements.txt  # requirements-dev.txt adds the test and data-building tools
```

### 2. Environment variables (`.env`)

Copy `.env.example` to `.env` and fill it in. `.env` is never committed and never copied into the Docker image.

```env
# LLM
OPENAI_API_KEY=sk-...

# PostgreSQL and the encrypted vault. Docker Compose builds the database address from these.
POSTGRES_PASSWORD=...
APP_DB_PASSWORD=...
VAULT_MASTER_KEY=...   # 32 random bytes, base64. Losing it makes every stored real value unreadable.
FINGUARD_API_KEY=...   # the key apps send to the REST door as X-API-Key
ANALYST_KEY_PRIMARY=...   # one key per analyst. Never give this to the AI side

# Agent secret keys, one per agent. Make each with: python -c "import secrets; print(secrets.token_urlsafe(32))"
AGENT_KEY_INVESTIGATION=inv-secret-...
AGENT_KEY_COMPLIANCE=cmp-secret-...
AGENT_KEY_SUMMARIZATION=sum-secret-...

# Active policy version
ACTIVE_POLICY_VERSION=1.12.0
```

### 3. Run FastAPI (standalone)

```bash
uvicorn src.api.main:app --reload --port 8000

# Test
curl http://localhost:8000/health
curl -H "X-API-Key: <your key>" http://localhost:8000/policy
curl http://localhost:8000/metrics   # planned
```

### 4. Run MCP server (for AI agents)

```bash
python -m src.mcp_server.server
```

With the default `MCP_TRANSPORT=stdio` the server prints nothing and waits: an MCP client (an agent) starts it and talks to it over stdin/stdout. Set `MCP_TRANSPORT=streamable-http` to serve it on port 8001 instead.

```bash
# Tests for both doors (no LLM, fictional sanctions list, no audit file)
pip install -r requirements-dev.txt
python -m pytest test -v

# A scripted agent runs all 12 sample alerts through the MCP tools (real NLI model)
python scripts/fake_agent.py
python scripts/fake_agent.py --rogue ALT-2026-001   # agents misbehave on purpose
```

The fake agent plays finintel's investigation, compliance and summarization agents and obeys FinGuard's answers: 6 reports ready for review, 5 alerts closed as low risk, 1 sanctions hit escalated (ALT-2026-011: low risk, but a confirmed match, so parties are screened at every risk level). In `--rogue` mode FinGuard refuses the out-of-role tool call and holds the report with a false legal claim.

### 5. Docker

Docker Desktop must be running. Start one door at a time unless the machine has plenty of memory (see below). Starting a door also starts PostgreSQL and runs the table migration once:

```bash
docker compose up -d --build mcp   # one door; the first start also downloads the models
docker compose ps                  # wait for (healthy), which can take several minutes
docker compose stop mcp            # then `docker compose up -d api` to switch doors
docker compose stop                # stops everything; data and models stay
```

**Memory.** One door with its models uses about 2 GB. Both together did not fit in 3.7 GB of Docker memory: the second door crashed while loading. A machine running both needs more (about 6 GB is an estimate, not tested). `scripts/check_doors.py` and `scripts/check_restart.py` call both doors, so they need that too.

| What | How it is handled |
|------|-------------------|
| Secrets | Passed from `.env` when a container starts; `.dockerignore` keeps them out of the image |
| Real sanctions list, licensed law texts | Read-only folders mounted at run time, never stored in the image |
| Access | Ports open on `127.0.0.1` only. The REST door needs an app key, and tools that change a case need an agent key |
| Database | No outside port. The doors log in as a limited user that can only read and add rows |
| Audit log and case history | The database refuses edits and deletes |
| User | Runs as a non-root user |
| Models | Downloaded on first start, loaded when a door starts, kept in a Docker volume |

**Limits for now:**

- Logins are simple secret keys: no user accounts, no expiry and no rate limit. Add TLS and a proxy before opening these ports to a network.
- `/docs` stays open. It shows the shape of requests, but no data.
- An agent with a valid key can call a tool on any case.
- Analysts use simple keys: no second approver, no expiry, and no limit on how many times a report can be released.
- Policies before v1.8.0 still return real values from Tool 8. Do not switch back to one.
- Real values travel over plain HTTP until you add TLS, so they must stay on `127.0.0.1`.
- The master key sits in `.env`: anyone with both that file and the database can read the values. A real deployment needs a secrets manager or key service.
- A Docker volume is not a backup.
- Known issue: `docker compose down` once failed with a "zombie" error while stopping the doors (1 of 6 shutdowns, cause unknown). Data was not affected. Running it again finished the job.

---

## Metrics & Evaluation

### Measured results (policy v1.12.0, test sets)

Run with `python evals/run_evals.py`. Results are saved in `evals/results/latest.json`. Tuning used the dev sets only.

| Check | Result | Notes |
|---|---|---|
| Attacks caught, AML test set | **0 of 6** | Phrase rules only, no trained classifier yet |
| Attacks caught, deepset | **0 of 60** | Includes German text; the rules are in English. 0 of 56 normal prompts blocked |
| Attacks caught, Gandalf | **19 of 112 (17%)** | |
| Names hidden, 8 groups including Arabic script | 106 of 106 | Small hand-written set |
| Ordinary sentences with a word masked as a name | 1 of 20 | |
| Personal data hidden | 27 of 28 | Missed: an IBAN split across lines |
| Normal sentences with a number wrongly masked | 2 of 12 | Hiding too much is the safer mistake |
| Sanctions name variations caught | 17 of 22 (77%) | Missed: missing name parts (0 of 3), initials (1 of 2), Arabic typo (0 of 1) |
| Innocent names wrongly flagged | 1 of 21 | Small fictional list |
| True legal claims accepted | 8 of 9 | |
| False legal claims refused | 17 of 18 (94%) | |
| Masking speed, p50 / p95 | 383 / 616 ms | Rules only: 6 / 7 ms. Laptop CPU, varies with load |

**Read these with care.** The sets are small, so one case moves a score a lot. Attack detection is the weakest part. Not measured yet: false alarms on the real sanctions lists, confidence calibration, escalation rate and tool-call accuracy.

### Level 1 — Per-layer metrics

| Metric | Layer | Target |
|--------|-------|--------|
| Catch rate | Injection, PII, Safety, NLI | As high as possible (measure per layer) |
| False block rate | Injection, Safety, NLI | As low as possible |
| PII entity recall | Presidio vs regex (compare both) | As high as possible |
| PII entity precision | Presidio vs regex | As high as possible |
| NLI score separation | NLI grounding | Correct vs hallucinated groups must not overlap |
| Latency p50 / p95 | Every layer separately | Measure and report — not averages |

### Level 2 — System metrics

| Metric | Applied to | Notes |
|--------|-----------|-------|
| Tool call accuracy | `validate_tool_call()` | `correct_decisions / total` |
| Escalation rate | Full system | Target 5–15% for AML (too low = under-checking; too high = analyst overload) |
| Confidence calibration | Confidence gate | Does GPT's self-reported confidence match actual accuracy? |
| Schema retry rate + cost/request | LLM call | Each retry costs money |
| Sanctions match recall | `sanctions_screen()` | Name variation testing: Mohammed / Muhammad / Mohamed |
| Task success rate | Full MCP flow | Scripted fake-agent now; real finintel later |
| Policy version coverage | All decisions | Must be 100% — automatic test, not a score |
| Policy v1.0.0 → v1.12.0 | Full eval suite | Run on every version; compare results |
| End-to-end agent run | `scripts/fake_agent.py` (12 alerts, `--rogue`) | Task success: every investigation reaches the expected outcome; rogue calls refused |
| Sample alert regression | `data/sample_alerts/alerts.json` | `python scripts/run_sample_alerts.py` — must stay 12/12 unless a policy change explains the difference |

> ⚠️ **Confidence calibration is the biggest gap to close.** The confidence gate trusts the number GPT reports. If GPT says 0.9 but is only right 70% of the time at that score, the 0.85 gate means nothing. Measure with a calibration curve before reporting any threshold as meaningful.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API layer | FastAPI + Pydantic v2 | Type-safe schemas, auto-generated docs, async |
| MCP server | MCP Python SDK | Standard agent tool protocol |
| PII detection | Microsoft Presidio + regex fallback | Named-entity PII with custom recognizer support |
| NLI grounding | `cross-encoder/nli-deberta-v3-small` | State-of-the-art NLI without GPT dependency |
| Safety classification | Regex + Llama Guard 1B (optional) | Fast layer-1, nuanced layer-2 |
| Vector store | FAISS (optional, NumPy fallback) | Fast similarity search, local deployment |
| Storage | PostgreSQL + SQLAlchemy + Alembic (JSONL in memory mode) | Queryable, persistent, shared by both doors |
| Encryption | AES-256-GCM (`cryptography`) | Stolen tables alone reveal no values, and tampering is detected |
| LLM | OpenAI GPT-4o (configurable) | Swappable via the `LLMClient` protocol in `llm/client.py` |

### Relation to other projects

| Project | Role | Connection to FinGuard |
|---------|------|------------------------|
| `finrisk_detection_engine` | AML detection (ML) | Sends alerts via FastAPI `/validate` or MCP Tool 1. No import dependency. |
| `finintel_sentinel_ai` | LangGraph agents + SAR | Calls FinGuard's 8 MCP tools. Wrappers enforce validate→run→pii order. No import dependency. |
| **FinGuard** | Guardrail middleware | Standalone. Works without either project. Any project connects via YAML config. |

---

## Status

**In active development.** Part of the Financial Risk Intelligence Platform portfolio.

- [x] Architecture + design (dual-door, 6 guardrail layers, 8 MCP tools)
- [x] Versioned YAML policy engine (v1.0.0 – v1.11.0, v1.12.0 active)
- [x] Guardrail modules for all 6 layers (input, tool-call, output, safety, privacy, compliance)
- [x] Security model designed (5 properties) — implementation in progress
- [x] `orchestrator.py` — single brain for both doors (one `handle_*` per job; risk score = higher of FinGuard and upstream engine score)
- [x] Law store from official sources (US, FATF, UAE, CBUAE) with citation lookup and NLI grounding
- [x] Official sanctions lists (UN, UAE Local, OFAC) with fast fuzzy screening
- [x] Identifier matching (date of birth, ID number) and FATF black/grey-list country risk
- [x] Sample alert set (12 fictional scenarios) with an expected-result runner
- [x] FastAPI door (`/health`, `/policy`, `/validate`, `/chat`, `/assess-risk`, `/sanctions-screen`, `/cases/{case_id}/release`) with tests
- [x] MCP server: Tools 1–8 over stdio or streamable HTTP, with 12 tests (in-memory MCP client, fake NLI)
- [x] Fake agent: scripted stand-in for finintel runs all 12 alerts end to end over MCP, with a `--rogue` mode (5 tests)
- [x] Per-agent secret keys for Tool 2 (10 tests)
- [x] Cash-only Currency Transaction Report flag in Tool 1 (7 tests)
- [x] Reversible masking and Tool 8 `finguard_finalize_check`
- [x] Pinned requirements, Docker image and Compose for both doors, GitHub Actions CI
- [x] Tests + evals
- [x] PostgreSQL: encrypted vault, case history and append-only audit log, shared by both doors and surviving a restart
- [x] Name masking (policy v1.6.0): party list, Presidio, Tool 4 tokens, Tool 8 report re-check
- [x] Access control (policy v1.7.0): REST app key, agent keys on every tool that changes a case, required `case_id` in a set format
- [x] Analyst release (policy v1.8.0): REST only, one key per analyst, the exact approved report, checked again at release
- [x] GLiNER finds names and addresses (policy v1.10.0)
- [x] Arabic-script sanctions screening (policy v1.11.0)
- [x] Exact check of amounts, periods and dates in legal claims (policy v1.12.0)
- [x] Measure name detection across names of different origins (Step 8)
- [ ] `/metrics` endpoint

**Planned extras (after the core):**

- [ ] Tool 9 KYC/CDD check (records whether identifiers were verified, beneficial owners, company age)
- [x] OFAC 50% ownership rule (caller sends owners and %; sanctioned shares are added up; threshold per jurisdiction in YAML)
- [ ] PEP screening (result `edd_required`, not a block; real PEP data stays local, fictional test list committed)
- [x] Unverified identifiers can't clear an exact name match (stops a faked date of birth clearing a real hit)
- [ ] SAR fact grounding against case evidence
- [ ] Sanctions transliteration / phonetic matching
- [ ] One policy file per jurisdiction (UAE, UK, EU)
- [ ] `docs/SECURITY.md` (OWASP LLM Top 10 mapping)

---

## Author

Built by [Hana Al Haris](https://github.com/Hanah7511) · [Portfolio](https://hanaalharis.vercel.app) · [LinkedIn](https://linkedin.com/in/hana-al-haris-3b7528255)
