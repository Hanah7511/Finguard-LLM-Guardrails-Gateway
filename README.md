# FinGuard LLM Guardrails Gateway

> **LLMOps · Safety Infrastructure**  
> A composable, policy-driven safety middleware for LLM applications in financial services.  
> Two entry points — FastAPI for humans, MCP for AI agents — one set of guardrail rules.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![MCP](https://img.shields.io/badge/MCP_Server-8_Tools-blue?style=flat)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_Audit-Planned-336791?style=flat&logo=postgresql&logoColor=white)
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

> **Reversible PII masking (planned).** Real values are stored in FinGuard under a token (`[US_SSN_1]`). AI components only ever see the token. A non-AI program restores real values after human approval — required for SAR filing with FinCEN.

### Layer 2 — Tool-Call Guardrails (OWASP LLM06)
Every agent tool call passes through an allowlist check before execution, and every tool result is PII-scanned on the way out. Agent identity will be derived from a **per-agent secret key** (planned) — not the agent name, which can be forged.

### Layer 3 — Output Guardrails (OWASP LLM09)
LLM responses validated in three stages:
1. Pydantic schema validation
2. NLI-based grounding check — agent sends claim + citation only; FinGuard fetches authoritative law text from its own trusted store
3. Confidence gate — routes uncertain outputs to human review

### Layer 4 — Safety Guardrails
Two-layer pipeline: fast regex pattern matching (Layer 1) + optional Llama Guard 1B for nuanced harm classification (Layer 2). Each runs independently.

### Layer 5 — Privacy Guardrails
PII masked before it reaches the LLM, and scrubbed again from all audit logs. Consent flag enforced per request.

### Layer 6 — Compliance Guardrails
NLI grounding against official law texts (US BSA regulations; FATF Recommendations, UAE Federal Decree-Law No. 10 of 2025 and CBUAE guidance kept locally, see [Data Sources](#data-sources--regulatory-basis)). Sanctions screening against the UN, UAE Local Terrorist List and OFAC SDN lists, with FATF black- and grey-list country risk. Versioned YAML policies with `approved_by` and `change_reason`. Append-only, PII-scrubbed audit trail (JSONL today, PostgreSQL planned) with policy version on every decision record.

---

## MCP Tools (8)

*Tools 1–7 are live in `src/mcp_server/server.py`; Tool 8 is planned (Step 6).* All tools take a `case_id` to link calls into one audit trail per case. Each tool's description is written for the agent: when to call it and what to do with the answer. Agent authentication via secret key (not self-reported name) is planned; until then Tool 2 trusts the agent name it is given.

| # | Tool | Description |
|---|------|-------------|
| 01 | `finguard_assess_risk()` | Risk triage before any investigation starts. Scores alert signals against YAML weights. Returns: `LOW / MEDIUM / HIGH` |
| 02 | `finguard_validate_tool_call()` | Authorizes or blocks an agent's tool request. Checks agent identity (from secret key) against YAML allowlist. No LLM call — fast. |
| 03 | `finguard_pii_and_compliance_check()` | PII detection and reversible masking. Uses Presidio + regex fallback. Real values stored in FinGuard, not returned. |
| 04 | `finguard_sanctions_screen()` | Screens a person or company against the UN, UAE Local Terrorist List and OFAC SDN lists (exact + fuzzy names, aliases). Date of birth / ID number confirm or rule out a name match; a name alone goes to review. Country risk from the FATF black and grey lists. `transaction_type` check for CTR applicability (cash-only rule, planned). |
| 05 | `finguard_validate_grounding()` | NLI hallucination check. Agent sends claim + citation reference only — FinGuard fetches the authoritative law text. Agent cannot supply a fake source. |
| 06 | `finguard_get_policy_info()` | Returns active policy version, last change, `approved_by`, and applicable rules. Also available via `GET /policy`. |
| 07 | `finguard_scan_documents()` | Scans RAG-retrieved documents for indirect prompt injection — hidden instructions embedded in regulatory text or case files. |
| 08 | `finguard_finalize_check()` | *(Planned)* Final gate before human review. Reads audit trail for the `case_id` and verifies all required steps completed. BLOCKs if any step is missing. |

---

## Orchestrator — handle_* Pattern

The orchestrator loads all workers once at startup (PII masker, NLI checker, sanctions screener, law store, pipelines, audit log). Both doors share the same orchestrator — no logic is duplicated.

| Function | Door | Workers used |
|----------|------|--------------|
| `handle_chat()` | API `/chat` | All: input → law lookup → GPT → output guards |
| `handle_validate()` | API `/validate` | Input guards only — no GPT, no cost |
| `handle_assess_risk()` | MCP Tool 1 | Injection + safety scan → risk assessor |
| `handle_validate_tool_call()` | MCP Tool 2 | Allowlist → parameter check |
| `handle_pii_check()` | MCP Tool 3 | PII masker → hidden instruction scanner |
| `handle_sanctions_screen()` | MCP Tool 4 | Sanctions screener only |
| `handle_validate_grounding()` | MCP Tool 5 | Law store + NLI checker, per claim |
| `handle_get_policy_info()` | MCP Tool 6 + `/policy` | YAML rulebook reader |
| `handle_scan_documents()` | MCP Tool 7 | Hidden instruction scanner |
| `handle_finalize_check()` | MCP Tool 8 | Audit log reader — were all steps done? *(planned)* |

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
   └─ [wrapper] Tool 3: pii_check → SSN → [US_SSN_1], name → [PERSON_1]

4. Compliance Agent (wrapped calls)
   ├─ [wrapper] Tool 2: validate_tool_call → sanctions_check APPROVED
   ├─ Tool 4: sanctions_screen (real name from finintel DB)
   └─ Tool 7: scan_documents (RAG results scanned for injection)

5. Summarization Agent → SAR draft
   └─ Tool 5: validate_grounding(claims_list, citations)
      └─ FinGuard fetches real law text → NLI check → GROUNDED or BLOCKED

6. Tool 8: finguard_finalize_check(case_id)
   └─ ✅ risk assessed · ✅ tools validated · ✅ sanctions screened
      ✅ claims grounded · ✅ no unmasked PII · ✅ policy version attached

7. Human analyst approves → non-AI program restores real values
   └─ [US_SSN_1] → 123-45-6789 · SAR filed with FinCEN
```

---

## Security Model

Five properties enforced by construction — not by agent cooperation:

| Fix | Property |
|-----|----------|
| **FIX 1** | **Agent cannot skip guardrails.** In finintel, every tool is wrapped: `validate_tool_call()` → run tool → `pii_check()`. The agent calls the wrapper, never the tool directly. |
| **FIX 2** | **All calls linked by `case_id`.** Every tool call carries a `case_id`. Tool 8 reads the full audit trail before approving finalization. |
| **FIX 3** | **Agent identity from secret key, not self-report.** A compromised agent cannot impersonate a higher-privileged one by changing its `agent_name` field. |
| **FIX 4** | **Agent cannot supply its own grounding source.** Tool 5 accepts claim + citation reference only. FinGuard fetches the law text — a hallucinating agent cannot provide a fake source that confirms its own false claim. |
| **FIX 5** | **Reversible masking for regulatory compliance.** SARs must contain real SSN/TIN. FinGuard masks values as tokens for all AI steps, a non-AI program restores them after human approval. Both properties hold simultaneously. |

---

## Folder Structure

```
finguard-llm-guardrails-gateway/
├── src/
│   ├── config/
│   │   └── policies/
│   │       ├── v1.0.0_guardrail_policy.yaml   # versioned policy (approved_by, change_reason)
│   │       ├── v1.1.0_guardrail_policy.yaml
│   │       ├── v1.2.0_guardrail_policy.yaml   # active
│   │       └── CHANGELOG.md
│   ├── core/
│   │   ├── orchestrator.py          # THE BRAIN — one handle_* per job
│   │   ├── guardrail_pipeline.py
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
│   │   └── server.py                # 7 tools (Tool 8 planned), one per handle_*
│   ├── api/                         # FastAPI door
│   │   ├── main.py                  # 6 endpoints, one per handle_*
│   │   └── schemas.py               # request shapes (422 on bad input)
│   └── monitoring/
│       ├── telemetry.py
│       ├── decision_logger.py
│       └── cost_tracker.py
├── test/
│   └── unit/                        # cost-free tests: test sanctions list, no LLM, no audit file
├── data/
│   ├── law_store/                   # official law texts, split by section
│   ├── sanctions/                   # sanctions lists (fictional test list committed)
│   └── sample_alerts/               # fictional alerts: the upstream-engine contract
├── scripts/
│   ├── build_law_store.py           # US regulations from eCFR / govinfo
│   ├── build_local_law_store.py     # FATF, UAE and CBUAE texts from local PDFs
│   ├── build_sanctions_lists.py     # UN, UAE Local Terrorist List, OFAC SDN
│   └── run_sample_alerts.py         # runs the sample alerts, compares with expected
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
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
| Screen against the UN Consolidated List and the UAE Local Terrorist List | CBUAE TM & Sanctions Screening Guidance, §1.3, §3.2 | Both lists built from the official files; screened under policy v1.2.0 |
| Name screening before onboarding and at least daily | CBUAE TM & Sanctions Screening Guidance, §3.4.1 | Rebuild script for the lists; daily scheduling planned |
| Use date of birth / nationality to tell a true match from a false match | CBUAE TM & Sanctions Screening Guidance, §3.4.1 | Identifier matching: confirmed / possible / ruled out, with a reason |
| Freeze without delay for designated persons | FATF Recommendation 6 | Confirmed match → `block_and_escalate` |
| Enhanced due diligence for countries the FATF calls for | FATF Recommendation 19 | Black list → review; grey list → higher risk score |
| Identify beneficial owners of legal-entity customers | 31 CFR 1010.230, FATF Recommendation 10 | KYC/CDD tool (planned) |
| Report cash transactions over $10,000 | 31 CFR 1010.311 | Cash-only CTR rule (planned) |
| Suspicious activity reports must be accurate | 31 CFR 1020.320 | Tool 5: each claim checked against FinGuard's own copy of the cited law |

---

## Setup

> ⚠️ **Not runnable yet.** These steps describe the target setup and will work once the API, MCP server and `requirements.txt` are complete. Items marked *planned* are not implemented yet.

### 1. Clone & install

```bash
git clone https://github.com/Hanah7511/Finguard-LLM-Guardrails-Gateway
cd Finguard-LLM-Guardrails-Gateway
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables (`.env`)

```env
# LLM
OPENAI_API_KEY=sk-...

# PostgreSQL audit trail (planned)
DATABASE_URL=postgresql://user:pass@localhost:5432/finguard_audit

# Agent secret keys, one per agent (planned)
AGENT_KEY_INVESTIGATION=inv-secret-...
AGENT_KEY_COMPLIANCE=cmp-secret-...
AGENT_KEY_SUMMARIZATION=sum-secret-...

# Active policy version
ACTIVE_POLICY_VERSION=1.2.0
```

### 3. Run FastAPI (standalone)

```bash
uvicorn src.api.main:app --reload --port 8000

# Test
curl http://localhost:8000/health
curl http://localhost:8000/metrics   # planned
```

### 4. Run MCP server (for AI agents)

```bash
python -m src.mcp_server.server
```

With the default `MCP_TRANSPORT=stdio` the server prints nothing and waits: an MCP client (an agent) starts it and talks to it over stdin/stdout. Set `MCP_TRANSPORT=streamable-http` to serve it on port 8001 instead.

```bash
# Tests for both doors (no LLM, fictional sanctions list, no audit file)
python -m pytest test -v
```

### 5. Docker

```bash
docker-compose up --build
```

```yaml
# docker-compose.yml
services:
  finguard:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: finguard_audit
      POSTGRES_USER: finguard
      POSTGRES_PASSWORD: your-password
```

---

## Metrics & Evaluation

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
| Policy v1.0.0 → v1.2.0 | Full eval suite | Run on every version; compare results |
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
| Audit trail | JSONL today → PostgreSQL (planned) | Queryable, persistent, regulator-ready |
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
- [x] Versioned YAML policy engine (v1.0.0, v1.1.0, v1.2.0 active)
- [x] Guardrail modules for all 6 layers (input, tool-call, output, safety, privacy, compliance)
- [x] Security model designed (5 properties) — implementation in progress
- [x] `orchestrator.py` — single brain for both doors (one `handle_*` per job; risk score = higher of FinGuard and upstream engine score)
- [x] Law store from official sources (US, FATF, UAE, CBUAE) with citation lookup and NLI grounding
- [x] Official sanctions lists (UN, UAE Local, OFAC) with fast fuzzy screening
- [x] Identifier matching (date of birth, ID number) and FATF black/grey-list country risk
- [x] Sample alert set (12 fictional scenarios) with an expected-result runner
- [x] FastAPI door (`/health`, `/policy`, `/validate`, `/chat`, `/assess-risk`, `/sanctions-screen`) with 11 tests
- [x] MCP server: Tools 1–7 over stdio or streamable HTTP, with 9 tests (in-memory MCP client, fake NLI)
- [ ] Tool 8 `finguard_finalize_check` + per-agent secret keys
- [ ] Tests + evals
- [ ] PostgreSQL audit trail
- [ ] `/metrics` endpoint

---

## Author

Built by [Hana Al Haris](https://github.com/Hanah7511) · [Portfolio](https://hanaalharis.vercel.app) · [LinkedIn](https://linkedin.com/in/hana-al-haris-3b7528255)
