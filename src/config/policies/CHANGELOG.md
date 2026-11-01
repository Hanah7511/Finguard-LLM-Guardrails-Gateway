# Guardrail Policy Changelog

Every policy change gets a new versioned file, an entry here and a git commit.
Old versions stay in place: reports record the version they were validated under.

## 1.5.0 — 2026-11-01 (active)
Approved by: compliance-team

- `tool_call_guardrails.reversible_masking: true`. With a `case_id`, Tool 3 masks values as per-case tokens
  (`[IBAN_CODE_1]`). The same value gets the same token within a case. Real values are held in memory only
  and never logged. Without a `case_id`, values are masked for good, as before.
- `finalize_check`: new Tool 8 `finguard_finalize_check`, the last gate before human review. It needs the
  agent's key, an agent in `allowed_agents`, and every `required_steps` item met on FinGuard's own record of
  the case. Approved: the report comes back with real values restored. Refused: `missing` says why and the
  report stays masked. A report using a token the case never issued is refused too.

## 1.4.0 — 2026-10-31
Approved by: compliance-team

- `tool_call_guardrails.require_agent_keys: true`. Tool 2 refuses a call unless it carries the secret key
  of the agent it names (`AGENT_KEY_<AGENT>` in `.env`, e.g. `AGENT_KEY_INVESTIGATION`). FinGuard works
  out the caller from the key, so an agent can no longer claim another agent's name to get its tools.
  Keys are compared in constant time and never logged; the audit record holds `verified_agent`.
- `risk_assessment.ctr_rule`: Tool 1 takes `transaction_type` and returns `ctr`. A Currency Transaction
  Report is required only for cash of more than 10,000 (31 CFR 1010.311); a large wire is not a CTR case.
  The CTR flag does not change the risk score. Same-day cash totals (31 CFR 1010.313) are not added up.

## 1.3.0 — 2026-10-29
Approved by: compliance-team

- `sanctions_screening.ownership_rule` (OFAC 50% rule): a company owned 50% or more, in total, by
  blocked persons is blocked even if its own name is on no list. The caller sends the owners
  (Tool 4 `owners`); a blocked company counts in full in the companies it owns. A listed owner
  below the threshold sends the case to review.
- `sanctions_screening.clear_requires_verified_identifiers: true`. A different date of birth / ID
  clears an exact name match only when the caller says it was verified (`identifiers_verified`);
  otherwise the match stays `possible` and goes to review. Fuzzy matches are still cleared.
- `pii_entities`: added `EMAIL_ADDRESS`. Emails are also scrubbed from the audit log.
- FATF lists unchanged: checked 29 October 2026, the October 2026 plenary outcomes are not yet published.

## 1.2.0 — 2026-10-23
Approved by: compliance-team

- `sanctions_screening.lists`: `["OFAC", "UN", "EU"]` → `["OFAC", "UN", "UAE_LOCAL"]`. The UAE Local
  Terrorist List was loaded but never screened; no EU list is loaded, so listing it overstated coverage.
- `sanctions_screening.use_identifiers: true`. A name match is confirmed by ID number or full date of
  birth and ruled out by a different date of birth; a name alone now goes to review instead of a block.
- Added the FATF grey list (`increased_monitoring_jurisdictions`, 19 June 2026 statement): country risk
  `ELEVATED` and `risk_assessment.monitored_country_weight` 0.10. Unlike the black list, it does not send
  a case to review on its own.
- Added `"dprk"` to `high_risk_jurisdictions`.

## 1.1.0 — 2026-06-25  
Approved by: compliance-team

- `output_guardrails.minimum_confidence` raised 0.80 → 0.85 after the Q2 false-negative review.
- `safety_patterns`: replaced the bare `"launder"` with intent phrases (`"help me launder"`,
  `"how to launder"`, ...). The bare word blocked analysts asking legitimate questions
  about laundering typologies.
- Added injection patterns (`"reveal your system prompt"`, `"developer mode"`, ...) and
  `indirect_injection_patterns` for retrieved documents and tool output.
- Added `parameter_rules` for `transaction_lookup` and `customer_history`.
- Added risk signals `rapid_movement_flag`, `pep_flag` and `dormant_account_flag`.
- Added FATF high-risk jurisdictions to sanctions screening.

## 1.0.0 — 2026-03-01
Approved by: compliance-team

- Initial policy.
