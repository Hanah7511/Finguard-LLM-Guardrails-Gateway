# Guardrail Policy Changelog

Every policy change gets a new versioned file, an entry here and a git commit.
Old versions stay in place: reports record the version they were validated under.

## 1.2.0 — 2026-10-23 (active)
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
