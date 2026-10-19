# Guardrail Policy Changelog

Every policy change gets a new versioned file, an entry here and a git commit.
Old versions stay in place: reports record the version they were validated under.

## 1.1.0 — 2026-06-25 (active)
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
