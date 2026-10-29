"""Run the sample alerts through FinGuard and compare each answer with the expected one.

Every alert goes through Tool 1 (assess_risk) and, if it names a party, Tool 4
(sanctions_screen). Screening always uses the FICTIONAL test list, so the results are
the same on every machine. Decisions are not written to the audit log.

Usage:  python scripts/run_sample_alerts.py           compare with "expected"
        python scripts/run_sample_alerts.py --update  rewrite "expected" from this run
                                                      (only after checking every change)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.policy_loader import get_policy  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.core.orchestrator import Orchestrator  # noqa: E402
from src.core.sanctions_screener import SanctionsScreener  # noqa: E402
from src.monitoring.decision_logger import DecisionLogger  # noqa: E402

ALERTS_PATH = ROOT / "data" / "sample_alerts" / "alerts.json"


def run_alert(orch: Orchestrator, alert: dict) -> dict:
    risk = orch.handle_assess_risk(
        alert["query"], alert["signals"], alert.get("external_risk_score"), alert["case_id"]
    )
    result = {
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "own_score": risk.get("own_score"),
        "scores_disagree": risk.get("scores_disagree"),
        "ignored_signals": sorted(risk.get("ignored_signals", [])),
    }
    party = alert.get("party")
    if party:
        screen = orch.handle_sanctions_screen(
            party["name"],
            country=alert["signals"].get("country"),
            case_id=alert["case_id"],
            dob=party.get("dob"),
            nationality=party.get("nationality"),
            id_number=party.get("id_number"),
            identifiers_verified=party.get("identifiers_verified", False),
        )
        result["sanctions"] = screen["recommendation"]
        result["country_risk"] = screen["country_risk"]
    return result


def main() -> int:
    update = "--update" in sys.argv
    data = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    policy = get_policy()
    if policy.version != data["expected_policy_version"] and not update:
        print(f"Active policy is {policy.version}, but 'expected' was written for "
              f"{data['expected_policy_version']}. Differences are expected.")

    test_list = SanctionsScreener.from_policy(settings.sanctions_test_list_path, policy.sanctions_screening, "test")
    orch = Orchestrator(policy, settings, sanctions=test_list, audit=DecisionLogger(None))

    failures = 0
    for alert in data["alerts"]:
        got = run_alert(orch, alert)
        if update:
            alert["expected"] = got
            continue
        diffs = {k: (v, got.get(k)) for k, v in alert["expected"].items() if got.get(k) != v}
        failures += bool(diffs)
        sanctions = f" | sanctions {got['sanctions']}" if "sanctions" in got else ""
        print(f"{'FAIL' if diffs else 'PASS'}  {alert['alert_id']}  {alert['typology']:<40} "
              f"{got['risk_level']:<6} {got['risk_score']:.2f}{sanctions}")
        for key, (want, have) in diffs.items():
            print(f"        {key}: expected {want}, got {have}")

    if update:
        data["expected_policy_version"] = policy.version
        ALERTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Rewrote 'expected' for {len(data['alerts'])} alerts under policy {policy.version}.")
        return 0
    print(f"\n{len(data['alerts']) - failures}/{len(data['alerts'])} alerts as expected "
          f"(policy {policy.version}).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())