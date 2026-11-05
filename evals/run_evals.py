"""How good is FinGuard? Prints scores and saves them to evals/results/latest.json.

Run:  python evals/run_evals.py

Misses are printed for the dev set only. The test and public sets show counts, so nobody
tunes the rules against them, which would make their scores look better than they really are.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.scoring import NameScore, name_is_hidden, percent, score_attacks  # noqa: E402
from src.config.policy_loader import get_policy  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.input_guardrails.pii.presidio_detector import build_pii_detector  # noqa: E402
from src.input_guardrails.pipeline import InjectionDetector, PatternSafetyClassifier  # noqa: E402

DATASETS = ROOT / "evals" / "datasets"
CACHE = ROOT / "evals" / "cache"  # public data: downloaded once, never committed
RESULTS = ROOT / "evals" / "results" / "latest.json"
# Public test splits. Their train splits stay unused here, free for training a model later.
DEEPSET = "deepset/prompt-injections"  # Apache 2.0. Label 1 is an attack, 0 is normal text.
GANDALF = "Lakera/gandalf_ignore_instructions"  # MIT. Attacks only, so it has no false block rate.
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def attack_checker():
    """The same two pattern checks Tool 1 runs on a query."""
    policy = get_policy()
    cfg = policy.input_guardrails
    checks = (
        InjectionDetector(cfg.injection_patterns, policy.version),
        PatternSafetyClassifier(cfg.safety_patterns, policy.version),
    )
    return lambda text: any(check.check(text).stops_pipeline for check in checks)


def public_rows(dataset: str) -> list[dict]:
    """A public test split, downloaded once. Raises OSError if it cannot be fetched."""
    path = CACHE / f"{dataset.replace('/', '_')}_test.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": "test", "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=30) as response:
            page = json.load(response)
        rows += [r["row"] for r in page["rows"]]
        offset += len(page["rows"])
        if offset >= page["num_rows_total"] or not page["rows"]:
            break
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def run_attack_eval() -> dict:
    blocked = attack_checker()
    sets = json.loads((DATASETS / "attacks.json").read_text(encoding="utf-8"))
    results = {}
    for name, examples in sets.items():
        decisions = [blocked(e["text"]) for e in examples]
        results[f"AML {name}"] = score_attacks(examples, decisions).as_row()
        if name == "dev":  # never for the test set
            for example, decision in zip(examples, decisions):
                if decision != (example["label"] == "attack"):
                    print(f"  dev miss ({example['label']}, {example['kind']}): {example['text']}")
    for name, dataset, to_label in (
        ("public, deepset", DEEPSET, lambda r: "attack" if r["label"] == 1 else "normal"),
        ("public, Gandalf", GANDALF, lambda r: "attack"),
    ):
        try:
            rows = public_rows(dataset)
        except OSError as exc:
            print(f"  {name} skipped: could not download it ({type(exc).__name__})")
            continue
        labelled = [{"label": to_label(r), "text": r["text"]} for r in rows]
        results[name] = score_attacks(labelled, [blocked(e["text"]) for e in labelled]).as_row()
    return results


def run_names_eval() -> dict:
    """Does masking hide names across origins, and does it leave ordinary words alone?"""
    policy = get_policy()
    data = json.loads((DATASETS / "names.json").read_text(encoding="utf-8"))
    detector = build_pii_detector(policy.input_guardrails, "presidio", get_settings().spacy_model)
    templates = data["templates"]
    table = {}
    for group, splits in data["groups"].items():
        row = {}
        for split, names in splits.items():
            hidden = 0
            for i, name in enumerate(names):
                sentence = templates[i % len(templates)].format(name=name)
                masked = detector.mask(sentence).masked_text
                ok = name_is_hidden(name, masked)
                hidden += ok
                if split == "dev" and not ok:  # never for the test set
                    print(f"  dev miss ({group}): {masked}")
            row[split] = NameScore(len(names), hidden).as_row()
        table[group] = row
    wrong = {}
    for split, sentences in data["normal"].items():
        flagged = [s for s in sentences if any(e.entity_type == "PERSON" for e in detector.detect(s))]
        if split == "dev":
            for sentence in flagged:
                print(f"  dev false match: {detector.mask(sentence).masked_text}")
        wrong[split] = {"wrongly_masked": f"{len(flagged)} of {len(sentences)}", "rate": percent(len(flagged) / len(sentences))}
    return {"groups": table, "normal_text": wrong}


def print_names(names: dict) -> None:
    print("\nNames: is every word of the name hidden?")
    print(f"  {'group':<30}{'dev':<14}{'test':<14}test recall")
    for group, row in names["groups"].items():
        print(f"  {group:<30}{row['dev']['names_hidden']:<14}{row['test']['names_hidden']:<14}{row['test']['recall']}")
    print("\nNormal text: sentences where an ordinary word was masked as a name")
    for split, row in names["normal_text"].items():
        print(f"  {split:<30}{row['wrongly_masked']:<14}{row['rate']}")


def print_table(title: str, rows: dict) -> None:
    print(f"\n{title}")
    print(f"  {'set':<22}{'attacks caught':<18}{'catch rate':<13}{'normal blocked':<18}false block rate")
    for name, r in rows.items():
        print(f"  {name:<22}{r['attacks_caught']:<18}{r['catch_rate']:<13}{r['normal_blocked']:<18}{r['false_block_rate']}")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")  # Arabic text must not crash a Windows terminal
    print(f"Policy {get_policy().version}")
    print("\nAttacks: what the dev set got wrong")
    attacks = run_attack_eval()
    print_table("Attacks", attacks)
    print("  Note: deepset includes German text, and FinGuard's rules are in English. Gandalf has attacks only.")
    print("\nNames: what the dev set got wrong (loading the name model can take a minute)")
    names = run_names_eval()
    print_names(names)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = {"policy": get_policy().version, "attacks": attacks, "names": names}
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
