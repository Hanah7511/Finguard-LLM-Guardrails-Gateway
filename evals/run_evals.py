"""How good is FinGuard? Prints scores and saves them to evals/results/latest.json.

Run:  python evals/run_evals.py

Misses are printed for the dev set only. The test and public sets show counts, so nobody
tunes the rules against them, which would make their scores look better than they really are.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.scoring import NameScore, name_is_hidden, percent, score_attacks, score_by_kind, value_is_hidden  # noqa: E402
from src.config.policy_loader import get_policy, load_policy  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.core.sanctions_screener import SanctionsScreener  # noqa: E402
from src.input_guardrails.pii.presidio_detector import build_pii_detector  # noqa: E402
from src.input_guardrails.attack_classifier import AttackClassifier  # noqa: E402
from src.input_guardrails.pipeline import InjectionDetector, PatternSafetyClassifier  # noqa: E402
from src.input_guardrails.pii.regex_detector import RegexPIIDetector  # noqa: E402
from src.output_guardrails.grounding.base import GroundingVerdict  # noqa: E402
from src.output_guardrails.grounding.nli_grounding import NLIGrounder  # noqa: E402
from src.output_guardrails.pipeline import OutputGuardrailPipeline  # noqa: E402
from src.vector_store.retriever import RegulationRetriever, SentenceTransformerEmbedder  # noqa: E402

DATASETS = ROOT / "evals" / "datasets"
CACHE = ROOT / "evals" / "cache"  # public data: downloaded once, never committed
RESULTS = ROOT / "evals" / "results" / "latest.json"
# Public test splits. Their train splits train the attack classifier (scripts/train_attack_classifier.py).
DEEPSET = "deepset/prompt-injections"  # Apache 2.0. Label 1 is an attack, 0 is normal text.
GANDALF = "Lakera/gandalf_ignore_instructions"  # MIT. Attacks only, so it has no false block rate.
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def build_attack_classifier():
    """The trained classifier, or None if the active policy has it switched off."""
    policy, settings = get_policy(), get_settings()
    cfg = policy.input_guardrails.attack_classifier
    if not cfg.enabled:
        return None
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return AttackClassifier(settings.attack_classifier_path, embedder, settings.embedding_model, cfg.block_at, cfg.escalate_at, policy.version)


def attack_checker():
    """The same checks Tool 1 runs on a query. A request counts as stopped if it is blocked or sent to a person."""
    policy = get_policy()
    cfg = policy.input_guardrails
    checks = [
        InjectionDetector(cfg.injection_patterns, policy.version),
        PatternSafetyClassifier(cfg.safety_patterns, policy.version),
    ]
    classifier = build_attack_classifier()
    if classifier:
        checks.append(classifier)
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


def run_grounding_eval() -> dict:
    """Does the real NLI model accept true legal claims and refuse false ones? The public law store only."""
    policy = get_policy()
    settings = get_settings()
    cfg = policy.output_guardrails
    data = json.loads((DATASETS / "grounding.json").read_text(encoding="utf-8"))
    pipeline = OutputGuardrailPipeline(
        policy,
        NLIGrounder(cfg.entailment_threshold, cfg.contradiction_threshold, model_name=settings.nli_model),
        RegulationRetriever.from_json(settings.regulations_path, None),
        RegexPIIDetector(["EMAIL_ADDRESS"], []),  # not used by this check
    )

    def judge(item: dict) -> tuple[bool, float | None]:
        _, result, _ = pipeline.check_grounding_for_citation(item["citation"], item["claim"])
        if result is None:
            return False, None
        return result.verdict is GroundingVerdict.GROUNDED, result.entailment_score

    true_rows, false_rows, scores = {}, {}, {}
    for split in ("dev", "test"):
        judged_true = [judge(c) for c in data["true"][split]]
        judged_false = [judge(c) for c in data["false"][split]]
        accepted = sum(ok for ok, _ in judged_true)
        true_rows[split] = {"accepted": f"{accepted} of {len(judged_true)}", "rate": percent(accepted / len(judged_true))}
        refused = [not ok for ok, _ in judged_false]
        false_rows[split] = score_by_kind(data["false"][split], refused)
        true_scores = [s for _, s in judged_true if s is not None]
        false_scores = [s for _, s in judged_false if s is not None]
        scores[split] = {"lowest_true": round(min(true_scores), 2), "highest_false": round(max(false_scores), 2)}
        if split == "dev":  # never for the test set
            for item, (ok, score) in zip(data["true"][split], judged_true):
                if not ok:
                    print(f"  dev true claim refused (entailment {score:.2f}): {item['claim']}")
            for item, (ok, score) in zip(data["false"][split], judged_false):
                if ok:
                    print(f"  dev FALSE claim accepted ({item['kind']}, entailment {score:.2f}): {item['claim']}")
    return {"true_claims": true_rows, "false_claims": false_rows, "scores": scores}


def print_grounding(grounding: dict) -> None:
    print("\nGrounding: true claims accepted")
    for split, row in grounding["true_claims"].items():
        print(f"  {split:<30}{row['accepted']:<14}{row['rate']}")
    print("\nGrounding: false claims refused")
    print(f"  {'kind':<30}{'dev':<14}{'test':<14}test rate")
    for kind in grounding["false_claims"]["test"]:
        dev, test = grounding["false_claims"]["dev"][kind], grounding["false_claims"]["test"][kind]
        print(f"  {kind:<30}{dev['caught']:<14}{test['caught']:<14}{test['rate']}")
    print("\nGrounding: do the model's scores separate true from false? (lowest true should be above highest false)")
    for split, row in grounding["scores"].items():
        print(f"  {split:<30}lowest true {row['lowest_true']:<8}highest false {row['highest_false']}")


def run_pii_eval() -> dict:
    """Are IBANs, cards, SSNs, emails and account numbers hidden? Presidio against the simple regex detector."""
    cfg = get_policy().input_guardrails
    entities = [e for e in cfg.pii_entities if e not in ("PERSON", "ADDRESS")]  # regex cannot find these
    detectors = {
        "active": build_pii_detector(cfg, "presidio", get_settings().spacy_model),
        "regex": RegexPIIDetector(entities, cfg.custom_pii_recognizers),
    }
    data = json.loads((DATASETS / "pii.json").read_text(encoding="utf-8"))
    results = {}
    for name, detector in detectors.items():
        by_split, wrong = {}, {}
        for split, examples in data["values"].items():
            hidden = []
            for example in examples:
                masked = detector.mask(example["template"].format(value=example["value"])).masked_text
                hidden.append(value_is_hidden(example["value"], masked))
                if split == "dev" and not hidden[-1]:  # never for the test set
                    print(f"  dev miss ({name}, {example['type']}, {example['kind']}): {masked}")
            by_split[split] = score_by_kind([{"kind": e["type"]} for e in examples], hidden)
        for split, sentences in data["normal"].items():
            flagged = [s for s in sentences if any(e.entity_type != "PERSON" for e in detector.detect(s))]
            wrong[split] = {"wrongly_masked": f"{len(flagged)} of {len(sentences)}", "rate": percent(len(flagged) / len(sentences))}
            if split == "dev":
                for sentence in flagged:
                    print(f"  dev false match ({name}): {detector.mask(sentence).masked_text}")
        results[name] = {"values": by_split, "normal_text": wrong}
    return results


def print_pii(pii: dict) -> None:
    print("\nPII: was the value hidden? (test set)")
    print(f"  {'type':<30}{'active':<14}{'regex':<14}")
    for kind in pii["active"]["values"]["test"]:
        print(f"  {kind:<30}{pii['active']['values']['test'][kind]['caught']:<14}{pii['regex']['values']['test'][kind]['caught']:<14}")
    print("\nPII: normal sentences where a number was wrongly masked")
    print(f"  {'set':<30}{'active':<14}{'regex':<14}")
    for split in pii["active"]["normal_text"]:
        print(f"  {split:<30}{pii['active']['normal_text'][split]['wrongly_masked']:<14}{pii['regex']['normal_text'][split]['wrongly_masked']:<14}")


def run_speed_eval() -> dict:
    """How long does masking take? The active policy against the last one without GLiNER."""
    data = json.loads((DATASETS / "names.json").read_text(encoding="utf-8"))
    sentences = [t.format(name="Clara Openfield") for t in data["templates"]] * 5
    rows = {}
    for label, cfg in (("v1.9.0, Presidio only", load_policy("1.9.0").input_guardrails), ("active policy", get_policy().input_guardrails)):
        detector = build_pii_detector(cfg, "presidio", get_settings().spacy_model)
        detector.mask(sentences[0])  # the first call loads the models, so it is not timed
        times = []
        for sentence in sentences:
            start = time.perf_counter()
            detector.mask(sentence)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        rows[label] = {"p50_ms": round(times[len(times) // 2]), "p95_ms": round(times[int(len(times) * 0.95)])}
    classifier = build_attack_classifier()
    if classifier:  # the attack classifier's own cost, per request
        classifier.score(sentences[0])  # the first call loads MiniLM, so it is not timed
        times = []
        for sentence in sentences:
            start = time.perf_counter()
            classifier.score(sentence)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        rows["attack classifier"] = {"p50_ms": round(times[len(times) // 2]), "p95_ms": round(times[int(len(times) * 0.95)])}
    return rows


def print_speed(speed: dict) -> None:
    print("\nSpeed: time for one short sentence (masking, and the attack classifier if on)")
    print(f"  {'detector':<30}{'p50':<14}p95")
    for label, row in speed.items():
        print(f"  {label:<30}{str(row['p50_ms']) + ' ms':<14}{row['p95_ms']} ms")


def run_sanctions_eval() -> dict:
    """Does screening catch variations of a listed name, and leave innocent people alone?
    Always the fictional list, so the score is the same on every machine: it tests the matching, not the real lists."""
    policy = get_policy()
    screener = SanctionsScreener.from_policy(get_settings().sanctions_test_list_path, policy.sanctions_screening, "test")
    data = json.loads((DATASETS / "sanctions.json").read_text(encoding="utf-8"))

    def flagged(name: str) -> bool:
        return screener.screen(name).recommendation != "clear"

    listed, innocent = {}, {}
    for split, examples in data["listed"].items():
        caught = [flagged(e["name"]) for e in examples]
        listed[split] = score_by_kind(examples, caught)
        if split == "dev":  # never for the test set
            for example, was_caught in zip(examples, caught):
                if not was_caught:
                    print(f"  dev miss ({example['kind']}): {example['name']}")
    for split, names in data["innocent"].items():
        wrongly = [n for n in names if flagged(n)]
        innocent[split] = {"wrongly_flagged": f"{len(wrongly)} of {len(names)}", "rate": percent(len(wrongly) / len(names))}
        if split == "dev":
            for name in wrongly:
                print(f"  dev false alarm: {name}")
    return {"threshold": policy.sanctions_screening.fuzzy_match_threshold, "listed": listed, "innocent": innocent}


def print_sanctions(sanctions: dict) -> None:
    print(f"\nSanctions: was a variation of a listed name caught? (match threshold {sanctions['threshold']})")
    print(f"  {'kind':<30}{'dev':<14}{'test':<14}test rate")
    for kind in sanctions["listed"]["test"]:
        dev, test = sanctions["listed"]["dev"][kind], sanctions["listed"]["test"][kind]
        print(f"  {kind:<30}{dev['caught']:<14}{test['caught']:<14}{test['rate']}")
    print("\nSanctions: innocent names wrongly flagged")
    for split, row in sanctions["innocent"].items():
        print(f"  {split:<30}{row['wrongly_flagged']:<14}{row['rate']}")


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
    print("\nSanctions: what the dev set got wrong")
    sanctions = run_sanctions_eval()
    print_sanctions(sanctions)
    print("\nGrounding: what the dev set got wrong (loading the NLI model can take a minute)")
    grounding = run_grounding_eval()
    print_grounding(grounding)
    print("\nPII: what the dev set got wrong")
    pii = run_pii_eval()
    print_pii(pii)
    speed = run_speed_eval()
    print_speed(speed)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results = {"policy": get_policy().version, "attacks": attacks, "names": names, "sanctions": sanctions, "grounding": grounding, "pii": pii, "speed": speed}
    RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
