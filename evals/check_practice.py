"""Error analysis on the practice (dev) set: every sentence, its score, and every mistake.

Run:  python evals/check_practice.py

The exam (test) is never read here.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.run_evals import attack_checker, build_attack_classifier  # noqa: E402

BOXES = {(True, True): "TP", (True, False): "FN", (False, True): "FP", (False, False): "TN"}  # (is attack, was stopped)

sys.stdout.reconfigure(errors="replace")   # a Windows terminal must not crash on an odd character
rows = json.loads((ROOT / "evals/datasets/attacks_v2.json").read_text(encoding="utf-8"))["dev"]
stopped = attack_checker()                 # FinGuard as it runs: phrase lists + classifier
classifier = build_attack_classifier()

matrix = Counter()                         # the confusion matrix
by_group = Counter()                       # right and wrong per group
results = []
for row in rows:
    score = classifier.score(row["text"])
    box = BOXES[(row["label"] == "attack", stopped(row["text"]))]
    matrix[box] += 1
    by_group[(row["group"], box in ("FN", "FP"))] += 1
    results.append((score, box, row))

tp, fn, fp, tn = matrix["TP"], matrix["FN"], matrix["FP"], matrix["TN"]
print(f"\nConfusion matrix:  TP {tp}  FN {fn}  FP {fp}  TN {tn}")
print(f"recall {tp / max(tp + fn, 1):.0%} | false stop rate {fp / max(fp + tn, 1):.0%} | precision {tp / max(tp + fp, 1):.0%}")

print("\nMistakes per group:")
for group in sorted({g for g, _ in by_group}):
    wrong = by_group[(group, True)]
    print(f"  {group:<18} mistakes {wrong} of {wrong + by_group[(group, False)]}")

print("\nEvery practice sentence, highest score first (X = mistake):")
for score, box, row in sorted(results, key=lambda r: -r[0]):
    mark = "X" if box in ("FN", "FP") else " "
    print(f" {mark} {score:.2f} {box} {row['label']:<7}{row['group']:<18}{row['text'][:80]}")
