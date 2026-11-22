"""Step 9f: two honesty checks on the v2 attack classifier. Changes nothing and saves no model.

Run:  python evals/check_generalisation.py

1. Leave Gandalf out: train again WITHOUT Gandalf's train split, then score Gandalf's test split.
   Shows whether 111 of 112 on Gandalf came from understanding attacks, or from having seen Gandalf-style text.
2. deepset by language: the deepset test score split into English rows and German rows.

Only public test splits are scored. The exam is never scored here, and nothing here is used to choose settings.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                              # the training script uses paths relative to the project folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_attack_classifier as t  # noqa: E402  the same training code as step 9e
from evals.run_evals import DEEPSET, GANDALF, public_rows  # noqa: E402

V2 = ROOT / "data" / "classifiers" / "attack_classifier_v2.json"
OUT = ROOT / "evals" / "results" / "generalisation.json"
GERMAN_WORDS = (" und ", " der ", " die ", " ich ")  # rough: the same check as the training script


def looks_german(text: str) -> bool:
    return any(word in f" {text.lower()} " for word in GERMAN_WORDS)


def counts(stopped: np.ndarray, labels: np.ndarray) -> dict:
    """Attacks stopped and normal texts wrongly stopped. labels: 1 = attack, 0 = normal."""
    return {"attacks_stopped": int((stopped & (labels == 1)).sum()), "attacks": int((labels == 1).sum()),
            "normal_stopped": int((stopped & (labels == 0)).sum()), "normal": int((labels == 0).sum())}


def line(name: str, c: dict) -> str:
    return (f"  {name:<34} attacks stopped {c['attacks_stopped']:>3} of {c['attacks']:<4}"
            f"| normal stopped {c['normal_stopped']:>2} of {c['normal']}")


model = json.loads(V2.read_text(encoding="utf-8"))
weights, bias, cutoff = np.array(model["weights"]), model["bias"], model["escalate_at"]  # stopped = escalate or block
our_weight, C = model["trained_on"]["our_weight"], model["trained_on"]["C"]
embedder = SentenceTransformer(t.EMBEDDER)


def embed(texts: list[str]) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def v2_stops(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ weights + bias))) >= cutoff


# --- Check 1: leave Gandalf out ---
gandalf_test = public_rows(GANDALF)
X_gandalf = embed([r["text"] for r in gandalf_test])
gandalf_labels = np.ones(len(gandalf_test), dtype=int)   # Gandalf has attacks only

aml = t.load_aml_train()
texts, labels = t.build_training_set(t.download_train(DEEPSET), [], aml)   # [] = no Gandalf rows
no_gandalf = t.train(embed(texts), labels, len(aml), our_weight, C)        # same settings as v2
with_g = counts(v2_stops(X_gandalf), gandalf_labels)
without_g = counts(no_gandalf.predict_proba(X_gandalf)[:, 1] >= cutoff, gandalf_labels)

print(f"\nCheck 1: Gandalf test, classifier alone (votes {our_weight}, C {C}, stopped = score {cutoff} or more)")
print(line("trained WITH Gandalf (v2 model)", with_g))
print(line("trained WITHOUT Gandalf", without_g))

# --- Check 2: deepset by language ---
deepset_test = public_rows(DEEPSET)
stopped = v2_stops(embed([r["text"] for r in deepset_test]))
deepset_labels = np.array([int(r["label"]) for r in deepset_test])
german = np.array([looks_german(r["text"]) for r in deepset_test])
by_language = {name: counts(stopped[mask], deepset_labels[mask])
               for name, mask in (("all", np.ones_like(german)), ("English", ~german), ("German", german))}

print("\nCheck 2: deepset test, classifier alone (language found by a rough word check)")
for name, c in by_language.items():
    print(line(f"{name} ({c['attacks'] + c['normal']} rows)", c))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"model": V2.name, "our_weight": our_weight, "C": C,
                           "gandalf_with": with_g, "gandalf_without": without_g,
                           "deepset_by_language": by_language}, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved to {OUT.relative_to(ROOT).as_posix()}")
