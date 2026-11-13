"""Trains FinGuard's attack classifier on public train splits and our own AML examples."""
import datetime
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"  # same model as settings.embedding_model
CACHE = Path("evals/cache")  # gitignored: public data is never committed
URL = "https://datasets-server.huggingface.co/rows?"
AML_TRAIN = Path("data/training/aml_attack_train.json")
EVAL_SET = Path("evals/datasets/attacks.json")
OUR_WEIGHT = 3  # each of our AML rows counts 3x; chosen on dev and the traffic check (5 blocked more normal texts)
MODEL_OUT = Path("data/classifiers/attack_classifier.json")  # small, readable, safe to commit
BLOCK_AT = 0.70     # sure it's an attack: block. Chosen on dev only.
ESCALATE_AT = 0.50  # unsure: a human decides. Above every sample alert (top 0.45); 7 of 9 dev attacks stopped.
ALERTS = Path("data/sample_alerts/alerts.json")
# Normal FinGuard traffic that must pass: the sample alert queries and two sentences from test/unit/test_api.py.
# Never trained on. Like dev, it checks the model; it is used for choosing cut-offs, so the test set stays the judge.
TRAFFIC_EXTRA = ["Customer IBAN is GB82 WEST 1234 5698 7654 32", "Customer email is jane.doe@example.com"]


def download_train(dataset: str) -> list[dict]:
    """A public train split, downloaded once. Test splits are never used for training."""
    path = CACHE / (dataset.replace("/", "_") + "_train.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": 100})
        request = urllib.request.Request(URL + query, headers={"User-Agent": "finguard-eval"})
        with urllib.request.urlopen(request, timeout=60) as response:
            batch = [item["row"] for item in json.load(response)["rows"]]
        rows += batch
        offset += len(batch)
        if len(batch) < 100:  # a short page means this was the last one
            break
        time.sleep(0.3)  # be polite to the free service
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def load_traffic_check() -> list[str]:
    """Normal FinGuard requests: every one of these should be allowed."""
    alerts = json.loads(ALERTS.read_text(encoding="utf-8"))["alerts"]
    return [a["query"] for a in alerts] + TRAFFIC_EXTRA


def load_aml_train() -> list[dict]:
    """Our hand-written AML examples. Stops if any sentence also appears in dev, test or the traffic check."""
    rows = json.loads(AML_TRAIN.read_text(encoding="utf-8"))["rows"]
    evals = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    held_out = {r["text"].strip().lower() for r in evals["dev"] + evals["test"]}
    held_out |= {text.strip().lower() for text in load_traffic_check()}
    leaked = [r["text"] for r in rows if r["text"].strip().lower() in held_out]
    assert not leaked, f"training rows copied from dev/test/traffic check: {leaked}"
    return rows


def build_training_set(deepset: list[dict], gandalf: list[dict], aml: list[dict]) -> tuple[list[str], np.ndarray]:
    """All texts in one list, and a label for each: 1 = attack, 0 = normal."""
    texts = [r["text"] for r in deepset + gandalf + aml]
    labels = ([int(r["label"]) for r in deepset]            # deepset already uses 1 / 0
              + [1] * len(gandalf)                          # Gandalf has attacks only
              + [int(r["label"] == "attack") for r in aml])  # ours use words
    return texts, np.array(labels)


def report(name: str, scores: np.ndarray, labels: np.ndarray, cutoff: float) -> None:
    """Attacks caught and normal texts wrongly blocked at one cut-off. Never one 'accuracy' number."""
    flagged = scores >= cutoff
    print(f"{name} at {cutoff}: attacks caught {(flagged & (labels == 1)).sum()} of {(labels == 1).sum()}"
          f" | normal blocked {(flagged & (labels == 0)).sum()} of {(labels == 0).sum()}")


def decide(score: float) -> str:
    """Three outcomes, not two: the unsure middle goes to a person."""
    if score >= BLOCK_AT:
        return "block"
    if score >= ESCALATE_AT:
        return "escalate"
    return "allow"


def score_from_saved(model: dict, X: np.ndarray) -> np.ndarray:
    """The same maths FinGuard will use at runtime: only numpy, no scikit-learn."""
    return 1 / (1 + np.exp(-(X @ np.array(model["weights"]) + model["bias"])))


def save_model(clf: LogisticRegression, labels: np.ndarray, aml: list[dict], X_dev: np.ndarray) -> None:
    """Saves the 384 weights and the bias as JSON (never pickle: loading a pickle can run code).
    Stops before writing if the saved numbers don't give the same scores as scikit-learn."""
    model = {
        "embedder": EMBEDDER,
        "weights": [float(w) for w in clf.coef_[0]],
        "bias": float(clf.intercept_[0]),
        "block_at": BLOCK_AT,
        "escalate_at": ESCALATE_AT,
        "trained_on": {
            "rows": len(labels),
            "attacks": int((labels == 1).sum()),
            "normal": int((labels == 0).sum()),
            "public": ["deepset/prompt-injections (train)", "Lakera/gandalf_ignore_instructions (train)"],
            "our_weight": OUR_WEIGHT,
            # fingerprint of the rows, not the file bytes, so Windows/Linux line endings don't change it
            "aml_rows_sha256": hashlib.sha256(json.dumps(aml, sort_keys=True).encode("utf-8")).hexdigest(),
        },
        "trained": datetime.date.today().isoformat(),
    }
    text = json.dumps(model, indent=2)
    same = np.allclose(score_from_saved(json.loads(text), X_dev), clf.predict_proba(X_dev)[:, 1])
    assert same, "saved weights give different scores: file not written"
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(text + "\n", encoding="utf-8")
    print("\nsaved", MODEL_OUT.as_posix())
    print("saved file gives the same scores as scikit-learn:", same)


def show(title: str, rows: list[dict], n: int = 3) -> None:
    """Print the first few texts of a group, cut short so they fit on one line."""
    print(f"\n{title}")
    for r in rows[:n]:
        print("  -", r["text"][:90].replace("\n", " "))


if __name__ == "__main__":
    deepset = download_train("deepset/prompt-injections")
    gandalf = download_train("Lakera/gandalf_ignore_instructions")
    aml = load_aml_train()
    print("our AML examples:", len(aml), "| attack:", sum(r["label"] == "attack" for r in aml),
          "| normal:", sum(r["label"] == "normal" for r in aml))

    attacks = [r for r in deepset if r["label"] == 1] + gandalf  # Gandalf has attacks only
    normal = [r for r in deepset if r["label"] == 0]
    print("attacks:", len(attacks), "| normal:", len(normal))

    show("Attacks from deepset", [r for r in deepset if r["label"] == 1])
    show("Attacks from Gandalf", gandalf)
    show("Normal text from deepset", normal)

    # A rough check: common German words. Not perfect, but enough to see the share.
    german = [r for r in deepset if any(w in f" {r['text'].lower()} " for w in (" und ", " der ", " die ", " ich "))]
    print("\ndeepset rows that look German:", len(german), "of", len(deepset))

    lengths = sorted(len(r["text"]) for r in attacks + normal)
    print("text length in characters: shortest", lengths[0], "| middle", lengths[len(lengths) // 2], "| longest", lengths[-1])

    # --- Step 4: turn texts into numbers, train, and check ---
    texts, labels = build_training_set(deepset, gandalf, aml)
    print("\ntraining rows:", len(labels), "| attack:", (labels == 1).sum(), "| normal:", (labels == 0).sum())

    embedder = SentenceTransformer(EMBEDDER)  # frozen: we only read its numbers, we never change it
    X = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    print("each text is now", X.shape[1], "numbers")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")  # balanced: fixes 980 vs 343

    # Public rows count 1, our AML rows count OUR_WEIGHT, so our rows aren't drowned out by 1,300.
    weights = np.ones(len(labels))
    weights[-len(aml):] = OUR_WEIGHT

    # 5-fold check: train on 4/5, score the 1/5 it never saw, repeat 5 times.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_predict(clf, X, labels, cv=folds, method="predict_proba",
                                  params={"sample_weight": weights})[:, 1]
    report("\n5-fold, all training rows", cv_scores, labels, 0.5)
    report("5-fold, our AML rows only", cv_scores[-len(aml):], labels[-len(aml):], 0.5)

    # Train once on everything, then score the dev set. The test set is never scored here.
    clf.fit(X, labels, sample_weight=weights)
    dev = json.loads(EVAL_SET.read_text(encoding="utf-8"))["dev"]
    dev_labels = np.array([int(r["label"] == "attack") for r in dev])
    X_dev = embedder.encode([r["text"] for r in dev], normalize_embeddings=True)
    dev_scores = clf.predict_proba(X_dev)[:, 1]
    print()
    for cutoff in (0.3, 0.5, 0.7, 0.9):
        report("AML dev", dev_scores, dev_labels, cutoff)

    print("\nAML dev, highest score first:")
    for row, score in sorted(zip(dev, dev_scores), key=lambda pair: -pair[1]):
        print(f"  {score:.2f}  {decide(score):<9}{row['label']:<7} {row['text'][:70]}")

    # --- Step 6: three outcomes and a saved model ---
    outcomes = [decide(s) for s in dev_scores]
    normal_out = [o for o, y in zip(outcomes, dev_labels) if y == 0]
    print(f"\nAML dev: block {outcomes.count('block')} | escalate {outcomes.count('escalate')} | allow {outcomes.count('allow')}"
          f"  (normal: block {normal_out.count('block')}, escalate {normal_out.count('escalate')})")

    # Real-looking FinGuard traffic: none of it should be stopped.
    traffic = load_traffic_check()
    traffic_scores = clf.predict_proba(embedder.encode(traffic, normalize_embeddings=True))[:, 1]
    stopped = [(s, t) for s, t in zip(traffic_scores, traffic) if decide(s) != "allow"]
    print(f"\nTraffic check: {len(stopped)} of {len(traffic)} normal requests stopped | highest score {traffic_scores.max():.2f}")
    for score, text in sorted(stopped, reverse=True):
        print(f"  {score:.2f}  {decide(score):<9}{text[:70]}")

    save_model(clf, labels, aml, X_dev)
