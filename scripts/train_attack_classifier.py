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
EVAL_SET = Path("evals/datasets/attacks_v2.json")  # dev = practice (tune here), test = the exam (never tuned on)
MODEL_OUT = Path("data/classifiers/attack_classifier_v2.json")  # new file: policy v1.13.0 keeps the old model
# Settings we try, 3 x 4 = 12 models. Chosen on practice + traffic check only, never on the exam.
WEIGHTS = (1, 3, 5)         # "row votes": how many times each of our AML rows counts
CS = (0.1, 0.3, 1.0, 3.0)   # C: small = broad rules only, big = allowed to memorise
NEAR_COPY = 0.90            # a training row this similar to a held-out sentence is a near-copy
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


def check_near_copies(X_aml: np.ndarray, X_held_out: np.ndarray) -> None:
    """Stops if any AML training row is almost the same sentence as a practice, exam or traffic sentence.
    Prints row numbers only, never the sentences."""
    S = X_aml @ X_held_out.T   # vectors have length 1, so this is cosine similarity
    pairs = list(zip(*(S >= NEAR_COPY).nonzero()))
    print(f"near-copy check: highest similarity {S.max():.2f} | pairs at {NEAR_COPY} or more: {len(pairs)}")
    for i, j in pairs:
        print(f"  AML training row {i} ~ held-out row {j}  similarity {S[i, j]:.2f}")
    assert not pairs, "training rows too close to held-out sentences: rewrite those training rows"


def train(X: np.ndarray, labels: np.ndarray, n_aml: int, our_weight: int, C: float) -> LogisticRegression:
    """Public rows count 1, our AML rows (the last n_aml) count our_weight."""
    weights = np.ones(len(labels))
    weights[-n_aml:] = our_weight
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")  # balanced: attacks vs normal counts
    return clf.fit(X, labels, sample_weight=weights)


def compare_settings(X, labels, n_aml, X_dev, dev_labels, X_traffic) -> tuple[int, float]:
    """Trains 12 models and picks one by a rule written BEFORE looking at the table:
      1. no normal FinGuard traffic stopped
      2. at most 1 normal practice sentence stopped
      3. then: the most practice attacks stopped
      4. tie: the lowest top traffic score (most room before a false stop)"""
    n_att, n_norm = int((dev_labels == 1).sum()), int((dev_labels == 0).sum())
    print(f"\nvotes     C | practice attacks stopped | practice normal stopped | traffic stopped | top traffic")
    results = []
    for w in WEIGHTS:
        for C in CS:
            clf = train(X, labels, n_aml, w, C)
            dev_stop = clf.predict_proba(X_dev)[:, 1] >= ESCALATE_AT    # escalate or block = stopped
            traffic = clf.predict_proba(X_traffic)[:, 1]
            r = {"w": w, "C": C,
                 "attacks": int((dev_stop & (dev_labels == 1)).sum()),
                 "normal": int((dev_stop & (dev_labels == 0)).sum()),
                 "traffic": int((traffic >= ESCALATE_AT).sum()),
                 "top": float(traffic.max())}
            results.append(r)
            print(f"{w:>5} {C:>5} | {r['attacks']:>10} of {n_att:<11} | {r['normal']:>10} of {n_norm:<10} "
                  f"| {r['traffic']:>7} of {len(traffic):<5} | {r['top']:.2f}")
    safe = [r for r in results if r["traffic"] == 0 and r["normal"] <= 1]
    assert safe, "no setting passes the safety rules: keep the old model"
    best = max(safe, key=lambda r: (r["attacks"], -r["top"]))
    print(f"\nchosen by the rule: votes {best['w']}, C {best['C']}")
    return best["w"], best["C"]


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


def save_model(clf: LogisticRegression, labels: np.ndarray, aml: list[dict], X_dev: np.ndarray, our_weight: int) -> None:
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
            "our_weight": our_weight,
            "C": float(clf.C),
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

    evals = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    dev = evals["dev"]
    dev_labels = np.array([int(r["label"] == "attack") for r in dev])
    X_dev = embedder.encode([r["text"] for r in dev], normalize_embeddings=True)
    traffic = load_traffic_check()
    X_traffic = embedder.encode(traffic, normalize_embeddings=True)

    # The exam is turned into numbers only for the near-copy check: never scored, never printed.
    X_exam = embedder.encode([r["text"] for r in evals["test"]], normalize_embeddings=True)
    check_near_copies(X[-len(aml):], np.vstack([X_dev, X_exam, X_traffic]))

    # --- Step 9e: try 12 settings, pick one by the rule written in advance ---
    our_weight, C = compare_settings(X, labels, len(aml), X_dev, dev_labels, X_traffic)

    # 5-fold check of the chosen settings: train on 4/5, score the 1/5 it never saw, repeat 5 times.
    weights = np.ones(len(labels))
    weights[-len(aml):] = our_weight
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_predict(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"), X, labels,
                                  cv=folds, method="predict_proba", params={"sample_weight": weights})[:, 1]
    report("\n5-fold, all training rows", cv_scores, labels, 0.5)
    report("5-fold, our AML rows only", cv_scores[-len(aml):], labels[-len(aml):], 0.5)

    # Train the final model on everything with the chosen settings, then score practice. Never the exam.
    clf = train(X, labels, len(aml), our_weight, C)
    dev_scores = clf.predict_proba(X_dev)[:, 1]
    print()
    for cutoff in (0.3, 0.5, 0.7, 0.9):
        report("AML dev", dev_scores, dev_labels, cutoff)

    print("\nAML dev, highest score first:")
    for row, score in sorted(zip(dev, dev_scores), key=lambda pair: -pair[1]):
        print(f"  {score:.2f}  {decide(score):<9}{row['label']:<7} {row['text'][:70]}".encode("ascii", "replace").decode())

    # --- Step 6: three outcomes and a saved model ---
    outcomes = [decide(s) for s in dev_scores]
    normal_out = [o for o, y in zip(outcomes, dev_labels) if y == 0]
    print(f"\nAML dev: block {outcomes.count('block')} | escalate {outcomes.count('escalate')} | allow {outcomes.count('allow')}"
          f"  (normal: block {normal_out.count('block')}, escalate {normal_out.count('escalate')})")

    # Real-looking FinGuard traffic: none of it should be stopped.
    traffic_scores = clf.predict_proba(X_traffic)[:, 1]
    stopped = [(s, t) for s, t in zip(traffic_scores, traffic) if decide(s) != "allow"]
    print(f"\nTraffic check: {len(stopped)} of {len(traffic)} normal requests stopped | highest score {traffic_scores.max():.2f}")
    for score, text in sorted(stopped, reverse=True):
        print(f"  {score:.2f}  {decide(score):<9}{text[:70]}")

    save_model(clf, labels, aml, X_dev, our_weight)
