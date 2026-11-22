"""Near-copy check. Prints only row numbers and scores, never the sentences.

Part 1: the exam (test) must not be close to anything we train on or tune on (training + practice).
Part 2: practice (dev) must not be close to training, or practice scores would look better than they are.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

EVAL_FILE = Path("evals/datasets/attacks_v2.json")
TRAIN_FILE = Path("data/training/aml_attack_train.json")
DANGER = 0.90   # this close or closer = "almost the same sentence"

evals = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
train = [r["text"] for r in json.loads(TRAIN_FILE.read_text(encoding="utf-8"))["rows"]]
practice = [r["text"] for r in evals["dev"]]
exam = [r["text"] for r in evals["test"]]

# Every sentence becomes 384 numbers of length 1, so cosine similarity is a plain dot product.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
T, P, E = (model.encode(texts, normalize_embeddings=True) for texts in (train, practice, exam))


def report(title: str, A: np.ndarray, a_name: str, B: np.ndarray, b_names: list[str]) -> None:
    """Compares every row of A with every row of B. b_names says which set each row of B came from."""
    S = A @ B.T   # one similarity score per pair
    print(f"\n{title}")
    print("  pairs checked:", S.size, "| highest similarity:", round(float(S.max()), 2))
    print(f"  pairs at {DANGER} or more:", int((S >= DANGER).sum()))
    for i, j in zip(*(S >= DANGER).nonzero()):
        print(f"    {a_name} row {i} ~ {b_names[j]}  similarity {S[i, j]:.2f}")


report("Part 1: exam vs training + practice", E, "exam", np.vstack([T, P]),
       [f"training row {j}" for j in range(len(T))] + [f"practice row {j}" for j in range(len(P))])
report("Part 2: practice vs training", P, "practice", T, [f"training row {j}" for j in range(len(T))])
