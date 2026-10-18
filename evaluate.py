"""Evaluate baseline vs. fine-tuned embedder across thresholds.

Run:
    python evaluate.py

Writes:
    eval_results.csv

Uses the held-out test split (data/qa_test.csv) for retrieval accuracy and
a fixed list of out-of-domain questions for rejection rate.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"

OUT_OF_DOMAIN_TESTS = [
    "Who won the football match yesterday?",
    "Explain the history of ancient Rome.",
    "How do I write a Python web scraper?",
    "What is the capital of Japan?",
    "Recommend a romantic movie.",
    "How do I fix a broken laptop screen?",
    "Who is the current president of France?",
    "Write a poem about the ocean.",
    "What is the stock price of a technology company?",
    "Explain the rules of chess.",
    "What is the boiling point of mercury?",
    "How do I install Docker on Windows?",
]

THRESHOLDS = [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60]

CONFIGS = [
    ("baseline (pretrained MiniLM)",
     "sentence-transformers/all-MiniLM-L6-v2"),
    ("fine-tuned MiniLM",
     str(BASE_DIR / "models" / "agri-minilm")),
]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(embedder_path: str):
    path = embedder_path
    if path.startswith("models/") or path.startswith(str(BASE_DIR)):
        if not Path(path).exists():
            print(f"[skip] {path} missing — run train_embedder.py first")
            return None, None
    model = SentenceTransformer(path)
    test_rows = load_csv(DATA / "qa_test.csv")
    corpus = [r["answer"] for r in test_rows]
    corpus_emb = model.encode(
        corpus, normalize_embeddings=True, convert_to_numpy=True
    )
    return model, (test_rows, corpus_emb)


def evaluate(model, payload, threshold: float):
    test_rows, corpus_emb = payload

    # In-domain retrieval: for each test question, retrieve the closest answer
    # in the test corpus. Correct if the retrieved answer belongs to the same row.
    top1 = 0
    accepted_in = 0
    times = []
    for r in test_rows:
        t0 = time.perf_counter()
        q_emb = model.encode(
            [r["question"]], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = corpus_emb @ q_emb
        best = int(np.argmax(scores))
        times.append((time.perf_counter() - t0) * 1000)

        # "Correct" = retrieved answer is the expected one.
        if best == test_rows.index(r) or corpus[best] == r["answer"]:
            top1 += 1

        if float(scores[best]) >= threshold:
            accepted_in += 1

    n = len(test_rows)
    top1_acc = top1 / n
    in_accept = accepted_in / n

    # Out-of-domain rejection.
    rejected = 0
    for q in OUT_OF_DOMAIN_TESTS:
        q_emb = model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = corpus_emb @ q_emb
        if float(np.max(scores)) < threshold:
            rejected += 1
    ood_rej = rejected / len(OUT_OF_DOMAIN_TESTS)

    # Binary domain-detection P/R/F1.
    y_true = [1] * n + [0] * len(OUT_OF_DOMAIN_TESTS)
    y_pred = []
    for r in test_rows:
        q_emb = model.encode(
            [r["question"]], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        y_pred.append(int(float(np.max(corpus_emb @ q_emb)) >= threshold))
    for q in OUT_OF_DOMAIN_TESTS:
        q_emb = model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        y_pred.append(int(float(np.max(corpus_emb @ q_emb)) >= threshold))

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    return {
        "top1_accuracy": round(top1_acc, 4),
        "in_domain_acceptance": round(in_accept, 4),
        "ood_rejection_rate": round(ood_rej, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "mean_ms": round(float(np.mean(times)), 2),
    }


def main():
    results = []
    for label, path in CONFIGS:
        model, payload = build(path)
        if model is None:
            continue
        print(f"\n=== {label} ===")
        for th in THRESHOLDS:
            m = evaluate(model, payload, th)
            print(
                f"  th={th:.2f}  top1={m['top1_accuracy']:.3f}  "
                f"in={m['in_domain_acceptance']:.3f}  "
                f"ood_rej={m['ood_rejection_rate']:.3f}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  ms={m['mean_ms']:.1f}"
            )
            results.append({"config": label, "threshold": th, **m})

    out = BASE_DIR / "eval_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n[done] written to {out}")


if __name__ == "__main__":
    main()