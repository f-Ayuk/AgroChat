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

    if not test_rows:
        raise ValueError("The test dataset is empty.")

    if len(OUT_OF_DOMAIN_TESTS) == 0:
        raise ValueError("The out-of-domain test set is empty.")

    questions = [r["question"] for r in test_rows]
    expected_answers = [r["answer"] for r in test_rows]

    # Encode each in-domain question once.
    t0 = time.perf_counter()
    question_emb = model.encode(
        questions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    times = [elapsed_ms / len(questions)] * len(questions)

    # Similarity between every question and every answer in the corpus.
    scores_in = question_emb @ corpus_emb.T
    best_indices = np.argmax(scores_in, axis=1)
    best_scores = scores_in[np.arange(len(test_rows)), best_indices]

    # Correct if the retrieved answer matches the expected answer.
    top1 = sum(
        corpus_answer == expected
        for corpus_answer, expected in zip(
            [test_rows[i]["answer"] for i in best_indices],
            expected_answers,
        )
    )

    accepted_in = int(np.sum(best_scores >= threshold))

    # Encode out-of-domain questions once.
    ood_emb = model.encode(
        OUT_OF_DOMAIN_TESTS,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores_ood = ood_emb @ corpus_emb.T
    best_ood_scores = np.max(scores_ood, axis=1)
    rejected = int(np.sum(best_ood_scores < threshold))

    n = len(test_rows)
    n_ood = len(OUT_OF_DOMAIN_TESTS)

    # 1 = in-domain, 0 = out-of-domain.
    y_true = [1] * n + [0] * n_ood
    y_pred = (
        (best_scores >= threshold).astype(int).tolist()
        + (best_ood_scores >= threshold).astype(int).tolist()
    )

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)

    return {
        "top1_accuracy": round(top1 / n, 4),
        "in_domain_acceptance": round(accepted_in / n, 4),
        "ood_rejection_rate": round(rejected / n_ood, 4),
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