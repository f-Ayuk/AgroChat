"""Fine-tune a Sentence-Transformer on agriculture QA pairs.

Usage:
    python scripts/train_embedder.py --output models/agri-minilm

MultipleNegativesRankingLoss treats every other answer in the batch as a
negative for the current question. No explicit negative sampling needed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
)
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"


def load_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["question"].strip(), r["answer"].strip()) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--output", default=str(BASE_DIR / "models" / "agri-minilm"))
    args = ap.parse_args()

    train_pairs = load_pairs(DATA / "qa_train.csv")
    val_pairs = load_pairs(DATA / "qa_val.csv")
    print(f"[info] train={len(train_pairs)}  val={len(val_pairs)}")

    examples = [InputExample(texts=[q, a]) for q, a in train_pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    model = SentenceTransformer(args.base_model)
    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries={i: q for i, (q, _) in enumerate(val_pairs)},
        corpus={i: a for i, (_, a) in enumerate(val_pairs)},
        relevant_docs={i: {i} for i in range(len(val_pairs))},
        name="agri-val",
    )

    steps = max(1, len(loader))
    warmup = max(1, int(0.1 * steps * args.epochs))

    model.fit(
        train_objectives=[(loader, loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup,
        output_path=args.output,
        show_progress_bar=True,
    )
    print(f"[done] saved fine-tuned embedder to {args.output}")


if __name__ == "__main__":
    main()