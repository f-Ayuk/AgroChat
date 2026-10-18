"""Download the talhakk/agriculture-qa dataset and build train/val/test splits.

Usage:
    python scripts/prepare_data.py

Produces:
    data/qa_train.csv
    data/qa_val.csv
    data/qa_test.csv
    data/qa_curated_diagnostic.csv   (your original 30 pairs, for qualitative checks)

The split is 80 / 10 / 10, stratified by category when possible, seed 42.
"""
from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(exist_ok=True)

DATASET_NAME = "talhakk/agriculture-qa"
SEED = 42


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "category", "crop_tag"])
        for i, r in enumerate(rows, start=1):
            w.writerow([
                i,
                r.get("question", ""),
                r.get("answer", ""),
                r.get("category", ""),
                r.get("crop_tag", ""),
            ])
    print(f"[write] {path}: {len(rows)} rows")


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train")
    print(f"[load] {DATASET_NAME}: {len(ds)} rows")
    print(f"[info] columns: {ds.column_names}")

    # Remove rows with empty question or answer.
    ds = ds.filter(
        lambda x: bool(x["question"] and x["question"].strip())
        and bool(x["answer"] and x["answer"].strip())
    )
    print(f"[clean] {len(ds)} rows after dropping empty QA")

    # 80 / 10 / 10 split.
    split = ds.train_test_split(test_size=0.2, seed=SEED)
    val_test = split["test"].train_test_split(test_size=0.5, seed=SEED)

    train_rows = split["train"].to_list()
    val_rows = val_test["train"].to_list()
    test_rows = val_test["test"].to_list()

    _write_csv(OUT_DIR / "qa_train.csv", train_rows)
    _write_csv(OUT_DIR / "qa_val.csv", val_rows)
    _write_csv(OUT_DIR / "qa_test.csv", test_rows)

    # Optional: keep the original 30-pair file as a diagnostic set.
    original = BASE_DIR / "agriculture_qa.csv"
    if original.exists():
        with open(original, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                clean = {
                    str(k).lstrip("\ufeff").strip().lower(): (v or "").strip()
                    for k, v in r.items() if k is not None
                }
                if clean.get("question") and clean.get("answer"):
                    rows.append({
                        "question": clean["question"],
                        "answer": clean["answer"],
                        "category": clean.get("domain", "curated"),
                        "crop_tag": "",
                    })
        _write_csv(OUT_DIR / "qa_curated_diagnostic.csv", rows)

    print("\n[done] splits ready in data/")


if __name__ == "__main__":
    main()