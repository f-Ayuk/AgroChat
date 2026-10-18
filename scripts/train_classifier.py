# scripts/train_classifier.py
import numpy as np, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def to_ds(df, tok):
    ds = Dataset.from_pandas(df[["text", "label"]])
    ds = ds.map(lambda x: tok(x["text"], truncation=True,
                                padding="max_length", max_length=128),
                batched=True)
    return ds

def metrics(ep):
    logits, labels = ep
    preds = np.argmax(logits, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds,
                                                 average="binary",
                                                 zero_division=0)
    return {"accuracy": accuracy_score(labels, preds),
            "precision": p, "recall": r, "f1": f}

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

train = to_ds(pd.read_csv("data/clf_train.csv"), tok)
val   = to_ds(pd.read_csv("data/clf_val.csv"),   tok)

args = TrainingArguments(
    output_dir="models/domain-classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=2e-5,
)

Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
        compute_metrics=metrics).train()
model.save_pretrained("models/domain-classifier")
tok.save_pretrained("models/domain-classifier")