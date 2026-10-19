# AgroChat fine-tuned sentence encoder

This directory contains the local SentenceTransformer used by AgroChat. The
Streamlit and Flask applications load these files from the GitHub checkout;
they do not download this fine-tuned model from a separate model-hosting
service at runtime.

## Model

- **Architecture:** Transformer encoder, mean pooling, then L2 normalization.
- **Base encoder:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector size:** 384 dimensions.
- **Maximum input length:** 256 tokens.
- **Training objective:** `MultipleNegativesRankingLoss` on agriculture
  question-answer pairs.
- **Training script:** `scripts/train_embedder.py`.
- **Data source/preparation:** `scripts/prepare_data.py`.

The base encoder is an external pretrained resource; its model card is cited
for provenance and architecture, not as a deployment location:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Reproducibility notes

The checked-in training script uses three epochs, batch size 64, and warm-up
steps equal to 10% of total training steps. It does not explicitly set the
learning rate or random seeds, and dependency versions are not pinned in the
project requirements. For that reason, a retrained model may not be bitwise
identical to the weights stored here.

The validation evaluator in the training script compares validation
questions against validation answers, including each question's paired
answer. Its generated retrieval metrics are therefore not a clean measure of
the deployed app's behavior and are not used as the final test results in
`report.md`. The corrected app-aligned comparison is produced by
`evaluate.py`.

## Local loading

From the project root, load the model with:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/agri-minilm")
```

The complete model directory, including its weight file, must be present in
the repository checkout.
