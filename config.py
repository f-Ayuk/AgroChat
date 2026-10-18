
"""Central configuration for ChatAgro.

Contains the embedding model, similarity threshold,
and fallback response used by app.py.
"""

from pathlib import Path

# Project directory containing app.py and config.py.
BASE_DIR = Path(__file__).resolve().parent

# Fine-tuned model directory.
FINETUNED_MODEL_PATH = BASE_DIR / "models" / "agri-minilm"

# Use the fine-tuned model when its directory exists.
# Otherwise, fall back to the pretrained MiniLM model.
if FINETUNED_MODEL_PATH.is_dir():
    MODEL_NAME = str(FINETUNED_MODEL_PATH)
else:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum cosine similarity required to accept a retrieved answer.
# Keep 0.45 only if it matches your chosen validation threshold.
SIMILARITY_THRESHOLD = 0.45

# Response when no sufficiently relevant answer is found.
FALLBACK = (
    "Sorry, I couldn't find a reliable answer to your question. "
    "Please try rephrasing it or asking another agriculture-related question."
)