
"""Central configuration for ChatAgro.

Contains the embedding model, similarity threshold,
and fallback response used by app.py.
"""

from pathlib import Path

# Project directory containing app.py and config.py.
BASE_DIR = Path(__file__).resolve().parent

# Fine-tuned model directory.
FINETUNED_MODEL_PATH = BASE_DIR / "models" / "agri-minilm"

# Ignore the model folder unless it contains weights; the repository may
# intentionally omit these large files because they are stored separately.
has_local_weights = any(FINETUNED_MODEL_PATH.glob("*.safetensors")) or (
    FINETUNED_MODEL_PATH / "pytorch_model.bin"
).is_file()

# Use the fine-tuned model when local weights are available.
# Otherwise, fall back to the pretrained MiniLM model.
if FINETUNED_MODEL_PATH.is_dir() and has_local_weights:
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