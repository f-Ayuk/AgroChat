
"""Central configuration for ChatAgro.

Contains the embedding model, similarity threshold,
and fallback response used by app.py.
"""

from pathlib import Path

# Project directory containing app.py and config.py.
BASE_DIR = Path(__file__).resolve().parent

# Fine-tuned model directory.
FINETUNED_MODEL_PATH = BASE_DIR / "models" / "agri-minilm"

# The complete fine-tuned model is committed with the application.
has_local_weights = any(FINETUNED_MODEL_PATH.glob("*.safetensors")) or (
    FINETUNED_MODEL_PATH / "pytorch_model.bin"
).is_file()

if not FINETUNED_MODEL_PATH.is_dir() or not has_local_weights:
    raise FileNotFoundError(
        f"Fine-tuned model weights not found in {FINETUNED_MODEL_PATH}. "
        "Ensure the complete models/agri-minilm directory is present."
    )

MODEL_NAME = str(FINETUNED_MODEL_PATH)

# Minimum cosine similarity required to accept a retrieved answer.
# Keep 0.45 only if it matches your chosen validation threshold.
SIMILARITY_THRESHOLD = 0.45

# Response when no sufficiently relevant answer is found.
FALLBACK = (
    "Sorry, I couldn't find a reliable answer to your question. "
    "Please try rephrasing it or asking another agriculture-related question."
)