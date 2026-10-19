"""AgroChat: agriculture QA with a fine-tuned sentence embedder.

Pipeline:
    1. Embed the user question with the fine-tuned (or baseline) encoder.
    2. Compute cosine similarity against every stored question embedding.
    3. Return the best answer if similarity >= threshold; otherwise fallback.
"""
import csv
import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

from config import FALLBACK, MODEL_NAME, SIMILARITY_THRESHOLD as _DEFAULT_TH

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "data" / "qa_train.csv"  # retrieved answers come from training split

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", str(_DEFAULT_TH)))

app = Flask(__name__)


# ------------------------------------------------------------------ knowledge
def load_knowledge(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_data.py` first."
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not r.get("question") or not r.get("answer"):
            raise ValueError("Every row must have a question and an answer.")
    return rows


knowledge = load_knowledge(KNOWLEDGE_FILE)
QUESTIONS = [r["question"] for r in knowledge]


# ------------------------------------------------------------------ model
def _resolve_embedder() -> tuple[SentenceTransformer, str]:
    configured_model = os.getenv("EMBEDDER_PATH")
    if configured_model:
        model_path = Path(configured_model)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Configured embedder directory not found: {model_path}"
            )
        return (
            SentenceTransformer(str(model_path)),
            f"finetuned:{model_path}",
        )

    model_path = Path(MODEL_NAME)
    tag = (
        f"finetuned:{MODEL_NAME}"
        if model_path.is_dir()
        else f"baseline:{MODEL_NAME}"
    )
    return SentenceTransformer(MODEL_NAME), tag


model, EMBEDDER_TAG = _resolve_embedder()
question_embeddings = model.encode(
    QUESTIONS, normalize_embeddings=True, convert_to_numpy=True
)


# ------------------------------------------------------------------ inference
def retrieve_answer(user_question: str) -> dict:
    start = time.perf_counter()

    query_embedding = model.encode(
        [user_question], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    scores = question_embeddings @ query_embedding
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer": FALLBACK,
            "topic": None,
            "matched_question": None,
            "similarity": round(best_score, 4),
            "accepted": False,
            "response_time_ms": elapsed_ms,
        }

    row = knowledge[best_index]
    return {
        "answer": row["answer"],
        "topic": row.get("category", "agriculture"),
        "matched_question": row["question"],
        "similarity": round(best_score, 4),
        "accepted": True,
        "response_time_ms": elapsed_ms,
    }


# ------------------------------------------------------------------ routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    question = request.form.get("question", "").strip()
    if not question:
        return render_template(
            "index.html", initial_error="Please enter a question."
        ), 400
    if len(question) > 500:
        return render_template(
            "index.html",
            initial_error="Please keep questions under 500 characters.",
        ), 400

    return render_template(
        "index.html",
        initial_question=question,
        initial_response=retrieve_answer(question),
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "knowledge_base_size": len(knowledge),
        "embedder": EMBEDDER_TAG,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not isinstance(question, str):
        return jsonify({"error": "Question must be text."}), 400

    question = question.strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if len(question) > 500:
        return jsonify({
            "error": "Please keep questions under 500 characters."
        }), 400

    return jsonify(retrieve_answer(question))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")))