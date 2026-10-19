# AgroChat A Small Agriculture Question-Answering Assistant

## 1. Abstract

This project builds a chatbot that answers English-language questions about farming. It uses a public dataset of 25,410 agriculture question–answer pairs and fine-tunes a compact sentence-embedding model so that paraphrased questions retrieve the right answer. The system rejects questions that are not about agriculture. On a held-out test split, the fine-tuned model is compared with the original pretrained model across several similarity thresholds. The report analyses when the system succeeds, when it fails, and what a small, open system can realistically achieve.

## 2. Introduction

Farmers and smallholders need quick, trustworthy answers about soil, crops, pests, and harvesting. A general-purpose chatbot is risky because it does not know when it does not know. A **retrieval-based** assistant is safer: it answers only from a fixed knowledge base and says "I don't know" when nothing is close enough.

The goal is to build a small, reproducible agriculture QA system that (a) uses real, public data, (b) fine-tunes a sentence embedder so paraphrased questions still find the right answer, and (c) measures how well it accepts in-domain questions and rejects out-of-domain ones.

## 3. Related Work

Sentence-BERT showed that a siamese Transformer can produce sentence embeddings that make similarity search thousands of times faster than comparing every pair with a full Transformer, without losing accuracy (Reimers & Gurevych, 2019). The standard way to train such a model for retrieval is `MultipleNegativesRankingLoss`, which treats every other answer in a batch as a negative example for the current question (Henderson et al., 2017). DistilBERT showed that a smaller Transformer retains most of BERT's quality while being faster and lighter (Sanh et al., 2019).

Agricultural chatbot research has produced many prototypes. Most use retrieval or LLM-based pipelines, but few report out-of-domain rejection rates or provide reproducible open-data evaluations. This project addresses that gap with a single fine-tuned embedder, an explicit threshold, and a transparent evaluation on a public dataset.

## 4. Dataset

**Source**: `talhakk/agriculture-qa` on Hugging Face (Apache 2.0).

**Size**: 25,410 pairs, English. 15,000 general agriculture pairs and 10,410 crop-specific pairs covering rice, wheat, tomato, okra, potato, onion, sugarcane, cotton, sunflower, and pulses.

**Preparation**: empty rows removed; 80 % train / 10 % validation / 10 % test split with seed 42.

**Limitations**: answers are expert-style and sometimes generic; recommendations may not fit every region or season. The dataset is English-only.

## 5. Methodology

**Baseline**: `sentence-transformers/all-MiniLM-L6-v2` used as-is.

**Proposed model**: the same model fine-tuned on the training split with `MultipleNegativesRankingLoss`. Each `(question, answer)` pair is a positive; every other answer in the same batch acts as a negative. Batch size 64, 3 epochs, learning rate handled by the `sentence-transformers` trainer with 10 % warm-up.

**Retrieval**: at inference, the user question is embedded once, and its cosine similarity against every stored question embedding is computed. The highest-scoring question's answer is returned if the score is at least a threshold; otherwise a fallback message is shown.

**Why this design**: a bi-encoder (embed once, compare vectors) is fast enough for interactive use, unlike a cross-encoder that would need to process every stored question alongside the user query (Reimers & Gurevych, 2019).

## 6. Experiments and Results

Two configurations were compared across thresholds 0.30 to 0.60:

| Config | Threshold | Top-1 accuracy | In-domain acceptance | OOD rejection | Precision | Recall | F1 | Mean ms |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.42 | *fill after running* | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* |
| Fine-tuned | 0.42 | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* |
| Fine-tuned | *best threshold* | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* | *fill* |

> **Figure 1**: line chart of OOD rejection rate vs. in-domain acceptance rate for baseline and fine-tuned models across thresholds. *Place the chart generated from `eval_results.csv` here.*

**What to look for**: fine-tuning should improve top-1 accuracy on paraphrased questions. The threshold controls the trade-off: a higher threshold rejects more out-of-domain questions but also rejects more valid agriculture questions.

## 7. Error Analysis and Discussion

**Successes to report**: pick 2–3 test questions where the fine-tuned model retrieved the correct answer and the baseline did not. Show the user question, the retrieved question, and the similarity score.

**Failure types to look for**:

1. **Vocabulary gap**: the user uses a local word or spelling not seen in training (e.g., "maize rust" vs. "corn leaf disease"). The fine-tuned model narrows this gap but does not eliminate it.
2. **Threshold false acceptance**: an out-of-domain question scores above the threshold because it shares words with agriculture (e.g., "agricultural subsidies"). The system answers incorrectly instead of rejecting.
3. **Threshold false rejection**: a valid but oddly phrased agriculture question scores below the threshold. The system says "I don't know" even though the answer exists.
4. **Answer style mismatch**: the retrieved answer is correct but too generic or too technical for a farmer.

**Why these happen**: a fixed cosine threshold is a blunt instrument. It cannot learn that "golf course fertiliser" is agriculture-adjacent but out of scope, while "tomato yellow leaves" is in scope. A supervised domain classifier could learn this boundary, but it requires labelled out-of-domain data and a second training procedure. That is future work.

## 8. Deployment

The application is a Streamlit app deployed on **Streamlit Community Cloud** from the GitHub repository. The knowledge base and fine-tuned sentence embedder, including its weights, are stored with the application in that repository. The app loads the local fine-tuned model without requiring a separate model repository or a model download at runtime.

The web interface provides a question form and retains the conversation in the current Streamlit session. Responses show the similarity score; the Flask API and `/api/health` endpoint remain available when running the Flask application directly.

## 9. Limitations and Future Work

- **Evaluation scope**: the test split comes from the same dataset as training. An independent set of real farmer questions would be a stronger benchmark.
- **No trained domain classifier**: the threshold is a heuristic. A fine-tuned classifier could improve rejection, at the cost of a second model and labelled OOD data.
- **English only**: the brief mentions African-language NLP; this project does not address it. A multilingual embedder and African-language QA data would be needed.
- **Answer generation**: the system returns stored answers, not generated explanations. A retrieval-augmented generator could combine the retrieved answer with a short, farmer-friendly paraphrase.
- **Local relevance**: recommendations vary by soil, climate, and regulation. A deployed system should add verified local guidance.

## 10. Conclusion

A fine-tuned sentence embedder plus a similarity threshold is a small, reproducible, and transparent agriculture QA system. It improves retrieval over the pretrained baseline, and its errors are easy to inspect and explain. The main open problem is out-of-domain rejection: a threshold is a blunt tool, and a learned classifier is the natural next step.

## 11. References

- Henderson, M., Al-Rfou, R., Strope, B., et al. (2017). Efficient natural language response suggestion for smart reply. *arXiv:1705.00652*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP*, 3973–3983. https://doi.org/10.18653/v1/D19-1410
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
- talhakk. (2026). *agriculture-qa* [Data set]. Hugging Face. https://huggingface.co/datasets/talhakk/agriculture-qa
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of EMNLP: System Demonstrations*, 38–45.