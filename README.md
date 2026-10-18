For a Hugging Face Docker Space, start the README with this configuration:

---
title: AgroChat
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# AgroChat — Agriculture Question Answering

AgroChat answers English-language questions about farming: soil, irrigation, crop health, pests, fertilisers, harvesting, storage, seeds, and weeds. It retrieves the closest answer from a curated agriculture knowledge base, and rejects questions that are not about agriculture.

## Live demo

- Web app: **`<paste your Hugging Face Space URL here>`**
- Health check: `GET /api/health`

## How it works

1. A **sentence embedder** turns your question into a vector.
2. That vector is compared with every stored question vector using **cosine similarity**.
3. If the best score is above a **threshold**, the matching answer is returned.
4. Otherwise, a fallback message explains that the question is outside the agriculture knowledge base.

The embedder is **fine-tuned** on 25,410 agriculture QA pairs from the public [`talhakk/agriculture-qa`](https://huggingface.co/datasets/talhakk/agriculture-qa) dataset. This fine-tuning teaches the model that farmer phrasing ("wilt", "yellow leaves") is close in meaning to stored answers ("water stress", "nutrient deficiency"), even when the wording is different.

## Dataset

- **Source**: `talhakk/agriculture-qa` (Hugging Face, Apache 2.0).
- **Size**: 25,410 question–answer pairs; English.
- **Split**: 80 % train / 10 % validation / 10 % test, seed 42.
- **Content**: general agriculture (irrigation, fertilisation, soil science, climate, sustainable farming) and crop-specific QA for rice, wheat, tomato, okra, potato, onion, sugarcane, cotton, sunflower, and pulses.

## Reproducing the project

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download data and build splits
python scripts/prepare_data.py

# 2. Fine-tune the embedder (CPU is fine; ~15–30 min)
python scripts/train_embedder.py --output models/agri-minilm

# 3. Evaluate baseline vs. fine-tuned across thresholds
python evaluate.py

# 4. Run tests
pytest -v

# 5. Start the web app
python app.py            # http://127.0.0.1:7860
```

If `models/agri-minilm` is missing, `app.py` falls back to the pretrained `all-MiniLM-L6-v2`, so the site still runs — but with lower retrieval accuracy. The `/api/health` endpoint reports which embedder is active.

## Project structure

```
├── app.py                  # Flask app + retrieval pipeline
├── config.py               # shared constants
├── evaluate.py             # baseline vs. fine-tuned × threshold sweep
├── test_app.py             # pytest suite
├── scripts/
│   ├── prepare_data.py     # talhakk → train/val/test CSVs
│   └── train_embedder.py   # fine-tune with MultipleNegativesRankingLoss
├── data/
│   ├── qa_train.csv
│   ├── qa_val.csv
│   ├── qa_test.csv
│   └── qa_curated_diagnostic.csv
├── models/                 # gitignored; produced by train_embedder.py
├── templates/index.html
├── static/{style.css,style.js}
├── Dockerfile
└── requirements.txt
```

## Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space). Choose **Docker** as the SDK and **Blank** as the template.
2. Push this repository to the Space repository.
3. The Space will build from the `Dockerfile` and serve the Flask app on port **7860**.
4. To use the fine-tuned model, either:
    - run `train_embedder.py` locally, then upload `models/agri-minilm` to the Space (remove it from `.gitignore` first), or
    - push the model to the Hugging Face Hub and set the Space variable `EMBEDDER_PATH=your-username/agri-minilm`.
5. Set the variable `SIMILARITY_THRESHOLD` in the Space settings if you tuned a different threshold.

Render's free tier provides 512 MB RAM, which is **not enough** for PyTorch and the sentence embedder. Use Hugging Face Spaces for the free Docker deployment.

## Limitations

- **Small comparison set**: the evaluation uses a 10 % held-out slice of talhakk, not an independently curated farmer-query benchmark.
- **Threshold is a heuristic**: a fixed cosine cut-off cannot separate every agriculture-adjacent question from every out-of-domain question. A supervised domain classifier is discussed as future work.
- **Answer style**: talhakk answers are expert-style and sometimes generic; a deployed system should add local extension guidance.
- **English only**: the system does not handle other languages.
- **Advice disclaimer**: educational information only. Confirm pesticide and fertiliser decisions with a local agricultural adviser.

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP*, 3973–3983. https://doi.org/10.18653/v1/D19-1410
- Henderson, M., et al. (2017). Efficient natural language response suggestion for smart reply. *arXiv:1705.00652*.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
- talhakk/agriculture-qa dataset. https://huggingface.co/datasets/talhakk/agriculture-qa