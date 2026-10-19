# AgroChat — Agriculture Question Answering

AgroChat answers English-language questions about farming: soil, irrigation, crop health, pests, fertilisers, harvesting, storage, seeds, and weeds. It retrieves the closest answer from a curated agriculture knowledge base and uses a fallback when no result is sufficiently similar.

## How it works

1. A sentence embedder turns the user question into a vector.
2. The vector is compared with stored agriculture questions using cosine similarity.
3. If the best score is above the configured threshold, the associated answer is returned.
4. Otherwise, AgroChat returns a fallback message.

The app runs on **Streamlit Community Cloud**. The Streamlit app, knowledge-base CSV, and fine-tuned model files are all in this GitHub repository. No separate model-hosting account or model repository is needed.

## Dataset

- **Source**: `talhakk/agriculture-qa` (Apache 2.0).
- **Size**: 25,410 question–answer pairs; English.
- **Split**: 80% train / 10% validation / 10% test, seed 42.
- **Content**: general agriculture and crop-specific QA for rice, wheat, tomato, okra, potato, onion, sugarcane, cotton, sunflower, and pulses.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run streamlit_app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

To train the model or reproduce the evaluation:

```powershell
py scripts/prepare_data.py
py scripts/train_embedder.py --output models/agri-minilm
py evaluate.py
py -m pytest -v
```

The preparation step downloads the dataset and writes train, validation, and test splits. The deployed app needs the committed `data/qa_train.csv` file and the complete `models/agri-minilm` directory, including `model.safetensors`.

## Deploy on Streamlit Community Cloud

1. Push the project to the GitHub repository `f-Ayuk/AgroChat`. Make sure `data/qa_train.csv` and all files under `models/agri-minilm/`, including `model.safetensors`, are committed and pushed. The model file is about 91 MB, below GitHub's 100 MB per-file limit.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from the `f-Ayuk/AgroChat` GitHub repository.
3. Select the `main` branch and set **Main file path** to `streamlit_app.py`.
4. Click **Deploy**. On first startup, Streamlit installs the requirements and encodes the knowledge-base questions. Allow several minutes for initialization.

No Streamlit secrets are required for the model: the app automatically loads the committed fine-tuned model from `models/agri-minilm`. If you want to explicitly set the same local directory in **Settings → Secrets**, use:

```toml
EMBEDDER_PATH = "models/agri-minilm"
```

An optional threshold override is:

```toml
SIMILARITY_THRESHOLD = "0.45"
```

The configured `EMBEDDER_PATH` must name a model directory inside the app's GitHub checkout. It is resolved relative to the project root. If the directory is missing, startup fails with an explicit error rather than trying to download a model.

## Project structure

```
├── app.py                  # Flask API and retrieval pipeline
├── streamlit_app.py        # Streamlit user interface
├── config.py               # shared model and retrieval settings
├── evaluate.py             # baseline vs. fine-tuned threshold evaluation
├── test_app.py             # application tests
├── scripts/
│   ├── prepare_data.py     # prepare train/validation/test CSVs
│   └── train_embedder.py   # fine-tune the sentence embedder
├── data/
│   └── qa_train.csv        # deployed knowledge base
├── models/agri-minilm/     # fine-tuned model and weights, committed to GitHub
├── templates/              # Flask page and shared branding partial
├── static/                 # shared stylesheet and Flask chat script
└── requirements.txt
```

Both frontends use `templates/_branding.html` for the header and welcome
section, and Streamlit applies the shared `static/style.css`. Streamlit-native
form controls submit questions directly to the shared Python retrieval
function. `templates/index.html` and `static/style.js` provide the separate
Flask page and its browser-side API calls; Streamlit does not execute that
Flask-specific JavaScript.

## Limitations

- **Small comparison set**: evaluation uses a held-out slice of the source dataset, not an independently curated farmer-query benchmark.
- **Threshold is a heuristic**: a fixed cosine cut-off cannot separate every agriculture-adjacent question from every out-of-domain question.
- **Answer style**: source answers can be generic; local agricultural guidance should be consulted.
- **English only**: the system does not handle other languages.
- **Advice disclaimer**: educational information only. Confirm pesticide and fertiliser decisions with a local agricultural adviser.

## References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP*, 3973–3983. https://doi.org/10.18653/v1/D19-1410
- Henderson, M., et al. (2017). Efficient natural language response suggestion for smart reply. *arXiv:1705.00652*.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
- talhakk/agriculture-qa dataset. https://huggingface.co/datasets/talhakk/agriculture-qa
