# AgroChat — Agriculture Question Answering

AgroChat answers English-language questions about farming: soil, irrigation, crop health, pests, fertilisers, harvesting, storage, seeds, and weeds. It retrieves the closest answer from a curated agriculture knowledge base and uses a fallback when no result is sufficiently similar.

## How it works

1. A sentence embedder turns the user question into a vector.
2. The vector is compared with stored agriculture questions using cosine similarity.
3. If the best score is above the configured threshold, the associated answer is returned.
4. Otherwise, AgroChat returns a fallback message.

The application runs on **Streamlit Community Cloud**. The fine-tuned model weights are stored separately from the application source; the app loads them from a model repository when configured. If no fine-tuned model is configured, it uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` model.

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

To train the model or reproduce the evaluation, first install the project requirements and prepare the data:

```powershell
py scripts/prepare_data.py
py scripts/train_embedder.py --output models/agri-minilm
py evaluate.py
py -m pytest -v
```

The preparation step downloads the dataset and writes the train, validation, and test splits. If you only need to run the app, `data/qa_train.csv` is already included.

## Deploy the app on Streamlit Community Cloud

### 1. Put the application code on GitHub

Push this project to a GitHub repository that you can access from Streamlit Community Cloud. Keep `data/qa_train.csv` in the repository; it is required at startup. The fine-tuned `*.safetensors` weights are ignored by `.gitignore` so that large model files do not get added to the app repository.

### 2. Store the fine-tuned weights separately (recommended)

If you have not trained the model yet, run `py scripts/train_embedder.py --output models/agri-minilm` locally. Then create a **model repository** in the Hugging Face Hub and upload the contents of `models/agri-minilm` to it, including the model configuration, tokenizer files, and weight files. This repository stores model files only; it does not host the AgroChat application.

Make the model repository public if you do not need access controls. Copy its repository ID, such as `your-account/agri-minilm`. For a private model repository, create a read token and keep it in Streamlit secrets; never commit the token to GitHub.

### 3. Create the Streamlit app

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create an app and connect the GitHub repository.
3. Select the branch and set **Main file path** to `streamlit_app.py`.
4. Deploy. The first startup installs dependencies, downloads the configured model, and encodes the knowledge-base questions for retrieval. It can take several minutes, depending on the available CPU and memory.

### 4. Configure app secrets

In the deployed app's **Settings → Secrets**, set the model repository ID:

```toml
EMBEDDER_PATH = "your-account/agri-minilm"
```

For a private model, also add a Hugging Face read token:

```toml
HF_TOKEN = "hf_your_read_token"
```

Do not add `HF_TOKEN` for a public model. You may optionally override the retrieval threshold:

```toml
SIMILARITY_THRESHOLD = "0.45"
```

The app reads these settings before loading the model. `EMBEDDER_PATH` may also be a local model directory when running locally. If it is unset, the app uses local fine-tuned weights when available, otherwise it falls back to the pretrained MiniLM model.

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
├── models/agri-minilm/     # model config; large weights stored separately
├── templates/ and static/  # Flask interface assets
└── requirements.txt
```

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
