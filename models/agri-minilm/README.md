---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:20328
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: control of stem rot in ash gourd
  sentences:
  - spray blitox 50 @ 2 gm per liter of water
  - advised him to irrigation should be donw
  - advised him to visit dist agril office
- source_sentence: control of fruit borer in french bean
  sentences:
  - advice to spray bavistin @ 2 g/lit of water
  - spray metador @ 2 ml per liter of water
  - adviced to apply malathion 5% dust powder
- source_sentence: want to know about weedicide in paddy
  sentences:
  - told him spray rogor 2ml/lit of water
  - suggested him to time of sowing may- july
  - adviced to go for manual weeding in paddy
- source_sentence: asking about control viral disese in okra
  sentences:
  - lotus is the good broiler for farming.
  - advice to apply tufgor @ 2 ml / litre
  - advice to apply rogor @ 2ml/l of water
- source_sentence: he has problem with pest in tomato
  sentences:
  - recommended to plant in mar - june month
  - advice to spray bavistin @ 2 g/lit of water
  - advice him to spray malathion 50 ec @2ml/litre
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: agri val
      type: agri-val
    metrics:
    - type: cosine_accuracy@1
      value: 0.2939787485242031
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.4458874458874459
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.5301062573789846
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.6324281778827233
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.2939787485242031
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.1486291486291486
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.10602125147579693
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.06324281778827233
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.2939787485242031
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.4458874458874459
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.5301062573789846
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.6324281778827233
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.44969077355088805
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.3928191937932204
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.40482048956893607
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
queries = [
    'he has problem with pest in tomato',
]
documents = [
    'advice him to spray malathion 50 ec @2ml/litre',
    'recommended to plant in mar - june month',
    'advice to spray bavistin @ 2 g/lit of water',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, 384] [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[ 0.4328, -0.0615,  0.1580]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `agri-val`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.294      |
| cosine_accuracy@3   | 0.4459     |
| cosine_accuracy@5   | 0.5301     |
| cosine_accuracy@10  | 0.6324     |
| cosine_precision@1  | 0.294      |
| cosine_precision@3  | 0.1486     |
| cosine_precision@5  | 0.106      |
| cosine_precision@10 | 0.0632     |
| cosine_recall@1     | 0.294      |
| cosine_recall@3     | 0.4459     |
| cosine_recall@5     | 0.5301     |
| cosine_recall@10    | 0.6324     |
| **cosine_ndcg@10**  | **0.4497** |
| cosine_mrr@10       | 0.3928     |
| cosine_map@100      | 0.4048     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 20,328 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                          |
  |:---------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                              |
  | modality | text                                                                              | text                                                                                |
  | details  | <ul><li>min: 9 tokens</li><li>mean: 14.41 tokens</li><li>max: 47 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 30.86 tokens</li><li>max: 182 tokens</li></ul> |
* Samples:
  | sentence_0                                                   | sentence_1                                                                               |
  |:-------------------------------------------------------------|:-----------------------------------------------------------------------------------------|
  | <code>asking about the seed rate of bhindi.</code>           | <code>advised him to take 3 kg / bigha seeds for bhindi.</code>                          |
  | <code>Name regions in which Pastoralism is practised.</code> | <code>arid and semi-arid regions of Sahara, Central Asia and some parts of India.</code> |
  | <code>asking control measure of damping off in tomato</code> | <code>apply captan @ 2 gram per litre of water</code>                                    |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 64
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 64
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss | agri-val_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:-----------------------:|
| 1.0    | 318  | -             | 0.3900                  |
| 1.5723 | 500  | 1.5689        | -                       |
| 2.0    | 636  | -             | 0.4325                  |
| 3.0    | 954  | -             | 0.4497                  |


### Training Time
- **Training**: 2.8 hours

### Framework Versions
- Python: 3.12.4
- Sentence Transformers: 6.1.0
- Transformers: 5.19.0
- PyTorch: 2.14.1+cpu
- Accelerate: 1.15.0
- Datasets: 5.1.0
- Tokenizers: 0.23.3

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->