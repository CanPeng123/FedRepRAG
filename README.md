# FedRepRAG

**Representation-based Federated Retrieval-Augmented Generation**

[Paper on arXiv](https://arxiv.org/abs/2609.22162)

FedRepRAG is a federated learning framework for retrieval-augmented question answering that supports VQA and QA tasks. It trains a lightweight MLP projector that injects retrieved knowledge representations into a frozen VLM or LLM, enabling federated RAG that protects raw context across heterogeneous clients.

---

## Overview

FedRepRAG answers questions using evidence from documents held by multiple clients. At inference time, it selects the top-K documents across clients and uses a shared MLP projector to convert their retrieval embeddings into input tokens for a frozen LLM or VLM. The source documents stay on their respective clients.

### Two-Stage MLP Projector Training

Only the projector is trained. The retrieval encoder and the LLM or VLM remain frozen throughout both stages.

- **Stage 1 — Semantic alignment.** Each client trains the projector on its local documents. Given a document’s projected embedding and an instruction prompt, the backbone reconstructs the source document token by token. 
- **Stage 2 — Task-specific alignment.** Starting from the global Stage 1 projector, each client retrieves the top-K document embeddings from its local corpus for each labeled question. The backbone receives the projected embeddings together with the original query, and the client updates the projector using cross-entropy loss on the reference answer tokens.

In both stages, the server aggregates the clients’ updated projector weights after each round, weighting each client by the number of local training examples it used in that stage.

---

## Datasets

Download the datasets using the links below and place them under `datasets/`.

| Dataset | Task | Domain | Model |
|---|---|---|---|
| [SLAKE](https://www.med-vqa.com/slake/) | VQA | Medical imaging | Qwen2.5-VL-7B |
| [ScienceQA](https://scienceqa.github.io/) | VQA | Science | Qwen2.5-VL-7B |
| [PubMedQA](https://pubmedqa.github.io/) | Text QA | Biomedical question answering | Qwen3-8B |
| [KILT-NQ](https://github.com/facebookresearch/KILT) | Text QA | Open-domain | Qwen3-8B |

---

## Requirements

```bash
pip install torch "transformers>=4.51.0" qwen-vl-utils faiss-cpu deepspeed pandas pyyaml tqdm tensorboard
```

---

## Running Experiments

### Interactive menu

```bash
bash run.sh
```

The menu will guide you through selecting:
1. Dataset (SLAKE / ScienceQA / PubMedQA / KILT-NQ)
2. Setting (Centralized / Federated)
3. Mode (Without RAG / Get embeddings / In-context RAG / MLP training / MLP inference)

---

### Command line

Set `DATASET` to one of `slake`, `scienceqa`, `pubmedqa`, or `kilt_nq`.

```bash
DATASET=pubmedqa
CONFIG="configs/${DATASET}.yaml"

# Build retrieval embeddings
bash run.sh --config "$CONFIG" --stage get_embedding

# Run in-context RAG
bash run.sh --config "$CONFIG" --stage incontext

# Train the federated projector on prepared client data
bash run.sh --config "$CONFIG" --stage fl_mlp_train

# Run projector inference with the checkpoint saved during training
bash run.sh --config "$CONFIG" --stage mlp_infer --checkpoint path/to/XXX.pt
```
