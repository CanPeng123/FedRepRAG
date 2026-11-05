import torch
import numpy as np
import faiss
import pathlib
from typing import List, Dict, Any, Tuple
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info


# --------------------------
# ---- Retrieval utils ----
# --------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return x / norms


def build_faiss_ip_index(train_emb: np.ndarray) -> faiss.Index:
    """Inner-product index; assumes embeddings are already normalized."""
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb)
    return index


def build_client_retrieval_pools(args):
    """
    Load one local retrieval bank per client.

    Expected structure:
        args.client_embeddings_dir/
            client_0/
                embeddings_train.pt
            client_1/
                embeddings_train.pt
            ...

    Returns:
        client_retrieval_pools: list[dict]
            each dict contains:
                - train_emb_normed
                - train_qids
                - qid_to_row
                - index
    """
    client_dirs = sorted(
        [
            p for p in pathlib.Path(args.client_embeddings_dir).glob("client_*")
            if p.is_dir()
        ],
        key=lambda p: p.name
    )

    if len(client_dirs) == 0:
        raise ValueError(f"No client_* directories found in {args.client_embeddings_dir}")

    client_retrieval_pools = []

    for client_dir in client_dirs:
        # support both naming conventions
        emb_path = client_dir / "embeddings_train.pt"
        if not emb_path.exists():
            emb_path = client_dir / "slake_embeddings_train.pt"
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found in {client_dir} (tried embeddings_train.pt and slake_embeddings_train.pt)")

        ckpt = torch.load(emb_path, map_location="cpu")
        train_emb = ckpt["embeddings"].cpu().to(torch.float32).numpy().astype("float32")
        train_qids = [str(q) for q in ckpt["qids"]]

        N, D_emb = train_emb.shape
        print(f"[Emb][{client_dir.name}] Loaded {N} embeddings (dim={D_emb}).")

        train_emb_normed = l2_normalize(train_emb)
        index = build_faiss_ip_index(train_emb_normed)
        print(f"[FAISS][{client_dir.name}] Built IP index with {index.ntotal} vectors.")

        qid_to_row = {int(qid): i for i, qid in enumerate(train_qids)}

        client_retrieval_pools.append({
            "client_name": client_dir.name,
            "train_emb_normed": train_emb_normed,
            "train_qids": train_qids,
            "qid_to_row": qid_to_row,
            "index": index,
        })

    return client_retrieval_pools
# ----------------------------------------------------------------
# ---- generate retrival embedding message ----
# ----------------------------------------------------------------
def embed_messages(messages: list[dict], model, processor, device) -> np.ndarray:
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    base_model_inner = model.module if hasattr(model, "module") else model
    _is_gemma = "gemma" in type(base_model_inner).__name__.lower()

    if _is_gemma:
        inputs = processor(
            text=texts,
            images=image_inputs or None,
            padding=True,
            return_tensors="pt"
        ).to(device)
    else:
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

    model.eval()
    with torch.no_grad():
        # compute embeddings in fp32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.float()

            outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            hidden = outputs.hidden_states[-1].float()  # [B,T,D]

            attn = inputs["attention_mask"].unsqueeze(-1).float()  # [B,T,1]
            lengths = attn.sum(dim=1)  # [B,1]

            summed = (hidden * attn).sum(dim=1)  # [B,D]
            emb = summed / lengths.clamp(min=1.0)  # if length==0 -> keep zeros

            # safe L2 normalize
            norm = emb.norm(p=2, dim=1, keepdim=True)
            emb = emb / norm.clamp(min=1e-6)

    emb = emb.cpu().numpy().astype("float32", copy=False)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)  # extra safety
    return emb


def embed_single_message_tokens(message: dict, model, processor, device, layer: str = "embeddings", 
strip_padding: bool = False, average_hidden: bool = False, l2_normalize: bool = False, 
add_generation_prompt: bool = True) -> torch.Tensor:
    """
    Turn a single chat `message` (with text + optional images) into either:
      - a sequence of token-like embeddings [T_valid, H]   (average_hidden=False), or
      - a single pooled embedding           [H]           (average_hidden=True).

    - message: a single chat message, e.g. from build_retrieval_augmented_message(...)
    - layer:
        "embeddings" -> use embedding layer outputs (hidden_states[0])
        "first"      -> first transformer layer outputs (hidden_states[1])
        "last"       -> last transformer layer outputs (hidden_states[-1])
    - strip_padding:
        if True, remove padding positions via attention_mask (for sequence output).
    - average_hidden:
        if True, return a masked mean over tokens -> [H] instead of [T_valid, H].
    - l2_normalize:
        if True and average_hidden=True, L2-normalize the pooled vector.
    """

    # 1) Build chat template text just like in your other code
    text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=add_generation_prompt
    )

    # 2) Process images/videos for this message
    base_model_inner = model.module if hasattr(model, "module") else model
    _is_gemma = "gemma" in type(base_model_inner).__name__.lower()

    # 3) Tokenize + pack into model inputs
    if _is_gemma:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
    else:
        image_inputs, video_inputs = process_vision_info([message])
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

    # 4) Pick which layer to use as "token embeddings"
    if layer == "embeddings":
        # hidden_states[0]: output of embedding layer (text+vision mixed)
        hidden = outputs.hidden_states[0][0]   # [T, H]
    elif layer == "first":
        hidden = outputs.hidden_states[1][0]   # [T, H]
    else:  # "last"
        hidden = outputs.hidden_states[-1][0]  # [T, H]

    # ---- Case A: we want a pooled embedding [H] ----
    if average_hidden:
        if strip_padding:
            # Remove pad tokens first, then simple mean
            attn_mask = inputs.attention_mask[0].bool()   # [T]
            hidden_valid = hidden[attn_mask]              # [T_valid, H]
            if hidden_valid.size(0) == 0:
                pooled = hidden.mean(dim=0)               # fallback
            else:
                pooled = hidden_valid.mean(dim=0)         # [H]
        else:
            # Use attention mask for a proper masked mean
            attn = inputs.attention_mask[0].unsqueeze(-1).to(hidden.dtype)  # [T, 1]
            summed = (hidden * attn).sum(dim=0)           # [H]
            lengths = attn.sum(dim=0)                     # [1]
            pooled = summed / lengths.clamp(min=1e-6)     # [H]

        if l2_normalize:
            pooled = F.normalize(pooled, p=2, dim=0)

        return pooled  # [H]

    # ---- Case B: we want per-token embeddings [T_valid, H] (old behavior) ----
    if strip_padding:
        attn_mask = inputs.attention_mask[0].bool()  # [T]
        hidden = hidden[attn_mask]                   # [T_valid, H]

    return hidden  # [T_valid, H] or [T, H] if strip_padding=False


# ----------------------------------------------------------------
# ---- Retrieval function to return topK similarity ID  ----
# ----------------------------------------------------------------
def retrieve_neighbors_for_batch(batch_tasks, train_emb_normed, train_qids, qid_to_row, index, top_k: int, invalid_score_thr: float = -3e38):
    batch_neighbors_qids = []
    batch_neighbors_scores = []

    # one-time sanity (optional)
    # assert train_emb_normed.shape[1] == index.d

    for t in batch_tasks:
        qid = int(t["qid"])
        if qid not in qid_to_row:
            batch_neighbors_qids.append([])
            batch_neighbors_scores.append([])
            continue

        row_idx = qid_to_row[qid]
        query_vec = train_emb_normed[row_idx].reshape(1, -1).astype("float32", copy=False)

        # guard query vec
        if not np.isfinite(query_vec).all():
            batch_neighbors_qids.append([])
            batch_neighbors_scores.append([])
            continue

        scores, idxs = index.search(query_vec, top_k + 1)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        pairs = []
        for s, i in zip(scores, idxs):
            # invalid id
            if i < 0:
                continue
            if i >= len(train_qids):
                continue
            # invalid score
            if (not np.isfinite(s)) or (s <= invalid_score_thr):
                continue
            # drop self
            if i == row_idx:
                continue
            pairs.append((float(s), int(i)))

        # fallback if nothing usable
        if not pairs:
            batch_neighbors_qids.append([])
            batch_neighbors_scores.append([])
            continue

        pairs = pairs[:top_k]
        batch_neighbors_scores.append([s for s, _ in pairs])
        batch_neighbors_qids.append([train_qids[i] for _, i in pairs])

    return batch_neighbors_qids, batch_neighbors_scores


def retrieve_neighbors_for_batch_by_embedding(batch_messages, model, processor, device, faiss_index, train_qids, top_k: int):
    q_emb = embed_messages(messages=batch_messages, model=model, processor=processor, device=device)
    
    if isinstance(q_emb, torch.Tensor):
        q_emb = q_emb.detach().float().cpu().numpy()
    elif isinstance(q_emb, np.ndarray):
        q_emb = q_emb.astype("float32", copy=False)
    else:
        raise TypeError(f"embed_messages returned unsupported type: {type(q_emb)}")

    if not np.isfinite(q_emb).all():
        bad = np.where(~np.isfinite(q_emb))
        raise ValueError(f"q_emb has NaN/Inf at {bad}")

    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    if (norms < 1e-6).any():  # match your eps
        print("WARNING: near-zero query embedding norms:", norms.squeeze().tolist())

    q_emb = l2_normalize(q_emb)  # safe normalize

    # sanity checks
    assert q_emb.shape[1] == faiss_index.d, f"dim mismatch: q_emb {q_emb.shape[1]} vs index.d {faiss_index.d}"
    # optional: print once
    # print("index ntotal:", faiss_index.ntotal, "is_trained:", getattr(faiss_index, "is_trained", True))

    scores, idx = faiss_index.search(q_emb, top_k)

    batch_neighbors_qids, batch_neighbors_scores = [], []
    for b in range(idx.shape[0]):
        qids, scs = [], []
        seen = set()
        for j, s in zip(idx[b].tolist(), scores[b].tolist()):
            # filter invalid
            if j < 0:
                continue
            if j >= len(train_qids):
                continue
            if not np.isfinite(s):
                continue
            if s <= -3e38:  # FAISS sentinel guard
                continue
            qid = train_qids[j]
            if qid in seen:
                continue
            seen.add(qid)
            qids.append(qid)
            scs.append(float(s))

        batch_neighbors_qids.append(qids)
        batch_neighbors_scores.append(scs)

    return batch_neighbors_qids, batch_neighbors_scores


# ----------------------------------------------------------------
# ---- Build VLM last layer embedding for embedding retrival  ----
# ----------------------------------------------------------------
def build_message_from_text(text: str) -> dict:
    if not text:
        text = "."
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def build_prefix_last_layer_for_batch_from_retrieved_generic(
    batch_neighbors_qids,
    batch_neighbors_scores,
    train_task_map,
    build_message_fn,           # ex -> message dict
    model, processor, device,
    l2_normalize=True,
    output_mode="concat",
    k_concat=None,
    pad_value=0.0,
    override_texts=None,        # Optional[List[List[str]]]
):
    B = len(batch_neighbors_qids)
    assert B == len(batch_neighbors_scores), "qids and scores batch size mismatch"

    if override_texts is not None:
        assert len(override_texts) == B, f"override_texts batch mismatch: {len(override_texts)} vs B={B}"

    base_model = model.module if hasattr(model, "module") else model
    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)

    if output_mode == "concat":
        if k_concat is None:
            k_concat = max((len(x) for x in batch_neighbors_qids), default=0)
        if k_concat <= 0:
            raise ValueError("k_concat must be a positive int for output_mode='concat'")

    # Underlying HF model (unwrap Deepspeed if needed)
    base_model = model.module if hasattr(model, "module") else model
    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)

    pooled_embs = []

    for bi, (neighbor_qids, neighbor_scores) in enumerate(zip(batch_neighbors_qids, batch_neighbors_scores)):
        example_embs_list = []
        kept_scores = []

        for j, qid in enumerate(neighbor_qids):
            # choose message source
            if override_texts is not None:
                txt = override_texts[bi][j] if j < len(override_texts[bi]) else "."
                msg = build_message_from_text(txt)
            else:
                ex = train_task_map.get(str(qid))
                if ex is None:
                    continue
                msg = build_message_fn(ex)

            score = neighbor_scores[j] if j < len(neighbor_scores) else 0.0
            kept_scores.append(float(score))

            neigh_emb = embed_single_message_tokens(msg, model, processor, device, layer="last", average_hidden=True, 
            add_generation_prompt=True)  # [H]
            example_embs_list.append(neigh_emb)

            if output_mode == "concat" and len(example_embs_list) >= k_concat:
                break

        # --- handle no valid neighbors ---
        if not example_embs_list:
            if output_mode == "weighted_avg":
                pooled_embs.append(torch.full((1, H), pad_value, device=device))
            else:
                pooled_embs.append(torch.full((k_concat, H), pad_value, device=device))
            continue

        # ================= weighted_avg =================
        if output_mode == "weighted_avg":
            ex_embs = torch.stack(example_embs_list, dim=0)  # Stack neighbor embeddings → [K, H]
            scores_tensor = torch.tensor(kept_scores[:ex_embs.size(0)], dtype=torch.float32, device=device)  # Match scores length to K (some neighbors may have been skipped)
            weights = F.softmax(scores_tensor, dim=0)  # [K] Softmax over neighbors to get weights
            ret_emb = (ex_embs * weights.unsqueeze(-1)).sum(dim=0)  # [H] Weighted average over neighbors
            
            if l2_normalize:
                ret_emb = torch.nn.functional.normalize(ret_emb, p=2, dim=0)

            pooled_embs.append(ret_emb.unsqueeze(0))  # [1,H]
        # ================= concat =================
        elif output_mode == "concat":
            # We need exactly k_concat rows: pad if fewer.
            K = len(example_embs_list)
            if K == 0:
                ex_embs = torch.full((k_concat, H), pad_value, device=device)
            elif K < k_concat:
                pad = torch.full((k_concat - K, H), pad_value, device=device)
                ex_embs = torch.cat([torch.stack(example_embs_list, dim=0), pad], dim=0)  # [k_concat, H]
            else:
                ex_embs = torch.stack(example_embs_list[:k_concat], dim=0)  # [k_concat, H]
            
            if l2_normalize:
                ex_embs = F.normalize(ex_embs, p=2, dim=-1)  # row-wise
            
            pooled_embs.append(ex_embs)
        
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}. Use 'weighted_avg' or 'concat'.")

    ret_emb_batch = torch.stack(pooled_embs, dim=0).float().to(device)
    return ret_emb_batch  # weighted_avg: [B,1,H], concat: [B,k_concat,H]
