"""
PubMedQA Stage-1 pretraining: paraphrase pretraining for the MLP projector.

The MLP learns to map pre-computed corpus content embeddings into
injection vectors that allow the frozen LM to reconstruct the passage text.

Key differences from SLAKE:
  - Text-only model: uses tokenizer, not processor
  - Retrieval uses pre-computed content embeddings (no online re-encoding)
  - Corpus-based retrieval: query_emb → corpus FAISS index
  - Evidence = passage title+abstract from corpus_task_map
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

from VQA.utilise.mlp_utilise.mlp_train_stage1_utilise import (
    token_accuracy_and_supervised_ratio_causal,
    median_l2_norm,
    cosine_pos_neg_margin,
)
from VQA.utilise.PubMedQA.PubMedQA_mlp_utilise import build_corpus_evidence_text

REPRAG_TOKEN = "<RepRAG>"
PARAPHRASE_INSTRUCTIONS = [
    "Background: {reprag_token} means the same as",
    "Background: {reprag_token} Can you put the above sentences in your own terms?",
    "Background: {reprag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {reprag_token}\n(2)",
    "Background: {reprag_token} is a paraphrase of what?",
]


# -------------------------------------------------------
# Corpus retrieval using pre-computed query embeddings
# -------------------------------------------------------

def retrieve_corpus_for_batch(
    batch_tasks: List[Dict[str, Any]],
    query_emb_normed: np.ndarray,
    query_qid_to_row: Dict[str, int],
    corpus_faiss_index,
    corpus_qids: List[str],
    top_k: int,
    invalid_score_thr: float = -3e38,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Retrieve top-k corpus items for each task using pre-computed training
    question embeddings as queries against the shared corpus FAISS index.
    """
    batch_qids, batch_scores = [], []
    for t in batch_tasks:
        qid = str(t["qid"])
        row_idx = query_qid_to_row.get(qid)
        if row_idx is None:
            batch_qids.append([])
            batch_scores.append([])
            continue

        query_vec = query_emb_normed[row_idx].reshape(1, -1).astype("float32", copy=False)
        if not np.isfinite(query_vec).all():
            batch_qids.append([])
            batch_scores.append([])
            continue

        scores, idxs = corpus_faiss_index.search(query_vec, top_k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        pairs = []
        for s, i in zip(scores, idxs):
            if i < 0 or i >= len(corpus_qids):
                continue
            if not np.isfinite(s) or s <= invalid_score_thr:
                continue
            pairs.append((float(s), corpus_qids[i]))

        batch_scores.append([s for s, _ in pairs[:top_k]])
        batch_qids.append([q for _, q in pairs[:top_k]])

    return batch_qids, batch_scores


def lookup_content_embeddings(
    batch_corpus_qids: List[List[str]],
    content_emb_map: Dict[str, np.ndarray],
    device: torch.device,
    k: int,
    H: Optional[int] = None,
) -> torch.Tensor:
    """Look up pre-computed content embeddings for retrieved corpus items.

    Returns tensor of shape [B, k, D] where D is the actual content embedding dim.
    H is kept as an optional fallback for the zero-pad dimension; if None, inferred
    from the first available embedding in content_emb_map.
    """
    # Infer D from the first available entry in content_emb_map if not supplied
    D = H
    if D is None:
        for v in content_emb_map.values():
            D = int(np.array(v).shape[0])
            break
    if D is None:
        raise ValueError("content_emb_map is empty and H was not provided.")

    pad = torch.zeros(D, dtype=torch.float32)
    result = []
    for qids in batch_corpus_qids:
        embs = []
        for qid in qids[:k]:
            vec = content_emb_map.get(str(qid))
            embs.append(
                torch.as_tensor(vec, dtype=torch.float32) if vec is not None else pad
            )
        while len(embs) < k:
            embs.append(pad)
        result.append(torch.stack(embs, dim=0))  # [k, D]
    return torch.stack(result, dim=0).to(device)  # [B, k, D]


# -------------------------------------------------------
# Sequence-packing helpers
# -------------------------------------------------------

def random_span_from_text(
    text: str,
    tokenizer,
    target_len: int,
    source_max_len: Optional[int] = None,
) -> str:
    if not text:
        return "."
    if source_max_len is None:
        model_max = getattr(tokenizer, "model_max_length", 4096)
        source_max_len = min(model_max, max(target_len, 4 * target_len))

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=source_max_len,
        padding=False,
    )["input_ids"]

    if len(ids) <= target_len:
        span_ids = ids
    else:
        start = random.randint(0, len(ids) - target_len)
        span_ids = ids[start : start + target_len]

    span = tokenizer.decode(
        span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).strip()
    return span if span else "."


def tokenize_targets(tokenizer, target_texts: List[str], max_tokens: int) -> List[List[int]]:
    tok = tokenizer(
        target_texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_tokens,
    )
    return tok["input_ids"]


def pack_left_padded_prompt_target(
    prompt_input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    target_ids_list: list,
    tgt_lens: torch.Tensor,
    pad_id: int,
    device: torch.device,
    attn_dtype: torch.dtype,
):
    B, T_prompt = prompt_input_ids.shape
    total_lens = prompt_lens + tgt_lens
    T_total = int(total_lens.max().item())

    input_ids = torch.full((B, T_total), pad_id, device=device, dtype=torch.long)
    attn_mask = torch.zeros((B, T_total), device=device, dtype=attn_dtype)
    starts = torch.empty((B,), device=device, dtype=torch.long)

    for b in range(B):
        p_len = int(prompt_lens[b].item())
        t_len = int(tgt_lens[b].item())
        total = p_len + t_len
        start = T_total - total
        starts[b] = start

        if p_len > 0:
            prompt_tokens = prompt_input_ids[b, T_prompt - p_len : T_prompt]
            input_ids[b, start : start + p_len] = prompt_tokens
            attn_mask[b, start : start + p_len] = 1

        if t_len > 0:
            tgt_tokens = torch.as_tensor(
                target_ids_list[b], device=device, dtype=torch.long
            )
            if tgt_tokens.numel() != t_len:
                tgt_tokens = tgt_tokens[:t_len]
                if tgt_tokens.numel() < t_len:
                    tgt_tokens = torch.cat(
                        [
                            tgt_tokens,
                            torch.full(
                                (t_len - tgt_tokens.numel(),),
                                pad_id,
                                device=device,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
            input_ids[b, start + p_len : start + p_len + t_len] = tgt_tokens
            attn_mask[b, start + p_len : start + p_len + t_len] = 1

    return input_ids, attn_mask, starts


def make_labels_supervise_only_target_span(
    input_ids: torch.Tensor,
    starts: torch.Tensor,
    prompt_lens: torch.Tensor,
    tgt_lens: torch.Tensor,
) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    for b in range(input_ids.shape[0]):
        p_len = int(prompt_lens[b].item())
        t_len = int(tgt_lens[b].item())
        if t_len <= 0:
            continue
        start = int(starts[b].item())
        ans_start = start + p_len
        ans_end = ans_start + t_len
        labels[b, ans_start:ans_end] = input_ids[b, ans_start:ans_end]
    return labels



def stage1_pretrain_step(
    args,
    base_model,
    tokenizer,
    mlp,
    optimizer,
    device,
    model_dtype,
    batch_tasks: List[Dict[str, Any]],
    query_emb_normed: np.ndarray,
    query_qid_to_row: Dict[str, int],
    corpus_faiss_index,
    corpus_qids: List[str],
    content_emb_map: Dict[str, np.ndarray],
    corpus_task_map: Dict[str, Dict[str, Any]],
    writer,
    global_step: int,
    reprag_token_id: int,
    pbar=None,
):
    # ---- 1) Retrieve top-1 corpus item per task ----
    K_eff = 1
    batch_corpus_qids, _ = retrieve_corpus_for_batch(
        batch_tasks=batch_tasks,
        query_emb_normed=query_emb_normed,
        query_qid_to_row=query_qid_to_row,
        corpus_faiss_index=corpus_faiss_index,
        corpus_qids=corpus_qids,
        top_k=K_eff,
    )

    # ---- 2) Build evidence text targets (passage spans) ----
    target_len = int(args.stage1_target_max_tokens)
    evidence_texts = []
    for qids in batch_corpus_qids:
        if not qids:
            evidence_texts.append(".")
            continue
        entry = corpus_task_map.get(str(qids[0]))
        if entry is None:
            evidence_texts.append(".")
            continue
        full_text = build_corpus_evidence_text(entry, max_chars=8000)
        span = random_span_from_text(full_text, tokenizer, target_len=target_len)
        evidence_texts.append(span)

    assert all(
        REPRAG_TOKEN not in t for t in evidence_texts
    ), "evidence_texts must not contain <RepRAG>"

    # ---- 3) Look up pre-computed content embeddings [B, 1, D_content] ----
    H = base_model.config.hidden_size
    ret_emb_batch = lookup_content_embeddings(
        batch_corpus_qids=batch_corpus_qids,
        content_emb_map=content_emb_map,
        device=device,
        k=K_eff,
    )  # [B, K_eff, D_content]
    B = ret_emb_batch.shape[0]
    D_content = ret_emb_batch.shape[-1]

    # ---- 4) MLP projection ----
    mlp_dtype = next(mlp.parameters()).dtype
    inj_vec = mlp(ret_emb_batch.reshape(B * K_eff, D_content).to(mlp_dtype)).reshape(
        B, K_eff, H
    ).to(dtype=model_dtype)

    # ---- 5) Tokenize targets ----
    target_ids_list = tokenize_targets(tokenizer, evidence_texts, max_tokens=target_len)
    tgt_lens = torch.tensor(
        [len(x) for x in target_ids_list], device=device, dtype=torch.long
    )

    # ---- 6) Build paraphrase prompt inputs ----
    reprag_token_str = " ".join([REPRAG_TOKEN] * K_eff)
    prompt_texts = [
        random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_token_str})
        for _ in range(B)
    ]
    prompt_enc = tokenizer(
        prompt_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    prompt_input_ids = prompt_enc["input_ids"]
    prompt_attn_mask = prompt_enc["attention_mask"]
    pad_id = tokenizer.pad_token_id
    prompt_lens = prompt_attn_mask.sum(dim=1).long()

    # ---- 7) Pack prompt + target into left-padded sequence ----
    input_ids, attn_mask, starts = pack_left_padded_prompt_target(
        prompt_input_ids=prompt_input_ids,
        prompt_lens=prompt_lens,
        target_ids_list=target_ids_list,
        tgt_lens=tgt_lens,
        pad_id=pad_id,
        device=device,
        attn_dtype=prompt_attn_mask.dtype,
    )

    labels = make_labels_supervise_only_target_span(
        input_ids=input_ids,
        starts=starts,
        prompt_lens=prompt_lens,
        tgt_lens=tgt_lens,
    )

    # ---- 8) Build embeddings and inject <RepRAG> ----
    with torch.no_grad():
        frozen_inputs_embeds = base_model.get_input_embeddings()(input_ids)

    inputs_embeds = frozen_inputs_embeds.clone().to(dtype=model_dtype)

    xmask = input_ids == reprag_token_id
    counts = xmask.sum(dim=1).tolist()
    assert all(
        c == K_eff for c in counts
    ), f"Expected {K_eff} <RepRAG> tokens per sample, got {counts}"

    inj_flat = inj_vec.reshape(B * K_eff, H).to(dtype=model_dtype)

    if not torch.isfinite(inj_flat).all():
        print(f"[stage1] WARNING: non-finite MLP output (NaN/Inf in inj_flat); skipping step.")
        if pbar is not None:
            pbar.set_postfix({"loss": "nan/inf"})
            pbar.update(1)
        return float("nan"), global_step + 1

    inputs_embeds[xmask] = inj_flat

    # ---- 9) Forward pass ----
    out = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        labels=labels,
        return_dict=True,
    )
    loss = out.loss
    assert loss.requires_grad, "Loss is detached; no gradient path to mlp."

    if not loss.isfinite():
        print(f"[stage1] WARNING: non-finite loss={loss.item()}; skipping backward to protect MLP weights.")
        if pbar is not None:
            pbar.set_postfix({"loss": "nan/inf"})
            pbar.update(1)
        return float("nan"), global_step + 1

    # ---- 10) Metrics ----
    train_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)
    inj_med = median_l2_norm(inj_flat)

    normal_mask = (attn_mask > 0) & (~xmask)
    normal_med = ratio = None
    if normal_mask.any():
        normal_emb = inputs_embeds[normal_mask]
        normal_med = median_l2_norm(normal_emb)
        ratio = inj_med / (normal_med + 1e-9)

    ret_flat = ret_emb_batch.reshape(B * K_eff, D_content).to(inj_flat.dtype)
    cos_pos, cos_neg, margin = cosine_pos_neg_margin(ret_flat, inj_flat)

    if writer is not None:
        writer.add_scalar("stage1/train_token_acc", train_tok_acc, global_step)
        writer.add_scalar("stage1/supervised_ratio", sup_ratio, global_step)
        writer.add_scalar("stage1/inj_norm_med", inj_med, global_step)
        if normal_med is not None:
            writer.add_scalar("stage1/normal_emb_norm_med", normal_med, global_step)
            writer.add_scalar("stage1/inj_norm_ratio", ratio, global_step)
        writer.add_scalar("stage1/ret_inj_cos_pos", cos_pos, global_step)
        writer.add_scalar("stage1/ret_inj_cos_neg", cos_neg, global_step)
        writer.add_scalar("stage1/ret_inj_cos_margin", margin, global_step)

    # ---- 11) Backward ----
    optimizer.zero_grad(set_to_none=True)
    loss.backward()


    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
    optimizer.step()

    if writer is not None:
        writer.add_scalar("stage1/loss", float(loss.item()), global_step)
        if hasattr(mlp, "alpha"):
            writer.add_scalar("stage1/alpha", float(torch.sigmoid(mlp.alpha).item()), global_step)

    if pbar is not None:
        postfix = {"loss": f"{loss.item():.4f}"}
        if hasattr(mlp, "alpha"):
            postfix["alpha"] = f"{torch.sigmoid(mlp.alpha).item():.4f}"
        pbar.set_postfix(postfix)
        pbar.update(1)

    return float(loss.item()), global_step + 1


# -------------------------------------------------------
# Stage 1 eval step
# -------------------------------------------------------

def stage1_eval_step(
    args,
    base_model,
    tokenizer,
    mlp,
    device,
    model_dtype,
    batch_tasks: List[Dict[str, Any]],
    query_emb_normed: np.ndarray,
    query_qid_to_row: Dict[str, int],
    corpus_faiss_index,
    corpus_qids: List[str],
    content_emb_map: Dict[str, np.ndarray],
    corpus_task_map: Dict[str, Dict[str, Any]],
    writer,
    global_step: int,
    reprag_token_id: int,
    eval_prefix: str = "stage1/val",
    pbar=None,
):
    random_state = random.getstate()
    random.seed(1234)

    base_was_training = base_model.training
    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    loss = None
    try:
        with torch.no_grad():
            K_eff = 1

            batch_corpus_qids, _ = retrieve_corpus_for_batch(
                batch_tasks=batch_tasks,
                query_emb_normed=query_emb_normed,
                query_qid_to_row=query_qid_to_row,
                corpus_faiss_index=corpus_faiss_index,
                corpus_qids=corpus_qids,
                top_k=K_eff,
            )

            target_len = int(args.stage1_target_max_tokens)
            evidence_texts = []
            for qids in batch_corpus_qids:
                if not qids:
                    evidence_texts.append(".")
                    continue
                entry = corpus_task_map.get(str(qids[0]))
                if entry is None:
                    evidence_texts.append(".")
                    continue
                full_text = build_corpus_evidence_text(entry, max_chars=8000)
                span = random_span_from_text(full_text, tokenizer, target_len=target_len)
                evidence_texts.append(span)

            H = base_model.config.hidden_size
            ret_emb_batch = lookup_content_embeddings(
                batch_corpus_qids=batch_corpus_qids,
                content_emb_map=content_emb_map,
                device=device,
                k=K_eff,
            )
            B = ret_emb_batch.shape[0]
            D_content = ret_emb_batch.shape[-1]

            mlp_dtype = next(mlp.parameters()).dtype
            inj_vec = mlp(ret_emb_batch.reshape(B * K_eff, D_content).to(mlp_dtype)).reshape(
                B, K_eff, H
            ).to(dtype=model_dtype)

            target_ids_list = tokenize_targets(tokenizer, evidence_texts, max_tokens=target_len)
            tgt_lens = torch.tensor(
                [len(x) for x in target_ids_list], device=device, dtype=torch.long
            )

            reprag_token_str = " ".join([REPRAG_TOKEN] * K_eff)
            prompt_texts = [
                random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_token_str})
                for _ in range(B)
            ]
            prompt_enc = tokenizer(
                prompt_texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            prompt_input_ids = prompt_enc["input_ids"]
            prompt_attn_mask = prompt_enc["attention_mask"]
            pad_id = tokenizer.pad_token_id
            prompt_lens = prompt_attn_mask.sum(dim=1).long()

            input_ids, attn_mask, starts = pack_left_padded_prompt_target(
                prompt_input_ids=prompt_input_ids,
                prompt_lens=prompt_lens,
                target_ids_list=target_ids_list,
                tgt_lens=tgt_lens,
                pad_id=pad_id,
                device=device,
                attn_dtype=prompt_attn_mask.dtype,
            )

            labels = make_labels_supervise_only_target_span(
                input_ids=input_ids,
                starts=starts,
                prompt_lens=prompt_lens,
                tgt_lens=tgt_lens,
            )

            inputs_embeds = base_model.get_input_embeddings()(input_ids).to(dtype=model_dtype)

            xmask = input_ids == reprag_token_id
            inj_flat = inj_vec.reshape(B * K_eff, H).to(dtype=model_dtype)
            inputs_embeds[xmask] = inj_flat

            out = base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels,
                return_dict=True,
            )
            loss = out.loss

            val_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(
                out.logits, labels
            )
            inj_med = median_l2_norm(inj_flat)

            normal_mask = (attn_mask > 0) & (~xmask)
            normal_med = ratio = None
            if normal_mask.any():
                normal_emb = inputs_embeds[normal_mask]
                normal_med = median_l2_norm(normal_emb)
                ratio = inj_med / (normal_med + 1e-9)

            ret_flat = ret_emb_batch.reshape(B * K_eff, D_content).to(inj_flat.dtype)
            cos_pos, cos_neg, margin = cosine_pos_neg_margin(ret_flat, inj_flat)

            if writer is not None:
                writer.add_scalar(f"{eval_prefix}_loss", float(loss.item()), global_step)
                writer.add_scalar(f"{eval_prefix}_token_acc", val_tok_acc, global_step)
                writer.add_scalar(f"{eval_prefix}_supervised_ratio", sup_ratio, global_step)
                writer.add_scalar(f"{eval_prefix}_inj_norm_med", inj_med, global_step)
                if normal_med is not None:
                    writer.add_scalar(f"{eval_prefix}_normal_emb_norm_med", normal_med, global_step)
                    writer.add_scalar(f"{eval_prefix}_inj_norm_ratio", ratio, global_step)
                writer.add_scalar(f"{eval_prefix}_ret_inj_cos_pos", cos_pos, global_step)
                writer.add_scalar(f"{eval_prefix}_ret_inj_cos_neg", cos_neg, global_step)
                writer.add_scalar(f"{eval_prefix}_ret_inj_cos_margin", margin, global_step)

    finally:
        base_model.train(base_was_training)
        mlp.train(mlp_was_training)
        random.setstate(random_state)

    if pbar is not None:
        if loss is not None:
            pbar.set_postfix({"eval_loss": f"{loss.item():.4f}"})
        pbar.update(1)

    return float(loss.item()) if loss is not None else None
