"""
KILT-NQ Stage-1: paraphrase pretraining for the MLP projector.

The MLP learns to map pre-computed Q+A content embeddings into injection
vectors that allow the frozen LM to reconstruct the Q+A pair text.

Unlike PubMedQA, the corpus IS the train split itself (no separate pool).
"""
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

from VQA.utilise.mlp_utilise.mlp_train_stage1_utilise import (
    token_accuracy_and_supervised_ratio_causal,
    median_l2_norm,
    cosine_pos_neg_margin,
)
from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage1 import (
    pack_left_padded_prompt_target,
    make_labels_supervise_only_target_span,
    tokenize_targets,
    random_span_from_text,
)

REPRAG_TOKEN = "<RepRAG>"
PARAPHRASE_INSTRUCTIONS = [
    "Background: {reprag_token} means the same as",
    "Background: {reprag_token} Can you put the above sentences in your own terms?",
    "Background: {reprag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {reprag_token}\n(2)",
    "Background: {reprag_token} is a paraphrase of what?",
]


def build_kilt_nq_evidence_text(rec: Dict[str, Any], max_chars: int = 2000) -> str:
    """Format a KILT-NQ record as evidence text (Q+A pair)."""
    q = rec.get("question", "").strip()
    ans = rec["gold_answers"][0].strip() if rec.get("gold_answers") else ""
    text = f"Question: {q}\nAnswer: {ans}" if ans else q
    return text[:max_chars]


# -------------------------------------------------------
# Retrieval helpers (train-set-as-corpus)
# -------------------------------------------------------

def retrieve_train_for_batch(
    batch_records: List[Dict[str, Any]],
    query_emb_normed: np.ndarray,
    query_id_to_row: Dict[str, int],
    train_faiss_index,
    train_ids: List[str],
    top_k: int,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Retrieve top-k train items for each record using pre-computed embeddings."""
    batch_ret_ids, batch_scores = [], []
    for r in batch_records:
        row = query_id_to_row.get(str(r["id"]))
        if row is None:
            batch_ret_ids.append([])
            batch_scores.append([])
            continue
        qvec = query_emb_normed[row].reshape(1, -1).astype("float32")
        scores, idxs = train_faiss_index.search(qvec, top_k + 1)  # +1 to skip self
        ids, scs = [], []
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0 or i >= len(train_ids):
                continue
            if train_ids[i] == str(r["id"]):  # skip self
                continue
            ids.append(train_ids[i])
            scs.append(float(s))
            if len(ids) >= top_k:
                break
        batch_ret_ids.append(ids)
        batch_scores.append(scs)
    return batch_ret_ids, batch_scores


def lookup_content_embeddings(
    batch_ret_ids: List[List[str]],
    content_emb_map: Dict[str, np.ndarray],
    device: torch.device,
    k: int,
) -> torch.Tensor:
    D = None
    for v in content_emb_map.values():
        D = int(np.array(v).shape[0])
        break
    pad = torch.zeros(D, dtype=torch.float32)
    result = []
    for ids in batch_ret_ids:
        embs = [
            torch.as_tensor(content_emb_map[i], dtype=torch.float32)
            if i in content_emb_map else pad
            for i in ids[:k]
        ]
        while len(embs) < k:
            embs.append(pad)
        result.append(torch.stack(embs))
    return torch.stack(result).to(device)  # [B, k, D]


# -------------------------------------------------------
# Stage 1 train step
# -------------------------------------------------------

def stage1_pretrain_step(
    args, base_model, tokenizer, mlp, optimizer, device, model_dtype,
    batch_records, query_emb_normed, query_id_to_row,
    train_faiss_index, train_ids, train_record_map, content_emb_map,
    writer, global_step, reprag_token_id, pbar=None,
):
    K_eff = 1
    batch_ret_ids, _ = retrieve_train_for_batch(
        batch_records, query_emb_normed, query_id_to_row,
        train_faiss_index, train_ids, K_eff,
    )

    # ---- 1) Build evidence texts (Q+A pair text of retrieved item) ----
    target_len = int(args.stage1_target_max_tokens)
    evidence_texts = []
    for ids in batch_ret_ids:
        if not ids:
            evidence_texts.append(".")
            continue
        rec = train_record_map.get(ids[0])
        if rec is None:
            evidence_texts.append(".")
            continue
        full_text = build_kilt_nq_evidence_text(rec, max_chars=8000)
        span = random_span_from_text(full_text, tokenizer, target_len=target_len)
        evidence_texts.append(span)

    assert all(REPRAG_TOKEN not in t for t in evidence_texts), \
        "evidence_texts must not contain <RepRAG>"

    # ---- 2) Look up pre-computed content embeddings [B, 1, D_content] ----
    ret_emb_batch = lookup_content_embeddings(batch_ret_ids, content_emb_map, device, K_eff)
    B = ret_emb_batch.shape[0]
    D_content = ret_emb_batch.shape[-1]
    H = base_model.config.hidden_size

    # ---- 3) MLP projection ----
    mlp_dtype = next(mlp.parameters()).dtype
    inj_vec = mlp(ret_emb_batch.reshape(B * K_eff, D_content).to(mlp_dtype)).reshape(
        B, K_eff, H).to(model_dtype)

    # ---- 4) Tokenize targets ----
    target_ids_list = tokenize_targets(tokenizer, evidence_texts, max_tokens=target_len)
    tgt_lens = torch.tensor([len(x) for x in target_ids_list], device=device, dtype=torch.long)

    # ---- 5) Build paraphrase prompt inputs ----
    reprag_str = " ".join([REPRAG_TOKEN] * K_eff)
    prompt_texts = [
        random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_str})
        for _ in range(B)
    ]
    prompt_enc = tokenizer(prompt_texts, padding=True, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1).long()

    # ---- 6) Pack prompt + target ----
    input_ids, attn_mask, starts = pack_left_padded_prompt_target(
        prompt_enc["input_ids"], prompt_lens, target_ids_list, tgt_lens,
        tokenizer.pad_token_id, device, prompt_enc["attention_mask"].dtype,
    )
    labels = make_labels_supervise_only_target_span(input_ids, starts, prompt_lens, tgt_lens)

    # ---- 7) Build embeddings and inject <RepRAG> ----
    with torch.no_grad():
        frozen_embeds = base_model.get_input_embeddings()(input_ids)
    inputs_embeds = frozen_embeds.clone().to(model_dtype)

    xmask = input_ids == reprag_token_id
    counts = xmask.sum(dim=1).tolist()
    assert all(c == K_eff for c in counts), \
        f"Expected {K_eff} <RepRAG> tokens per sample, got {counts}"

    inj_flat = inj_vec.reshape(B * K_eff, H).to(model_dtype)
    if not torch.isfinite(inj_flat).all():
        print("[stage1] WARNING: non-finite MLP output (NaN/Inf in inj_flat); skipping step.")
        if pbar is not None:
            pbar.set_postfix({"loss": "nan/inf"})
            pbar.update(1)
        return float("nan"), global_step + 1
    inputs_embeds[xmask] = inj_flat

    # ---- 8) Forward pass ----
    out = base_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                     labels=labels, return_dict=True)
    loss = out.loss
    assert loss.requires_grad, "Loss is detached; no gradient path to mlp."

    if not loss.isfinite():
        print(f"[stage1] WARNING: non-finite loss={loss.item()}; skipping backward to protect MLP weights.")
        if pbar is not None:
            pbar.set_postfix({"loss": "nan/inf"})
            pbar.update(1)
        return float("nan"), global_step + 1

    # ---- 9) Metrics ----
    tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)
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
        writer.add_scalar("stage1/loss", float(loss.item()), global_step)
        writer.add_scalar("stage1/train_token_acc", tok_acc, global_step)
        writer.add_scalar("stage1/supervised_ratio", sup_ratio, global_step)
        writer.add_scalar("stage1/inj_norm_med", inj_med, global_step)
        if normal_med is not None:
            writer.add_scalar("stage1/normal_emb_norm_med", normal_med, global_step)
            writer.add_scalar("stage1/inj_norm_ratio", ratio, global_step)
        writer.add_scalar("stage1/ret_inj_cos_pos", cos_pos, global_step)
        writer.add_scalar("stage1/ret_inj_cos_neg", cos_neg, global_step)
        writer.add_scalar("stage1/ret_inj_cos_margin", margin, global_step)
        if hasattr(mlp, "alpha"):
            writer.add_scalar("stage1/alpha", float(torch.sigmoid(mlp.alpha).item()), global_step)

    # ---- 10) Backward ----
    optimizer.zero_grad(set_to_none=True)
    loss.backward()


    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
    optimizer.step()

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
    args, base_model, tokenizer, mlp, device, model_dtype,
    batch_records, query_emb_normed, query_id_to_row,
    train_faiss_index, train_ids, train_record_map, content_emb_map,
    writer, global_step, reprag_token_id, eval_prefix="stage1/val", pbar=None,
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
            batch_ret_ids, _ = retrieve_train_for_batch(
                batch_records, query_emb_normed, query_id_to_row,
                train_faiss_index, train_ids, K_eff,
            )

            target_len = int(args.stage1_target_max_tokens)
            evidence_texts = []
            for ids in batch_ret_ids:
                if not ids:
                    evidence_texts.append(".")
                    continue
                rec = train_record_map.get(ids[0])
                if rec is None:
                    evidence_texts.append(".")
                    continue
                full_text = build_kilt_nq_evidence_text(rec, max_chars=8000)
                span = random_span_from_text(full_text, tokenizer, target_len=target_len)
                evidence_texts.append(span)

            ret_emb_batch = lookup_content_embeddings(batch_ret_ids, content_emb_map, device, K_eff)
            B = ret_emb_batch.shape[0]
            D_content = ret_emb_batch.shape[-1]
            H = base_model.config.hidden_size

            mlp_dtype = next(mlp.parameters()).dtype
            inj_vec = mlp(ret_emb_batch.reshape(B * K_eff, D_content).to(mlp_dtype)).reshape(
                B, K_eff, H).to(model_dtype)

            target_ids_list = tokenize_targets(tokenizer, evidence_texts, max_tokens=target_len)
            tgt_lens = torch.tensor([len(x) for x in target_ids_list], device=device, dtype=torch.long)

            reprag_str = " ".join([REPRAG_TOKEN] * K_eff)
            prompt_texts = [
                random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_str})
                for _ in range(B)
            ]
            prompt_enc = tokenizer(prompt_texts, padding=True, return_tensors="pt",
                                   truncation=True, max_length=256).to(device)
            prompt_lens = prompt_enc["attention_mask"].sum(dim=1).long()

            input_ids, attn_mask, starts = pack_left_padded_prompt_target(
                prompt_enc["input_ids"], prompt_lens, target_ids_list, tgt_lens,
                tokenizer.pad_token_id, device, prompt_enc["attention_mask"].dtype,
            )
            labels = make_labels_supervise_only_target_span(input_ids, starts, prompt_lens, tgt_lens)

            inputs_embeds = base_model.get_input_embeddings()(input_ids).to(model_dtype)
            xmask = input_ids == reprag_token_id
            inj_flat = inj_vec.reshape(B * K_eff, H).to(model_dtype)
            inputs_embeds[xmask] = inj_flat

            out = base_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                             labels=labels, return_dict=True)
            loss = out.loss

            val_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)
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