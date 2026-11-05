import os
import random
import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from VQA.utilise.retrieval_helper import retrieve_neighbors_for_batch, embed_single_message_tokens
from VQA.utilise.mlp_utilise.mlp_train_stage1_utilise import token_accuracy_and_supervised_ratio_causal, median_l2_norm, cosine_pos_neg_margin


REPRAG_TOKEN = "<RepRAG>"
PARAPHRASE_INSTRUCTIONS = [
    "Background: {reprag_token} means the same as",
    "Background: {reprag_token} Can you put the above sentences in your own terms?",
    "Background: {reprag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {reprag_token}\n(2)",
    "Background: {reprag_token} is a paraphrase of what?",
]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# =========================
# 1) Helpers to filter retrived samples
# =========================
def _get_solution_text(ex: Dict[str, Any]) -> str:
    # unify all possible fields you may have
    sol = (ex.get("solution") or ex.get("rationale") or "").strip()
    return sol

def _has_solution(ex: Dict[str, Any]) -> bool:
    return len(_get_solution_text(ex)) > 0

def select_first_neighbor_with_solution(batch_neighbors_qids: List[List[str]], batch_neighbors_scores: Optional[List[List[float]]], 
train_task_map: Dict[str, Dict[str, Any]], topk: int = 1) -> Tuple[List[List[str]], Optional[List[List[float]]]]:
    """
    For each sample, pick the FIRST retrieved neighbor (in ranked order)
    that has a non-empty solution/rationale. Always return exactly `topk` items
    (your use-case is topk=1).
    Fallback: if none has solution, use the first neighbor.
    """
    new_qids: List[List[str]] = []
    new_scores: Optional[List[List[float]]] = [] if batch_neighbors_scores is not None else None

    for i, neigh in enumerate(batch_neighbors_qids):
        picked_qids: List[str] = []
        picked_scores: List[float] = []

        # scan in order to find solution-containing examples
        for j, qid in enumerate(neigh):
            ex = train_task_map.get(str(qid))
            if ex is None:
                continue
            if _has_solution(ex):
                picked_qids.append(str(qid))
                if batch_neighbors_scores is not None:
                    picked_scores.append(float(batch_neighbors_scores[i][j]))
                if len(picked_qids) >= topk:
                    break

        # fallback: if none has solution, keep first valid neighbor (or empty)
        if len(picked_qids) == 0:
            fallback_qid = None
            fallback_score = None
            for j, qid in enumerate(neigh):
                ex = train_task_map.get(str(qid))
                if ex is not None:
                    fallback_qid = str(qid)
                    if batch_neighbors_scores is not None:
                        fallback_score = float(batch_neighbors_scores[i][j])
                    break

            if fallback_qid is not None:
                picked_qids = [fallback_qid]
                if batch_neighbors_scores is not None:
                    picked_scores = [fallback_score if fallback_score is not None else 0.0]
            else:
                picked_qids = []
                if batch_neighbors_scores is not None:
                    picked_scores = []

        new_qids.append(picked_qids)
        if new_scores is not None:
            new_scores.append(picked_scores)

    return new_qids, new_scores

def random_span_from_text(text: str, tokenizer, target_len: int, source_max_len: Optional[int] = None) -> str:
    """
    Take a random contiguous token window (length=target_len) from `text`.
    Returns decoded span text.
    """
    if not text:
        return "."

    if source_max_len is None:
        # give room for random windows without adding a new arg
        model_max = getattr(tokenizer, "model_max_length", 4096)
        source_max_len = min(model_max, max(target_len, 4 * target_len))

    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=source_max_len,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    if len(ids) <= target_len:
        span_ids = ids
    else:
        start = random.randint(0, len(ids) - target_len)
        span_ids = ids[start : start + target_len]

    span_text = tokenizer.decode(span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return span_text if span_text else "."


def deterministic_span_from_text(text: str, tokenizer, target_len: int, source_max_len: Optional[int] = None) -> str:
    """
    Take a deterministic contiguous token window (the first window) from `text`.
    Returns decoded span text.
    """
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
        return_tensors=None,
    )["input_ids"]

    span_ids = ids[:target_len]
    span_text = tokenizer.decode(span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return span_text if span_text else "."


# =========================
# 2) Helpers to build target text & embedding from VLM
# =========================
def build_solution_only_message(ex: dict) -> dict:
    """
    Build a message that contains ONLY the solution/rationale text.
    This is what you will tokenize+embed for prefix construction.
    """
    sol = _get_solution_text(ex)
    if not sol:
        # keep something minimal (avoid empty message)
        sol = "."

    content = [{"type": "text", "text": sol}]
    return {"role": "user", "content": content}

def build_message_from_text(text: str) -> dict:
    if not text:
        text = "."
    return {"role": "user", "content": [{"type": "text", "text": text}]}

def build_prefix_last_layer_for_batch_from_retrieved(batch_neighbors_qids: List[List[str]], batch_neighbors_scores: List[List[float]],
train_task_map: Dict[str, dict], model, processor, device, l2_normalize: bool = True, output_mode: Literal["weighted_avg", "concat"] = "concat", 
k_concat: Optional[int] = None, pad_value: float = 0.0, override_texts: Optional[List[List[str]]] = None) -> torch.Tensor:
    """
    Build retrieved-example embeddings per test item.

    Modes:
      - output_mode="weighted_avg": returns [B, 1, H] using softmax-weighted average over neighbors
      - output_mode="concat":       returns [B, k_concat, H] by stacking per-neighbor embeddings (pads/truncates)

    Notes for concat:
      - Uses neighbors in the same order as batch_neighbors_qids.
      - If fewer than k_concat valid neighbors, pads with pad_value.
      - If more than k_concat, truncates.
      - If l2_normalize=True, each neighbor embedding is L2-normalized (padding stays pad_value).
    """
    B = len(batch_neighbors_qids)
    assert B == len(batch_neighbors_scores), "qids and scores batch size mismatch"

    if override_texts is not None:
        assert len(override_texts) == B, f"override_texts batch mismatch: {len(override_texts)} vs B={B}"


    if output_mode == "concat":
        if k_concat is None:
            k_concat = max((len(x) for x in batch_neighbors_qids), default=0)
        if k_concat <= 0:
            raise ValueError("k_concat must be a positive int for output_mode='concat'")

    # Underlying HF model (unwrap Deepspeed if needed)
    base_model = model.module if hasattr(model, "module") else model
    H = base_model.config.hidden_size

    pooled_embs = []  # each element: [1, H] for weighted_avg, or [k_concat, H] for concat

    for bi, (neighbor_qids, neighbor_scores) in enumerate(zip(batch_neighbors_qids, batch_neighbors_scores)):
        example_embs_list = []
        kept_scores = []

        for j, qid in enumerate(neighbor_qids):
            if override_texts is not None:
                sol_text = override_texts[bi][j] if j < len(override_texts[bi]) else "."
                msg = build_message_from_text(sol_text)
            else:
                ex = train_task_map.get(str(qid))
                if ex is None:
                    continue
                msg = build_solution_only_message(ex)

            score = neighbor_scores[j] if j < len(neighbor_scores) else 0.0
            kept_scores.append(float(score))
            
            neigh_emb = embed_single_message_tokens(msg, model, processor, device, layer="last", average_hidden=True,
            add_generation_prompt=True)  # [H]
            example_embs_list.append(neigh_emb)

            # in concat mode, stop once we have enough
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
            ret_emb = (ex_embs * weights.unsqueeze(-1)).sum(dim=0)   # [H] Weighted average over neighbors

            if l2_normalize:
                ret_emb = F.normalize(ret_emb, p=2, dim=0)

            pooled_embs.append(ret_emb.unsqueeze(0))  # [1, H]

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


def tokenize_targets(processor, target_texts: List[str], max_tokens: int) -> List[List[int]]:
    tok = processor.tokenizer(target_texts, add_special_tokens=False, padding=False, truncation=True, max_length=max_tokens)
    return tok["input_ids"]


# =========================
# 3) Helpers to build teacher forcing training sequence
# =========================
def pack_left_padded_prompt_target(prompt_input_ids: torch.Tensor, prompt_lens: torch.Tensor, target_ids_list: list, tgt_lens: torch.Tensor, 
pad_id: int, device: torch.device, attn_dtype: torch.dtype):
    """
    Left-pad (right-align) [prompt][target] for each sample into a rectangular [B, T_total].

    Returns:
      input_ids: [B, T_total]
      attn_mask: [B, T_total] (0/1)
      starts:    [B] start offset where (prompt+target) begins
    """
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
        start = T_total - total  # left-pad so content is right-aligned
        starts[b] = start

        # prompt tokens: take the last p_len tokens from the prompt buffer
        if p_len > 0:
            prompt_tokens = prompt_input_ids[b, T_prompt - p_len : T_prompt]
            input_ids[b, start : start + p_len] = prompt_tokens
            attn_mask[b, start : start + p_len] = 1

        # target tokens: already exact length t_len
        if t_len > 0:
            tgt_tokens = torch.as_tensor(target_ids_list[b], device=device, dtype=torch.long)
            if tgt_tokens.numel() != t_len:
                # defensive: if list length disagrees with tgt_lens
                tgt_tokens = tgt_tokens[:t_len]
                if tgt_tokens.numel() < t_len:
                    # pad target if somehow shorter (shouldn't happen)
                    tgt_tokens = torch.cat(
                        [tgt_tokens, torch.full((t_len - tgt_tokens.numel(),), pad_id, device=device, dtype=torch.long)],
                        dim=0
                    )
            input_ids[b, start + p_len : start + p_len + t_len] = tgt_tokens
            attn_mask[b, start + p_len : start + p_len + t_len] = 1

    return input_ids, attn_mask, starts


def make_labels_supervise_only_target_span(input_ids: torch.Tensor, starts: torch.Tensor, prompt_lens: torch.Tensor, tgt_lens: torch.Tensor) -> torch.Tensor:
    """
    Create labels = -100 everywhere except the target token span.
    For left-padded packed sequences, target span begins at: start + p_len
    """
    labels = torch.full_like(input_ids, -100)
    B, T_total = input_ids.shape

    for b in range(B):
        p_len = int(prompt_lens[b].item())
        t_len = int(tgt_lens[b].item())
        if t_len <= 0:
            continue
        start = int(starts[b].item())
        ans_start = start + p_len
        ans_end = ans_start + t_len
        # supervise target tokens
        labels[b, ans_start:ans_end] = input_ids[b, ans_start:ans_end]

    return labels



def stage1_pretrain_step(args, base_model, qwen_model, processor, mlp, optimizer, device, model_dtype, batch_tasks, train_emb_normed,
train_qids, qid_to_row, index, train_task_map, writer, global_step, reprag_token_id: int, pbar=None):
    # ---- 1) Retrieve neighbors ----
    batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch(
        batch_tasks=batch_tasks,
        train_emb_normed=train_emb_normed,
        train_qids=train_qids,
        qid_to_row=qid_to_row,
        index=index,
        top_k=50,
    )
    assert batch_neighbors_scores is not None, "retrieve_neighbors_for_batch must return scores for stage1."

    # ---- 2) Pick FIRST neighbor with solution ----
    K_eff = 1
    batch_neighbors_qids, batch_neighbors_scores = select_first_neighbor_with_solution(
        batch_neighbors_qids=batch_neighbors_qids,
        batch_neighbors_scores=batch_neighbors_scores,
        train_task_map=train_task_map,
        topk=K_eff
    )

    # ---- 3) Build target span texts (random window) and align embedding inputs ----
    target_len = int(args.stage1_target_max_tokens)

    # For K_eff=1, each sample has at most 1 neighbor qid after selection.
    # Build span_texts per sample (list of list, aligned with batch_neighbors_qids)
    override_texts = []
    evidence_texts = []

    for neigh in batch_neighbors_qids:
        if len(neigh) == 0:
            span = "."
            override_texts.append([span])
            evidence_texts.append(span)
            continue

        qid = neigh[0]
        ex = train_task_map.get(str(qid))
        if ex is None:
            span = "."
            override_texts.append([span])
            evidence_texts.append(span)
            continue

        full_sol = _get_solution_text(ex)
        span = random_span_from_text(full_sol, processor.tokenizer, target_len=target_len)

        span_token_len = len(processor.tokenizer(span, add_special_tokens=False)["input_ids"])
        full_token_len = len(processor.tokenizer(full_sol, add_special_tokens=False)["input_ids"])
        # print("token_len full_sol:", full_token_len, "token_len span:", span_token_len)

        # strict alignment: embedding uses same span
        override_texts.append([span])
        evidence_texts.append(span)

    assert all(REPRAG_TOKEN not in t for t in evidence_texts), \
        "ERROR: evidence_texts (training targets) accidentally contain <RepRAG> token!"

    target_ids_list = tokenize_targets(processor, evidence_texts, max_tokens=target_len)
    tgt_lens = torch.tensor([len(x) for x in target_ids_list], device=device, dtype=torch.long)


    # ---- 4) Build retrieval embeddings [B,K,H] (frozen forward inside helper) ----
    H = base_model.config.hidden_size
    ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved(
        batch_neighbors_qids=batch_neighbors_qids,
        batch_neighbors_scores=batch_neighbors_scores,
        train_task_map=train_task_map,
        model=qwen_model,
        processor=processor,
        device=device,
        output_mode=args.embedding_output_mode,
        k_concat=K_eff,
        override_texts=override_texts
    )  # [B,K,H]
    # print("ret_emb_batch size: {0}".format(ret_emb_batch.size()))
    B, K, _ = ret_emb_batch.shape
    assert K == K_eff, f"Expected K={K_eff}, got {K}"

    # ---- 4) Project to injected embeddings (trainable) ----
    inj_vec = mlp(ret_emb_batch.reshape(B * K, H)).reshape(B, K, H).to(dtype=model_dtype)

    # ---- 5) RepRAG stage-1 prompt: instruction + K times <RepRAG> ----
    reprag_token_str = " ".join([REPRAG_TOKEN] * K)
    prompt_texts = [
        random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_token_str})
        for _ in range(B)
    ]
    # print("prompt_texts: {0}".format(prompt_texts))

    prompt_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt").to(device)

    prompt_input_ids = prompt_inputs["input_ids"]      # [B,T_prompt]
    prompt_attn_mask = prompt_inputs["attention_mask"] # [B,T_prompt]
    pad_id = processor.tokenizer.pad_token_id
    B2, T_prompt = prompt_input_ids.shape
    assert B2 == B

    prompt_lens = prompt_attn_mask.sum(dim=1).long()   # [B]


    # ---- 6) Pack prompt + target into input_ids/attn_mask (LEFT padded / right-aligned) ----
    input_ids, attn_mask, starts = pack_left_padded_prompt_target(
        prompt_input_ids=prompt_input_ids,
        prompt_lens=prompt_lens,
        target_ids_list=target_ids_list,
        tgt_lens=tgt_lens,
        pad_id=pad_id,
        device=device,
        attn_dtype=prompt_attn_mask.dtype
    )
    T_total = input_ids.shape[1]

    # ---- 7) Labels: supervise ONLY the target span ----
    # IMPORTANT: must use absolute span = start + p_len (NOT just prompt_lens)
    labels = make_labels_supervise_only_target_span(
        input_ids=input_ids,
        starts=starts,
        prompt_lens=prompt_lens,
        tgt_lens=tgt_lens
    )

    # ---- 8) Embed + overwrite <RepRAG> positions (RepRAG exact) ----
    with torch.no_grad():
        frozen_inputs_embeds = base_model.get_input_embeddings()(input_ids)  # [B, T_total, H]

    inputs_embeds = frozen_inputs_embeds.clone().to(dtype=model_dtype)

    mask = (input_ids == reprag_token_id)  # [B, T_total]
    counts = mask.sum(dim=1).tolist()
    assert all(c == K for c in counts), f"Each sample must have K <RepRAG> tokens, got {counts}"

    num_slots = int(mask.sum().item())
    expected = B * K
    assert num_slots == expected, (
        f"need exactly B*K <RepRAG> tokens, got {num_slots}, expected {expected}. "
        f"Likely <RepRAG> isn't a single token in tokenizer."
    )

    inj_flat = inj_vec.reshape(expected, H).to(dtype=model_dtype)
    inputs_embeds[mask] = inj_flat

    # ---- 9) Forward + optimize ----
    out = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        labels=labels,
        return_dict=True
    )
    loss = out.loss
    assert loss.requires_grad, "Loss is detached; no gradient path to mlp."

    # ====== A: token acc (train) ======
    train_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)

    # ====== B: inj norm ratio (proxy) ======
    inj_med = median_l2_norm(inj_flat)

    normal_mask = (attn_mask > 0) & (~mask)
    normal_med = None
    ratio = None
    if normal_mask.any():
        normal_emb = inputs_embeds[normal_mask]
        normal_med = median_l2_norm(normal_emb)
        ratio = inj_med / (normal_med + 1e-9)

    # ====== C: ret -> inj alignment (projector diagnostic) ======
    ret_flat = ret_emb_batch.reshape(B * K, H).to(inj_flat.dtype)
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
    else:
        print(f"global_step: {global_step} - stage1/loss: {loss.item():.4f}")
        if hasattr(mlp, "alpha"):
            print(f"global_step: {global_step} - stage1/alpha: {torch.sigmoid(mlp.alpha).item():.4f}")

    return float(loss.item()), global_step + 1


def stage1_eval_step(args, base_model, qwen_model, processor, mlp, device, model_dtype, batch_tasks, train_emb_normed,
train_qids, qid_to_row, index, train_task_map, writer, global_step, reprag_token_id: int, eval_prefix: str = "stage1/val", pbar=None):
    base_was_training = base_model.training
    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    try:
        with torch.no_grad():
            # ---- 1) Retrieve neighbors ----
            batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch(
                batch_tasks=batch_tasks,
                train_emb_normed=train_emb_normed,
                train_qids=train_qids,
                qid_to_row=qid_to_row,
                index=index,
                top_k=50,
            )
            assert batch_neighbors_scores is not None, "retrieve_neighbors_for_batch must return scores for stage1."

            # ---- 2) Pick FIRST neighbor with solution ----
            K_eff = 1
            batch_neighbors_qids, batch_neighbors_scores = select_first_neighbor_with_solution(
                batch_neighbors_qids=batch_neighbors_qids,
                batch_neighbors_scores=batch_neighbors_scores,
                train_task_map=train_task_map,
                topk=K_eff
            )

            # ---- 3) Build target span texts (random window) and align embedding inputs ----
            target_len = int(args.stage1_target_max_tokens)

            # For K_eff=1, each sample has at most 1 neighbor qid after selection.
            # Build span_texts per sample (list of list, aligned with batch_neighbors_qids)
            override_texts = []
            evidence_texts = []

            for neigh in batch_neighbors_qids:
                if len(neigh) == 0:
                    span = "."
                    override_texts.append([span])
                    evidence_texts.append(span)
                    continue

                qid = neigh[0]
                ex = train_task_map.get(str(qid))
                if ex is None:
                    span = "."
                    override_texts.append([span])
                    evidence_texts.append(span)
                    continue

                full_sol = _get_solution_text(ex)
                span = deterministic_span_from_text(full_sol, processor.tokenizer, target_len=target_len)

                span_token_len = len(processor.tokenizer(span, add_special_tokens=False)["input_ids"])
                full_token_len = len(processor.tokenizer(full_sol, add_special_tokens=False)["input_ids"])
                # print("token_len full_sol:", full_token_len, "token_len span:", span_token_len)

                # strict alignment: embedding uses same span
                override_texts.append([span])
                evidence_texts.append(span)

            assert all(REPRAG_TOKEN not in t for t in evidence_texts), \
                "ERROR: evidence_texts (training targets) accidentally contain <RepRAG> token!"

            target_ids_list = tokenize_targets(processor, evidence_texts, max_tokens=target_len)
            tgt_lens = torch.tensor([len(x) for x in target_ids_list], device=device, dtype=torch.long)


            # ---- 4) Build retrieval embeddings [B,K,H] (frozen forward inside helper) ----
            H = base_model.config.hidden_size
            ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved(
                batch_neighbors_qids=batch_neighbors_qids,
                batch_neighbors_scores=batch_neighbors_scores,
                train_task_map=train_task_map,
                model=qwen_model,
                processor=processor,
                device=device,
                output_mode=args.embedding_output_mode,
                k_concat=K_eff,
                override_texts=override_texts
            )  # [B,K,H]
            # print("ret_emb_batch size: {0}".format(ret_emb_batch.size()))
            B, K, _ = ret_emb_batch.shape
            assert K == K_eff, f"Expected K={K_eff}, got {K}"

            # ---- 4) Project to injected embeddings (trainable) ----
            inj_vec = mlp(ret_emb_batch.reshape(B * K, H)).reshape(B, K, H).to(dtype=model_dtype)

            # ---- 5) RepRAG stage-1 prompt: instruction + K times <RepRAG> ----
            reprag_token_str = " ".join([REPRAG_TOKEN] * K)
            prompt_template = PARAPHRASE_INSTRUCTIONS[0]
            prompt_texts = [
                prompt_template.format_map({"reprag_token": reprag_token_str})
                for _ in range(B)
            ]
            # print("prompt_texts: {0}".format(prompt_texts))
            
            prompt_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt").to(device)

            prompt_input_ids = prompt_inputs["input_ids"]      # [B,T_prompt]
            prompt_attn_mask = prompt_inputs["attention_mask"] # [B,T_prompt]
            pad_id = processor.tokenizer.pad_token_id
            B2, T_prompt = prompt_input_ids.shape
            assert B2 == B

            prompt_lens = prompt_attn_mask.sum(dim=1).long()   # [B]


            # ---- 6) Pack prompt + target into input_ids/attn_mask (LEFT padded / right-aligned) ----
            input_ids, attn_mask, starts = pack_left_padded_prompt_target(
                prompt_input_ids=prompt_input_ids,
                prompt_lens=prompt_lens,
                target_ids_list=target_ids_list,
                tgt_lens=tgt_lens,
                pad_id=pad_id,
                device=device,
                attn_dtype=prompt_attn_mask.dtype
            )
            T_total = input_ids.shape[1]

            # ---- 7) Labels: supervise ONLY the target span ----
            # IMPORTANT: must use absolute span = start + p_len (NOT just prompt_lens)
            labels = make_labels_supervise_only_target_span(
                input_ids=input_ids,
                starts=starts,
                prompt_lens=prompt_lens,
                tgt_lens=tgt_lens
            )

            # ---- 8) Embed + overwrite <RepRAG> positions (RepRAG exact) ----
            with torch.no_grad():
                inputs_embeds = base_model.get_input_embeddings()(input_ids).to(dtype=model_dtype)  # [B, T_total, H]

            mask = (input_ids == reprag_token_id)  # [B, T_total]
            counts = mask.sum(dim=1).tolist()
            assert all(c == K for c in counts), f"Each sample must have K <RepRAG> tokens, got {counts}"

            num_slots = int(mask.sum().item())
            expected = B * K
            assert num_slots == expected, (
                f"need exactly B*K <RepRAG> tokens, got {num_slots}, expected {expected}. "
                f"Likely <RepRAG> isn't a single token in tokenizer."
            )

            # inj_vec must match [B*K, H]
            inj_flat = inj_vec.reshape(expected, H).to(dtype=model_dtype)
            inputs_embeds[mask] = inj_flat

            # ---- 9) Forward + optimize ----
            # IMPORTANT: do NOT pass input_ids when passing inputs_embeds
            out = base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels,
                return_dict=True
            )
            loss = out.loss

            # ====== A: val loss & token acc ======
            val_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)

            # ====== B: inj norm ratio ======
            inj_med = median_l2_norm(inj_flat)
            normal_mask = (attn_mask > 0) & (~mask)
            if normal_mask.any():
                normal_emb = inputs_embeds[normal_mask]
                normal_med = median_l2_norm(normal_emb)
                ratio = inj_med / (normal_med + 1e-9)
            else:
                normal_med = None
                ratio = None

            # ====== C: ret -> inj alignment ======
            ret_flat = ret_emb_batch.reshape(B * K, H).to(inj_flat.dtype)
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

    if pbar is not None:
        pbar.set_postfix({"eval_loss": f"{loss.item():.4f}"})
        pbar.update(1)

    return float(loss.item())
