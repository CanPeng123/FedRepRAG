import os
import random
import sys
from typing import List, Dict, Any, Tuple, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from VQA.utilise.retrieval_helper import retrieve_neighbors_for_batch, retrieve_neighbors_for_batch_by_embedding
from VQA.utilise.retrieval_helper import embed_single_message_tokens, build_prefix_last_layer_for_batch_from_retrieved_generic
from VQA.utilise.mlp_utilise.mlp_train_stage1_utilise import token_accuracy_and_supervised_ratio_causal, median_l2_norm, cosine_pos_neg_margin
from VQA.utilise.SLAKE.Slake_data_process import load_pil_or_none
from VQA.utilise.SLAKE.Slake_prompt_message import build_solved_example_message_slake, build_slake_vqa_message


REPRAG_TOKEN = "<RepRAG>"
PARAPHRASE_INSTRUCTIONS = [
    "Background: {reprag_token} means the same as",
    "Background: {reprag_token} Can you put the above sentences in your own terms?",
    "Background: {reprag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {reprag_token}\n(2)",
    "Background: {reprag_token} is a paraphrase of what?",
]


# =========================
# 1) Helpers to filter retrieved samples
# =========================
def select_first_neighbor(batch_neighbors_qids: List[List[str]], batch_neighbors_scores: Optional[List[List[float]]], 
topk: int = 1) -> Tuple[List[List[str]], Optional[List[List[float]]]]:
    new_qids: List[List[str]] = []
    new_scores: Optional[List[List[float]]] = [] if batch_neighbors_scores is not None else None

    for i, neigh in enumerate(batch_neighbors_qids):
        picked = [str(q) for q in neigh[:topk]]
        new_qids.append(picked)
        if new_scores is not None:
            new_scores.append([float(s) for s in batch_neighbors_scores[i][:topk]])

    return new_qids, new_scores


def random_span_from_text(text: str, tokenizer, target_len: int, source_max_len: Optional[int] = None) -> str:
    if not text:
        return "."

    if source_max_len is None:
        model_max = getattr(tokenizer, "model_max_length", 4096)
        source_max_len = min(model_max, max(target_len, 4 * target_len))

    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=source_max_len, padding=False, return_tensors=None)["input_ids"]

    if len(ids) <= target_len:
        span_ids = ids
    else:
        start = random.randint(0, len(ids) - target_len)
        span_ids = ids[start:start + target_len]

    span_text = tokenizer.decode(span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return span_text if span_text else "."


def tokenize_targets(processor, target_texts: List[str], max_tokens: int) -> List[List[int]]:
    tok = processor.tokenizer(target_texts, add_special_tokens=False, padding=False, truncation=True, max_length=max_tokens)
    return tok["input_ids"]


# =========================
# 2) Helpers to build teacher forcing training sequence
# =========================
def pack_left_padded_prompt_target(prompt_input_ids: torch.Tensor, prompt_lens: torch.Tensor, target_ids_list: list, tgt_lens: torch.Tensor, 
pad_id: int, device: torch.device, attn_dtype: torch.dtype):
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
            prompt_tokens = prompt_input_ids[b, T_prompt - p_len:T_prompt]
            input_ids[b, start:start + p_len] = prompt_tokens
            attn_mask[b, start:start + p_len] = 1

        if t_len > 0:
            tgt_tokens = torch.as_tensor(target_ids_list[b], device=device, dtype=torch.long)
            if tgt_tokens.numel() != t_len:
                tgt_tokens = tgt_tokens[:t_len]
                if tgt_tokens.numel() < t_len:
                    tgt_tokens = torch.cat([tgt_tokens, torch.full((t_len - tgt_tokens.numel(),), pad_id, device=device, 
                    dtype=torch.long)], dim=0)
            input_ids[b, start + p_len:start + p_len + t_len] = tgt_tokens
            attn_mask[b, start + p_len:start + p_len + t_len] = 1

    return input_ids, attn_mask, starts


def make_labels_supervise_only_target_span(input_ids: torch.Tensor, starts: torch.Tensor, prompt_lens: torch.Tensor, 
tgt_lens: torch.Tensor) -> torch.Tensor:
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



def stage1_pretrain_step(args, base_model, qwen_model, processor, mlp, optimizer, device, model_dtype, batch_tasks, train_emb_normed, train_qids,
qid_to_row, index, train_task_map, writer, global_step, reprag_token_id: int, pbar=None):
    batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch(
        batch_tasks=batch_tasks,
        train_emb_normed=train_emb_normed,
        train_qids=train_qids,
        qid_to_row=qid_to_row,
        index=index,
        top_k=50,
    )
    assert batch_neighbors_scores is not None, "retrieve_neighbors_for_batch must return scores for stage1."

    K_eff = 1
    batch_neighbors_qids, batch_neighbors_scores = select_first_neighbor(batch_neighbors_qids=batch_neighbors_qids,
    batch_neighbors_scores=batch_neighbors_scores, topk=K_eff)

    target_len = int(args.stage1_target_max_tokens)
    override_texts = []
    evidence_texts = []

    for neigh in batch_neighbors_qids:
        if len(neigh) == 0:
            span = "Solved example: Question: . | Answer: ."
            override_texts.append([span])
            evidence_texts.append(span)
            continue

        qid = neigh[0]
        ex = train_task_map.get(str(qid))
        if ex is None:
            span = "Solved example: Question: . | Answer: ."
            override_texts.append([span])
            evidence_texts.append(span)
            continue

        msg = build_solved_example_message_slake(ex)
        full_text = msg["content"][0]["text"]
        span = random_span_from_text(full_text, processor.tokenizer, target_len=target_len)
        override_texts.append([span])
        evidence_texts.append(span)

    assert all(REPRAG_TOKEN not in t for t in evidence_texts), "ERROR: evidence_texts accidentally contain <RepRAG> token!"

    target_ids_list = tokenize_targets(processor, evidence_texts, max_tokens=target_len)
    tgt_lens = torch.tensor([len(x) for x in target_ids_list], device=device, dtype=torch.long)

    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved_generic(
        batch_neighbors_qids=batch_neighbors_qids,
        batch_neighbors_scores=batch_neighbors_scores,
        train_task_map=train_task_map,
        build_message_fn=build_solved_example_message_slake,
        model=qwen_model,
        processor=processor,
        device=device,
        output_mode=args.embedding_output_mode,
        k_concat=K_eff,
        override_texts=override_texts,
    )
    B, K, _ = ret_emb_batch.shape
    assert K == K_eff, f"Expected K={K_eff}, got {K}"

    inj_vec = mlp(ret_emb_batch.reshape(B * K, H)).reshape(B, K, H).to(dtype=model_dtype)

    reprag_token_str = " ".join([REPRAG_TOKEN] * K)
    prompt_texts = [random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_token_str}) for _ in range(B)]
    prompt_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt").to(device)

    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attn_mask = prompt_inputs["attention_mask"]
    pad_id = processor.tokenizer.pad_token_id
    prompt_lens = prompt_attn_mask.sum(dim=1).long()

    input_ids, attn_mask, starts = pack_left_padded_prompt_target(
        prompt_input_ids=prompt_input_ids,
        prompt_lens=prompt_lens,
        target_ids_list=target_ids_list,
        tgt_lens=tgt_lens,
        pad_id=pad_id,
        device=device,
        attn_dtype=prompt_attn_mask.dtype
    )

    labels = make_labels_supervise_only_target_span(input_ids=input_ids, starts=starts, prompt_lens=prompt_lens, tgt_lens=tgt_lens)

    with torch.no_grad():
        frozen_inputs_embeds = base_model.get_input_embeddings()(input_ids)

    inputs_embeds = frozen_inputs_embeds.clone().to(dtype=model_dtype)

    mask = (input_ids == reprag_token_id)
    counts = mask.sum(dim=1).tolist()
    assert all(c == K for c in counts), f"Each sample must have K <RepRAG> tokens, got {counts}"

    expected = B * K
    num_slots = int(mask.sum().item())
    assert num_slots == expected, f"need exactly B*K <RepRAG> tokens, got {num_slots}, expected {expected}"

    inj_flat = inj_vec.reshape(expected, H).to(dtype=model_dtype)
    inputs_embeds[mask] = inj_flat

    out = base_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels, return_dict=True)
    loss = out.loss
    assert loss.requires_grad, "Loss is detached; no gradient path to mlp."

    train_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)
    inj_med = median_l2_norm(inj_flat)

    normal_mask = (attn_mask > 0) & (~mask)
    normal_med = None
    ratio = None
    if normal_mask.any():
        normal_emb = inputs_embeds[normal_mask]
        normal_med = median_l2_norm(normal_emb)
        ratio = inj_med / (normal_med + 1e-9)

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


def stage1_eval_step(args, base_model, qwen_model, processor, mlp, device, model_dtype, batch_tasks, train_emb_normed, train_qids, 
qid_to_row, index, train_task_map, writer, global_step, reprag_token_id: int, eval_prefix: str = "stage1/val", pbar=None):
    random_state = random.getstate()
    random.seed(1234)

    base_was_training = base_model.training
    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    try:
        with torch.no_grad():
            base_messages = []
            for t in batch_tasks:
                img = load_pil_or_none(t.get("image_path"))
                base_messages.append(build_slake_vqa_message(question=t["question"], image=img, answer_type=t.get("answer_type"), 
                base_type=t.get("base_type"), triple=t.get("triple")))

            batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch_by_embedding(
                batch_messages=base_messages,
                model=qwen_model,
                processor=processor,
                device=device,
                faiss_index=index,
                train_qids=train_qids,
                top_k=50,
            )
            assert batch_neighbors_scores is not None, "retrieve_neighbors_for_batch must return scores for stage1."

            K_eff = 1
            batch_neighbors_qids, batch_neighbors_scores = select_first_neighbor(batch_neighbors_qids=batch_neighbors_qids, 
            batch_neighbors_scores=batch_neighbors_scores, topk=K_eff)

            target_len = int(args.stage1_target_max_tokens)
            override_texts = []
            evidence_texts = []

            for neigh in batch_neighbors_qids:
                if len(neigh) == 0:
                    span = "Solved example: Question: . | Answer: ."
                    override_texts.append([span])
                    evidence_texts.append(span)
                    continue

                qid = neigh[0]
                ex = train_task_map.get(str(qid))
                if ex is None:
                    span = "Solved example: Question: . | Answer: ."
                    override_texts.append([span])
                    evidence_texts.append(span)
                    continue

                msg = build_solved_example_message_slake(ex)
                full_text = msg["content"][0]["text"]
                span = random_span_from_text(full_text, processor.tokenizer, target_len=target_len)
                override_texts.append([span])
                evidence_texts.append(span)

            assert all(REPRAG_TOKEN not in t for t in evidence_texts), "ERROR: evidence_texts accidentally contain <RepRAG> token!"

            target_ids_list = tokenize_targets(processor, evidence_texts, max_tokens=target_len)
            tgt_lens = torch.tensor([len(x) for x in target_ids_list], device=device, dtype=torch.long)

            cfg = base_model.config
            H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved_generic(
                batch_neighbors_qids=batch_neighbors_qids,
                batch_neighbors_scores=batch_neighbors_scores,
                train_task_map=train_task_map,
                build_message_fn=build_solved_example_message_slake,
                model=qwen_model,
                processor=processor,
                device=device,
                output_mode=args.embedding_output_mode,
                k_concat=K_eff,
                override_texts=override_texts,
            )
            B, K, _ = ret_emb_batch.shape
            assert K == K_eff, f"Expected K={K_eff}, got {K}"

            inj_vec = mlp(ret_emb_batch.reshape(B * K, H)).reshape(B, K, H).to(dtype=model_dtype)

            reprag_token_str = " ".join([REPRAG_TOKEN] * K)
            prompt_texts = [random.choice(PARAPHRASE_INSTRUCTIONS).format_map({"reprag_token": reprag_token_str}) for _ in range(B)]
            prompt_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt").to(device)

            prompt_input_ids = prompt_inputs["input_ids"]
            prompt_attn_mask = prompt_inputs["attention_mask"]
            pad_id = processor.tokenizer.pad_token_id
            prompt_lens = prompt_attn_mask.sum(dim=1).long()

            input_ids, attn_mask, starts = pack_left_padded_prompt_target(
                prompt_input_ids=prompt_input_ids,
                prompt_lens=prompt_lens,
                target_ids_list=target_ids_list,
                tgt_lens=tgt_lens,
                pad_id=pad_id,
                device=device,
                attn_dtype=prompt_attn_mask.dtype
            )

            labels = make_labels_supervise_only_target_span(input_ids=input_ids, starts=starts, prompt_lens=prompt_lens, tgt_lens=tgt_lens)

            with torch.no_grad():
                inputs_embeds = base_model.get_input_embeddings()(input_ids).to(dtype=model_dtype)

            mask = (input_ids == reprag_token_id)
            counts = mask.sum(dim=1).tolist()
            assert all(c == K for c in counts), f"Each sample must have K <RepRAG> tokens, got {counts}"

            expected = B * K
            num_slots = int(mask.sum().item())
            assert num_slots == expected, f"need exactly B*K <RepRAG> tokens, got {num_slots}, expected {expected}"

            inj_flat = inj_vec.reshape(expected, H).to(dtype=model_dtype)
            inputs_embeds[mask] = inj_flat

            out = base_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels, return_dict=True)
            loss = out.loss

            val_tok_acc, sup_ratio = token_accuracy_and_supervised_ratio_causal(out.logits, labels)
            inj_med = median_l2_norm(inj_flat)

            normal_mask = (attn_mask > 0) & (~mask)
            normal_med = None
            ratio = None
            if normal_mask.any():
                normal_emb = inputs_embeds[normal_mask]
                normal_med = median_l2_norm(normal_emb)
                ratio = inj_med / (normal_med + 1e-9)

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
        random.setstate(random_state)

    if pbar is not None:
        pbar.set_postfix({"eval_loss": f"{loss.item():.4f}"})
        pbar.update(1)

    return float(loss.item())