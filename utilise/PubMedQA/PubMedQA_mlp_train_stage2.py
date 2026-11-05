"""
PubMedQA Stage-2 finetuning: answer CE loss + optional KL distillation.

Student receives K <RepRAG> tokens injected with MLP(content_emb).
Teacher (optional) receives the raw passage text as reference evidence.

Key differences from SLAKE:
  - Text-only model: uses tokenizer, not processor
  - Retrieval uses pre-computed content embeddings (corpus pool)
  - Answers are option letters A/B/C (yes/no/maybe)
  - No image inputs
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage1 import (
    retrieve_corpus_for_batch,
    lookup_content_embeddings,
)
from VQA.utilise.PubMedQA.PubMedQA_mlp_utilise import (
    build_mcq_vqa_message_text,
    build_answer_token_ids_pubmedqa,
    compute_pubmedqa_batch_acc,
    build_teacher_evidence_texts_pubmedqa,
    build_teacher_prompt_texts_pubmedqa,
    build_student_prompt_texts_pubmedqa,
    get_kl_loss_pubmedqa,
)

REPRAG_TOKEN = "<RepRAG>"


# -------------------------------------------------------
# Label helpers
# -------------------------------------------------------

def make_labels_for_answer_span_packed(
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    ans_lens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    B, T_total = input_ids.shape
    labels = torch.full((B, T_total), -100, device=device, dtype=torch.long)
    total_lens = prompt_lens + ans_lens
    for b in range(B):
        p_len = int(prompt_lens[b].item())
        a_len = int(ans_lens[b].item())
        if a_len <= 0:
            continue
        total = int(total_lens[b].item())
        seq_start = T_total - total
        ans_start = seq_start + p_len
        ans_end = ans_start + a_len
        labels[b, ans_start:ans_end] = input_ids[b, ans_start:ans_end]
    return labels


def pack_prompt_answer(
    prompt_input_ids: torch.Tensor,
    prompt_attn_mask: torch.Tensor,
    answer_ids_list: List[List[int]],
    ans_lens: torch.Tensor,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack left-padded prompt + right-appended answer tokens."""
    B = prompt_input_ids.shape[0]
    T_prompt = prompt_input_ids.shape[1]
    prompt_lens = prompt_attn_mask.sum(dim=1).long()
    total_lens = prompt_lens + ans_lens
    T_total = int(total_lens.max().item())

    input_ids = torch.full((B, T_total), pad_id, device=device, dtype=torch.long)
    attn_mask = torch.zeros((B, T_total), device=device, dtype=prompt_attn_mask.dtype)

    for b in range(B):
        p_len = int(prompt_lens[b].item())
        a_len = int(ans_lens[b].item())
        total = p_len + a_len
        start = T_total - total

        if p_len > 0:
            prompt_tokens = prompt_input_ids[b, T_prompt - p_len : T_prompt]
            input_ids[b, start : start + p_len] = prompt_tokens
            attn_mask[b, start : start + p_len] = 1

        if a_len > 0:
            ans_tokens = torch.as_tensor(
                answer_ids_list[b], device=device, dtype=torch.long
            )
            input_ids[b, start + p_len : start + p_len + a_len] = ans_tokens
            attn_mask[b, start + p_len : start + p_len + a_len] = 1

    return input_ids, attn_mask, prompt_lens, total_lens


# -------------------------------------------------------
# Stage 2 train step
# -------------------------------------------------------

def stage2_finetune_step(
    args,
    epoch: int,
    step: int,
    global_step: int,
    batch_tasks: List[Dict[str, Any]],
    device,
    tokenizer,
    base_model,
    mlp,
    optimizer,
    writer,
    model_dtype,
    query_emb_normed: np.ndarray,
    query_qid_to_row: Dict[str, int],
    corpus_faiss_index,
    corpus_qids: List[str],
    content_emb_map: Dict[str, np.ndarray],
    corpus_task_map: Dict[str, Dict[str, Any]],
    reprag_token_id: int,
    pbar=None,
):
    if not batch_tasks:
        return None, None, global_step

    assert tokenizer.padding_side == "left", "Tokenizer padding must be left."

    K = int(args.stage2_finetune_topk)

    # ---- 1) Build base messages + collect gold answers ----
    base_messages = [build_mcq_vqa_message_text(t) for t in batch_tasks]
    gold_texts = [t.get("gold_answer", "") or "" for t in batch_tasks]

    # ---- 2) Retrieve top-K corpus items ----
    batch_corpus_qids, _ = retrieve_corpus_for_batch(
        batch_tasks=batch_tasks,
        query_emb_normed=query_emb_normed,
        query_qid_to_row=query_qid_to_row,
        corpus_faiss_index=corpus_faiss_index,
        corpus_qids=corpus_qids,
        top_k=K,
    )

    # ---- 3) Look up content embeddings [B, K, D_content] ----
    H = base_model.config.hidden_size
    ret_emb_batch = lookup_content_embeddings(
        batch_corpus_qids=batch_corpus_qids,
        content_emb_map=content_emb_map,
        device=device,
        k=K,
    )
    B = ret_emb_batch.shape[0]
    D_content = ret_emb_batch.shape[-1]

    # ---- 4) Tokenize gold answers (option letters A/B/C) ----
    answer_ids_list = build_answer_token_ids_pubmedqa(tokenizer, gold_texts)
    ans_lens = torch.tensor(
        [len(x) for x in answer_ids_list], device=device, dtype=torch.long
    )
    if (ans_lens == 0).any():
        bad = [batch_tasks[i]["qid"] for i, l in enumerate(ans_lens.tolist()) if l == 0]
        print(f"[Stage2] WARNING: samples with empty answer tokens (qids={bad}); skipping batch to prevent NaN loss.")
        return None, None, global_step

    # ---- 5) Teacher forward (optional KL) ----
    teacher_logits = None
    teacher_labels = None

    if not args.stage2_wo_kl:
        teacher_max_chars = int(getattr(args, "teacher_max_chars", 5000))

        teacher_evidence = build_teacher_evidence_texts_pubmedqa(
            batch_neighbors_qids=batch_corpus_qids,
            corpus_task_map=corpus_task_map,
            topk=K,
            max_chars=teacher_max_chars,
        )
        teacher_prompt_texts = build_teacher_prompt_texts_pubmedqa(
            tokenizer, base_messages, teacher_evidence
        )

        teacher_enc = tokenizer(
            teacher_prompt_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        t_input_ids, t_attn_mask, t_prompt_lens, _ = pack_prompt_answer(
            prompt_input_ids=teacher_enc["input_ids"],
            prompt_attn_mask=teacher_enc["attention_mask"],
            answer_ids_list=answer_ids_list,
            ans_lens=ans_lens,
            pad_id=tokenizer.pad_token_id,
            device=device,
        )

        teacher_labels = make_labels_for_answer_span_packed(
            input_ids=t_input_ids,
            prompt_lens=t_prompt_lens,
            ans_lens=ans_lens,
            device=device,
        )

        base_model.eval()
        with torch.no_grad():
            teacher_out = base_model(
                input_ids=t_input_ids,
                attention_mask=t_attn_mask,
                return_dict=True,
            )
            teacher_logits = teacher_out.logits

    # ---- 6) Project injection embeddings ----
    mlp_dtype = next(mlp.parameters()).dtype
    inj_raw = mlp(ret_emb_batch.reshape(B * K, D_content).to(mlp_dtype)).reshape(B, K, H)
    inj_flat = inj_raw.to(model_dtype).reshape(B * K, H)

    if writer is not None:
        writer.add_scalar(
            "stage2/inj_norm", float(inj_raw.norm(dim=-1).mean().item()), global_step
        )

    # ---- 7) Build student prompt inputs ----
    student_prompt_texts = build_student_prompt_texts_pubmedqa(
        tokenizer, base_messages, k_prefix=K
    )
    student_enc = tokenizer(
        student_prompt_texts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    input_ids, attn_mask, prompt_lens, _ = pack_prompt_answer(
        prompt_input_ids=student_enc["input_ids"],
        prompt_attn_mask=student_enc["attention_mask"],
        answer_ids_list=answer_ids_list,
        ans_lens=ans_lens,
        pad_id=tokenizer.pad_token_id,
        device=device,
    )

    labels = make_labels_for_answer_span_packed(
        input_ids=input_ids,
        prompt_lens=prompt_lens,
        ans_lens=ans_lens,
        device=device,
    )

    xmask = input_ids == reprag_token_id
    counts = xmask.sum(dim=1).tolist()
    assert all(
        c == K for c in counts
    ), f"Expected K={K} <RepRAG> tokens per sample, got {counts}"
    assert int(xmask.sum().item()) == B * K
    assert reprag_token_id != tokenizer.unk_token_id

    if not args.stage2_wo_kl and teacher_labels is not None:
        t_sup = int((teacher_labels != -100).sum().item())
        s_sup = int((labels != -100).sum().item())
        if t_sup != s_sup:
            raise RuntimeError(
                f"Teacher/student supervised token mismatch: {t_sup} vs {s_sup}. "
                "Likely teacher prompt too long / truncation asymmetry."
            )

    # ---- 8) Register embedding injection hook ----
    emb_layer = base_model.get_input_embeddings()

    def _inject_hook(module, inputs, output):
        ids = inputs[0]
        mask = ids == reprag_token_id
        if not mask.any():
            return output
        out = output.clone()
        out[mask] = inj_flat.to(out.dtype)
        return out

    handle = emb_layer.register_forward_hook(_inject_hook)

    try:
        # ---- 9) Student forward ----
        base_model.eval()
        student_out = base_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True,
        )
        loss_main = student_out.loss
        student_logits = student_out.logits

        # ---- 10) KL distillation ----
        if not args.stage2_wo_kl and teacher_logits is not None:
            alpha_kl = float(getattr(args, "alpha_kl", 1.0))
            T_kl = float(getattr(args, "kl_temperature", 1.0))
            distill_topk = getattr(args, "distill_topk", None)

            loss_kl = get_kl_loss_pubmedqa(
                teacher_logits=teacher_logits.detach(),
                student_logits=student_logits,
                teacher_labels=teacher_labels,
                student_labels=labels,
                temperature=T_kl,
                distill_topk=distill_topk,
            )
            loss = loss_main + alpha_kl * loss_kl
            if writer is not None:
                writer.add_scalar("stage2/train_loss_kl", float(loss_kl.item()), global_step)
        else:
            loss = loss_main

        # ---- 11) Periodic train accuracy ----
        acc = None
        acc_every = int(getattr(args, "acc_every", 20))
        if (global_step % acc_every) == 0:
            max_new = int(getattr(args, "eval_max_new_tokens", 8))
            min_new = int(getattr(args, "eval_min_new_tokens", 1))
            do_sample = bool(getattr(args, "eval_do_sample", False))

            prompt_input_ids_only = student_enc["input_ids"]
            prompt_attn_only = student_enc["attention_mask"]
            prompt_len = prompt_input_ids_only.shape[1]
            min_length = prompt_len + min_new

            gen_args = dict(
                max_new_tokens=max_new,
                do_sample=do_sample,
                num_beams=int(getattr(args, "eval_num_beams", 1)),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                min_length=min_length,
            )
            if do_sample:
                temperature = getattr(args, "eval_temperature", None)
                top_p = getattr(args, "eval_top_p", None)
                if temperature is not None:
                    gen_args["temperature"] = float(temperature)
                if top_p is not None:
                    gen_args["top_p"] = float(top_p)

            fired = {"done": False}

            def _gen_hook(module, hook_inputs, output):
                if fired["done"]:
                    return output
                ids = hook_inputs[0]
                mask = ids == reprag_token_id
                if not mask.any():
                    return output
                out = output.clone()
                out[mask] = inj_flat.to(out.dtype)
                fired["done"] = True
                return out

            gen_handle = emb_layer.register_forward_hook(_gen_hook)
            try:
                _dynamo_was_disabled = torch._dynamo.config.disable
                torch._dynamo.config.disable = True
                try:
                    with torch.no_grad():
                        gen_out = base_model.generate(
                            input_ids=prompt_input_ids_only,
                            attention_mask=prompt_attn_only,
                            **gen_args,
                        )
                finally:
                    torch._dynamo.config.disable = _dynamo_was_disabled
            finally:
                gen_handle.remove()

            trimmed = [
                seq[prompt_len:] if seq.shape[0] > prompt_len else seq
                for seq in gen_out
            ]
            decoded = tokenizer.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            acc = compute_pubmedqa_batch_acc(batch_tasks, decoded)

        if writer is not None:
            writer.add_scalar("stage2/train_loss_ce", float(loss_main.item()), global_step)
            writer.add_scalar("stage2/train_loss_total", float(loss.item()), global_step)
            if acc is not None:
                writer.add_scalar("stage2/train_acc", float(acc), global_step)

        # ---- 12) Backward ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()


        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        loss_val = float(loss.item())
        loss_ce_val = float(loss_main.item())

        if pbar is not None:
            postfix = {"loss": f"{loss_val:.4f}", "ce": f"{loss_ce_val:.4f}"}
            if acc is not None:
                postfix["acc"] = f"{acc:.4f}"
            pbar.set_postfix(postfix)
            pbar.update(1)

        if (step + 1) % 20 == 0 and pbar is None:
            msg = (
                f"[Stage2][PubMedQA][Epoch {epoch+1}] step {step+1} | "
                f"loss={loss_val:.4f} | ce={loss_ce_val:.4f}"
            )
            if acc is not None:
                msg += f" | acc={acc:.4f}"
            print(msg)

        return loss_val, (float(acc) if acc is not None else None), global_step

    finally:
        handle.remove()


# -------------------------------------------------------
# Stage 2 eval step
# -------------------------------------------------------

def stage2_eval_step(
    args,
    epoch: int,
    step: int,
    global_step: int,
    batch_tasks: List[Dict[str, Any]],
    device,
    tokenizer,
    base_model,
    mlp,
    writer,
    model_dtype,
    query_emb_normed: np.ndarray,
    query_qid_to_row: Dict[str, int],
    corpus_faiss_index,
    corpus_qids: List[str],
    content_emb_map: Dict[str, np.ndarray],
    corpus_task_map: Dict[str, Dict[str, Any]],
    reprag_token_id: int,
    eval_prefix: str = "stage2/eval",
    pbar=None,
):
    if not batch_tasks:
        return None, None

    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    try:
        with torch.no_grad():
            K = int(getattr(args, "stage2_finetune_topk", getattr(args, "top_k", 1)))

            base_messages = [build_mcq_vqa_message_text(t) for t in batch_tasks]
            gold_texts = [t.get("gold_answer", "") or "" for t in batch_tasks]

            batch_corpus_qids, _ = retrieve_corpus_for_batch(
                batch_tasks=batch_tasks,
                query_emb_normed=query_emb_normed,
                query_qid_to_row=query_qid_to_row,
                corpus_faiss_index=corpus_faiss_index,
                corpus_qids=corpus_qids,
                top_k=K,
            )

            H = base_model.config.hidden_size
            ret_emb_batch = lookup_content_embeddings(
                batch_corpus_qids=batch_corpus_qids,
                content_emb_map=content_emb_map,
                device=device,
                k=K,
            )
            B = ret_emb_batch.shape[0]
            D_content = ret_emb_batch.shape[-1]

            mlp_dtype = next(mlp.parameters()).dtype
            inj_raw = mlp(ret_emb_batch.reshape(B * K, D_content).to(mlp_dtype)).reshape(B, K, H)
            inj_flat = inj_raw.to(dtype=model_dtype).reshape(B * K, H)

            # Build student prompt inputs
            student_prompt_texts = build_student_prompt_texts_pubmedqa(
                tokenizer, base_messages, k_prefix=K
            )
            student_enc = tokenizer(
                student_prompt_texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            prompt_input_ids = student_enc["input_ids"]
            prompt_attn_mask = student_enc["attention_mask"]

            xmask = prompt_input_ids == reprag_token_id
            counts = xmask.sum(dim=1).tolist()
            if not all(c == K for c in counts):
                raise RuntimeError(
                    f"[Eval] expected K={K} <RepRAG> tokens per sample, got {counts}"
                )

            emb_layer = base_model.get_input_embeddings()
            fired = {"done": False}

            def _inject_hook(module, hook_inputs, output):
                if fired["done"]:
                    return output
                ids = hook_inputs[0]
                mask = ids == reprag_token_id
                if not mask.any():
                    return output
                out = output.clone()
                out[mask] = inj_flat.to(out.dtype)
                fired["done"] = True
                return out

            handle = emb_layer.register_forward_hook(_inject_hook)

            try:
                # Generate predictions
                max_new = int(getattr(args, "eval_max_new_tokens", 8))
                min_new = int(getattr(args, "eval_min_new_tokens", 1))
                do_sample = bool(getattr(args, "eval_do_sample", False))

                prompt_len = prompt_input_ids.shape[1]
                min_length = prompt_len + min_new

                gen_args = dict(
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    num_beams=int(getattr(args, "eval_num_beams", 1)),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    min_length=min_length,
                )
                if do_sample:
                    temperature = getattr(args, "eval_temperature", None)
                    top_p = getattr(args, "eval_top_p", None)
                    if temperature is not None:
                        gen_args["temperature"] = float(temperature)
                    if top_p is not None:
                        gen_args["top_p"] = float(top_p)

                _dynamo_was_disabled = torch._dynamo.config.disable
                torch._dynamo.config.disable = True
                try:
                    gen_out = base_model.generate(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attn_mask,
                        **gen_args,
                    )
                finally:
                    torch._dynamo.config.disable = _dynamo_was_disabled

                trimmed = [
                    seq[prompt_len:] if seq.shape[0] > prompt_len else seq
                    for seq in gen_out
                ]
                decoded = tokenizer.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                acc = compute_pubmedqa_batch_acc(batch_tasks, decoded)

                # CE loss
                ce_loss_val = None
                if bool(getattr(args, "eval_log_ce", True)):
                    answer_ids_list = build_answer_token_ids_pubmedqa(tokenizer, gold_texts)
                    ans_lens = torch.tensor(
                        [len(x) for x in answer_ids_list], device=device, dtype=torch.long
                    )
                    prompt_lens = prompt_attn_mask.sum(dim=1).long()
                    total_lens = prompt_lens + ans_lens
                    T_total = int(total_lens.max().item())
                    pad_id = tokenizer.pad_token_id

                    ce_ids = torch.full((B, T_total), pad_id, device=device, dtype=torch.long)
                    ce_mask = torch.zeros((B, T_total), device=device, dtype=prompt_attn_mask.dtype)

                    T_prompt = prompt_input_ids.shape[1]
                    for b in range(B):
                        p_len = int(prompt_lens[b].item())
                        a_len = int(ans_lens[b].item())
                        total = p_len + a_len
                        start = T_total - total

                        if p_len > 0:
                            ce_ids[b, start : start + p_len] = prompt_input_ids[
                                b, T_prompt - p_len : T_prompt
                            ]
                            ce_mask[b, start : start + p_len] = 1

                        if a_len > 0:
                            ce_ids[b, start + p_len : start + p_len + a_len] = torch.as_tensor(
                                answer_ids_list[b], device=device, dtype=torch.long
                            )
                            ce_mask[b, start + p_len : start + p_len + a_len] = 1

                    ce_labels = make_labels_for_answer_span_packed(
                        input_ids=ce_ids,
                        prompt_lens=prompt_lens,
                        ans_lens=ans_lens,
                        device=device,
                    )

                    fired["done"] = False
                    ce_out = base_model(
                        input_ids=ce_ids,
                        attention_mask=ce_mask,
                        labels=ce_labels,
                        return_dict=True,
                    )
                    ce_loss_val = float(ce_out.loss.item())

                if writer is not None:
                    writer.add_scalar(f"{eval_prefix}/acc", float(acc), global_step)
                    if ce_loss_val is not None:
                        writer.add_scalar(f"{eval_prefix}/ce", float(ce_loss_val), global_step)

                if pbar is not None:
                    postfix = {"acc": f"{acc:.4f}"}
                    if ce_loss_val is not None:
                        postfix["ce"] = f"{ce_loss_val:.4f}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                if (step + 1) % 50 == 0 and pbar is None:
                    msg = f"[{eval_prefix}] epoch={epoch+1} step={step+1} acc={acc:.4f}"
                    if ce_loss_val is not None:
                        msg += f" ce={ce_loss_val:.4f}"
                    print(msg)

                return ce_loss_val, acc

            finally:
                handle.remove()

    finally:
        base_model.eval()
        mlp.train(mlp_was_training)
