"""
KILT-NQ Stage-2: answer CE fine-tuning + optional KL distillation.

Student receives K <RepRAG> tokens injected with MLP(content_emb).
Teacher (when --stage2_wo_kl is NOT set) receives raw retrieved Q+A pairs
as in-context text and provides logits for KL distillation.
Target: first gold answer token sequence.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np

from VQA.utilise.KILT_NQ.KILT_NQ_mlp_train_stage1 import (
    retrieve_train_for_batch,
    lookup_content_embeddings,
)
from VQA.utilise.KILT_NQ.KILT_NQ_mlp_utilise import (
    build_kilt_nq_vqa_message_text,
    build_student_prompt_texts_kilt_nq,
    build_answer_token_ids_kilt_nq,
    compute_kilt_nq_batch_acc,
)
from VQA.utilise.KILT_NQ.KILT_NQ_post_process import extract_short_answer
from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage2 import (
    pack_prompt_answer,
    make_labels_for_answer_span_packed,
)


REPRAG_TOKEN = "<RepRAG>"


# -------------------------------------------------------
# Teacher helpers
# -------------------------------------------------------

def build_teacher_prompt_texts_kilt_nq(
    tokenizer,
    base_messages: List[Dict[str, str]],
    batch_ret_ids: List[List[str]],
    train_record_map: Dict[str, Any],
    k: int,
) -> List[str]:
    """Build teacher prompts: base question with retrieved Q+A pairs as in-context text."""
    prompts = []
    for msg, ret_ids in zip(base_messages, batch_ret_ids):
        examples = ""
        for rid in ret_ids[:k]:
            rec = train_record_map.get(rid)
            if rec is None:
                continue
            ans = rec["gold_answers"][0] if rec.get("gold_answers") else "N/A"
            examples += f"Question: {rec['question']}\nAnswer: {ans}\n\n"

        q_text = msg["content"]
        if examples:
            content = (
                "Answer the following question with a short, concise phrase or entity. "
                "Do not explain — just give the answer.\n\n"
                f"Here are some examples:\n\n{examples}"
                f"Now answer this question:\n{q_text}"
            )
        else:
            content = msg["content"]

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)
    return prompts


def get_kl_loss_kilt_nq(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    topk: Optional[int] = None,
) -> torch.Tensor:
    """KL(teacher || student) on supervised answer token positions."""
    mask = labels != -100
    if not mask.any():
        return torch.tensor(0.0, device=teacher_logits.device, requires_grad=True)

    t_log = F.log_softmax(teacher_logits[mask] / temperature, dim=-1)
    s_log = F.log_softmax(student_logits[mask] / temperature, dim=-1)

    if topk is not None:
        top_ids = t_log.topk(topk, dim=-1).indices
        t_log = t_log.gather(-1, top_ids)
        s_log = s_log.gather(-1, top_ids)

    t_prob = t_log.exp()
    return F.kl_div(s_log, t_prob, reduction="batchmean") * (temperature ** 2)


# -------------------------------------------------------
# Shared: student forward with RepRAG injection hook
# -------------------------------------------------------

def _student_forward(base_model, input_ids, attn_mask, labels, inj_flat, reprag_token_id, model_dtype):
    emb_layer = base_model.get_input_embeddings()
    fired = {"done": False}

    def _hook(module, hook_in, output):
        if fired["done"]:
            return output
        mask = hook_in[0] == reprag_token_id
        if not mask.any():
            return output
        out = output.clone()
        out[mask] = inj_flat.to(out.dtype)
        fired["done"] = True
        return out

    handle = emb_layer.register_forward_hook(_hook)
    try:
        out = base_model(input_ids=input_ids, attention_mask=attn_mask,
                         labels=labels, return_dict=True)
    finally:
        handle.remove()
    return out


# -------------------------------------------------------
# Stage 2 train step
# -------------------------------------------------------

def stage2_finetune_step(
    args, epoch, step, global_step,
    batch_records, device, tokenizer, base_model, mlp, optimizer, writer, model_dtype,
    query_emb_normed, query_id_to_row, train_faiss_index, train_ids,
    train_record_map, content_emb_map, reprag_token_id, pbar=None,
):
    if not batch_records:
        return None, None, global_step

    assert tokenizer.padding_side == "left", "Tokenizer padding must be left."

    K = int(args.stage2_finetune_topk)
    B = len(batch_records)
    wo_kl = getattr(args, "stage2_wo_kl", True)

    # ---- 1) Retrieve top-K train items ----
    batch_ret_ids, _ = retrieve_train_for_batch(
        batch_records, query_emb_normed, query_id_to_row,
        train_faiss_index, train_ids, K,
    )

    # ---- 2) Content embeddings + MLP projection [B, K, H] ----
    ret_emb_batch = lookup_content_embeddings(batch_ret_ids, content_emb_map, device, K)
    D_content = ret_emb_batch.shape[-1]
    H = base_model.config.hidden_size
    mlp_dtype = next(mlp.parameters()).dtype
    inj_raw = mlp(ret_emb_batch.reshape(B * K, D_content).to(mlp_dtype)).reshape(B, K, H)
    inj_flat = inj_raw.to(model_dtype).reshape(B * K, H)

    if writer is not None:
        writer.add_scalar("stage2/inj_norm", float(inj_raw.norm(dim=-1).mean().item()), global_step)

    # ---- 3) Tokenize gold answers ----
    ans_max = int(getattr(args, "stage2_answer_max_tokens", 16))
    answer_ids_list = build_answer_token_ids_kilt_nq(tokenizer, batch_records, ans_max)
    ans_lens = torch.tensor([len(x) for x in answer_ids_list], device=device, dtype=torch.long)
    if (ans_lens == 0).any():
        bad = [batch_records[i]["id"] for i, l in enumerate(ans_lens.tolist()) if l == 0]
        print(f"[Stage2] WARNING: empty answer tokens (ids={bad}); skipping batch.")
        return None, None, global_step

    # ---- 4) Build student prompts + pack ----
    base_messages = [build_kilt_nq_vqa_message_text(r) for r in batch_records]
    student_texts = build_student_prompt_texts_kilt_nq(tokenizer, base_messages, k_prefix=K)
    student_enc = tokenizer(
        student_texts, padding=True, return_tensors="pt",
        truncation=True, max_length=1024,
    ).to(device)

    input_ids, attn_mask, prompt_lens, _ = pack_prompt_answer(
        student_enc["input_ids"], student_enc["attention_mask"],
        answer_ids_list, ans_lens, tokenizer.pad_token_id, device,
    )
    labels = make_labels_for_answer_span_packed(input_ids, prompt_lens, ans_lens, device)

    xmask = input_ids == reprag_token_id
    counts = xmask.sum(dim=1).tolist()
    assert all(c == K for c in counts), \
        f"Expected K={K} <RepRAG> tokens per sample, got {counts}"
    assert int(xmask.sum().item()) == B * K
    assert reprag_token_id != tokenizer.unk_token_id

    # ---- 5) Teacher forward (optional KL) ----
    teacher_logits = None
    if not wo_kl:
        teacher_texts = build_teacher_prompt_texts_kilt_nq(
            tokenizer, base_messages, batch_ret_ids, train_record_map, K,
        )
        teacher_enc = tokenizer(
            teacher_texts, padding=True, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(device)
        t_input_ids, t_attn_mask, t_prompt_lens, _ = pack_prompt_answer(
            teacher_enc["input_ids"], teacher_enc["attention_mask"],
            answer_ids_list, ans_lens, tokenizer.pad_token_id, device,
        )
        t_labels = make_labels_for_answer_span_packed(t_input_ids, t_prompt_lens, ans_lens, device)
        with torch.no_grad():
            teacher_out = base_model(
                input_ids=t_input_ids, attention_mask=t_attn_mask,
                labels=t_labels, return_dict=True,
            )
            teacher_logits = teacher_out.logits

    # ---- 6) Student forward ----
    base_model.eval()
    student_out = _student_forward(
        base_model, input_ids, attn_mask, labels, inj_flat, reprag_token_id, model_dtype,
    )
    loss_ce = student_out.loss
    if loss_ce is None or not loss_ce.isfinite():
        return None, None, global_step

    # ---- 7) KL loss ----
    loss = loss_ce
    if not wo_kl and teacher_logits is not None:
        alpha_kl = float(getattr(args, "alpha_kl", 1.0))
        T_kl = float(getattr(args, "kl_temperature", 1.0))
        T_s = student_out.logits.shape[1]
        T_t = teacher_logits.shape[1]
        if T_t >= T_s:
            teacher_logits_aligned = teacher_logits[:, -T_s:, :]
        else:
            pad = torch.zeros(B, T_s - T_t, teacher_logits.shape[-1],
                              device=device, dtype=teacher_logits.dtype)
            teacher_logits_aligned = torch.cat([pad, teacher_logits], dim=1)
        loss_kl = get_kl_loss_kilt_nq(
            teacher_logits_aligned, student_out.logits, labels, temperature=T_kl,
        )
        loss = loss_ce + alpha_kl * loss_kl
        if writer is not None:
            writer.add_scalar("stage2/train_loss_kl", float(loss_kl.item()), global_step)

    # ---- 8) Periodic train accuracy (generation-based) ----
    acc = None
    acc_every = int(getattr(args, "acc_every", 20))
    if (global_step % acc_every) == 0:
        max_new = int(getattr(args, "eval_max_new_tokens", 32))
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
        emb_layer = base_model.get_input_embeddings()

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
        decoded = [extract_short_answer(d) for d in decoded]
        acc = compute_kilt_nq_batch_acc(batch_records, decoded)

    if writer is not None:
        writer.add_scalar("stage2/train_loss_ce", float(loss_ce.item()), global_step)
        writer.add_scalar("stage2/train_loss_total", float(loss.item()), global_step)
        if acc is not None:
            writer.add_scalar("stage2/train_acc", float(acc), global_step)

    # ---- 9) Backward ----
    optimizer.zero_grad(set_to_none=True)
    loss.backward()


    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
    optimizer.step()

    global_step += 1
    loss_val = float(loss.item())
    loss_ce_val = float(loss_ce.item())

    if pbar is not None:
        postfix = {"loss": f"{loss_val:.4f}", "ce": f"{loss_ce_val:.4f}"}
        if acc is not None:
            postfix["acc"] = f"{acc:.4f}"
        pbar.set_postfix(postfix)
        pbar.update(1)

    if (step + 1) % 20 == 0 and pbar is None:
        msg = (f"[Stage2][KILT-NQ][Epoch {epoch+1}] step {step+1} | "
               f"loss={loss_val:.4f} | ce={loss_ce_val:.4f}")
        if acc is not None:
            msg += f" | acc={acc:.4f}"
        print(msg)

    return loss_val, (float(acc) if acc is not None else None), global_step


# -------------------------------------------------------
# Stage 2 eval step
# -------------------------------------------------------

def stage2_eval_step(
    args, epoch, step, global_step,
    batch_records, device, tokenizer, base_model, mlp, writer, model_dtype,
    query_emb_normed, query_id_to_row, train_faiss_index, train_ids,
    train_record_map, content_emb_map, reprag_token_id,
    eval_prefix="stage2/eval", pbar=None,
):
    if not batch_records:
        return None, None

    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    try:
        with torch.no_grad():
            K = int(getattr(args, "stage2_finetune_topk", getattr(args, "top_k", 1)))
            B = len(batch_records)

            batch_ret_ids, _ = retrieve_train_for_batch(
                batch_records, query_emb_normed, query_id_to_row,
                train_faiss_index, train_ids, K,
            )
            ret_emb_batch = lookup_content_embeddings(batch_ret_ids, content_emb_map, device, K)
            D_content = ret_emb_batch.shape[-1]
            H = base_model.config.hidden_size
            mlp_dtype = next(mlp.parameters()).dtype
            inj_raw = mlp(ret_emb_batch.reshape(B * K, D_content).to(mlp_dtype)).reshape(B, K, H)
            inj_flat = inj_raw.to(model_dtype).reshape(B * K, H)

            base_messages = [build_kilt_nq_vqa_message_text(r) for r in batch_records]
            student_texts = build_student_prompt_texts_kilt_nq(tokenizer, base_messages, k_prefix=K)
            student_enc = tokenizer(
                student_texts, padding=True, return_tensors="pt",
                truncation=True, max_length=1024,
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
                # ---- Generate predictions ----
                max_new = int(getattr(args, "eval_max_new_tokens", 32))
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
                decoded = [extract_short_answer(d) for d in decoded]
                acc = compute_kilt_nq_batch_acc(batch_records, decoded)

                # ---- CE loss ----
                ce_loss_val = None
                if bool(getattr(args, "eval_log_ce", True)):
                    ans_max = int(getattr(args, "stage2_answer_max_tokens", 16))
                    answer_ids_list = build_answer_token_ids_kilt_nq(
                        tokenizer, batch_records, ans_max
                    )
                    ans_lens = torch.tensor(
                        [len(x) for x in answer_ids_list], device=device, dtype=torch.long
                    )
                    prompt_lens = prompt_attn_mask.sum(dim=1).long()
                    input_ids_ce, attn_mask_ce, prompt_lens_ce, _ = pack_prompt_answer(
                        prompt_input_ids, prompt_attn_mask,
                        answer_ids_list, ans_lens, tokenizer.pad_token_id, device,
                    )
                    labels_ce = make_labels_for_answer_span_packed(
                        input_ids_ce, prompt_lens_ce, ans_lens, device
                    )
                    fired["done"] = False
                    ce_out = base_model(
                        input_ids=input_ids_ce, attention_mask=attn_mask_ce,
                        labels=labels_ce, return_dict=True,
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