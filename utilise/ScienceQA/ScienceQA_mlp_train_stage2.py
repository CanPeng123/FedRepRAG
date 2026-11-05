import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from qwen_vl_utils import process_vision_info
from typing import Optional

from VQA.utilise.retrieval_helper import retrieve_neighbors_for_batch
from VQA.utilise.data_process import load_image, load_pil_or_none
from VQA.utilise.ScienceQA.ScienceQA_prompt_message import build_multi_image_vqa_message
from VQA.utilise.ScienceQA.ScienceQA_data_process import discover_q_choice_by_convention
from VQA.utilise.ScienceQA.ScienceQA_post_process import parse_option_letter

# from VQA.utilise.ScienceQA.ScienceQA_retrival_utilise import build_prefix_last_layer_for_batch_from_retrieved
from VQA.utilise.ScienceQA.ScienceQA_mlp_train_stage1 import build_prefix_last_layer_for_batch_from_retrieved

from VQA.utilise.mlp_utilise.mlp_train_stage2_utilise import build_teacher_evidence_texts, build_teacher_prompt_texts
from VQA.utilise.mlp_utilise.mlp_train_stage2_utilise import build_student_prompt_texts, get_kl_loss_reprag_style


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REPRAG_TOKEN = "<RepRAG>"


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def build_messages_and_golds(batch_tasks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    base_messages = []
    golds = []

    for t in batch_tasks:
        ctx = t.get("context") or ""
        image_dir = t.get("image_dir") or (os.path.dirname(t["image_path"]) if t.get("image_path") else None)

        if image_dir:
            q_path, choice_paths = discover_q_choice_by_convention(image_dir, len(t["choices"]))
        else:
            q_path, choice_paths = None, [None] * len(t["choices"])

        q_img = load_pil_or_none(q_path)
        choice_imgs = [load_pil_or_none(p) for p in choice_paths]

        msg = build_multi_image_vqa_message(question=t["question"], choices=t["choices"], question_img=q_img, 
        choice_imgs=choice_imgs, context=ctx)
        base_messages.append(msg)
        golds.append(int(t["gold"]))  # answer

    return base_messages, torch.tensor(golds, dtype=torch.long)


def build_answer_token_ids(processor, gold_indices: torch.Tensor) -> List[List[int]]:
    answers = [" " + LETTERS[g] for g in gold_indices.tolist()]
    tok = processor.tokenizer(answers, add_special_tokens=False, padding=False)
    return tok["input_ids"]


def make_labels_for_answer_span_packed(input_ids: torch.Tensor, prompt_lens: torch.Tensor, ans_lens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    labels = -100 everywhere except the answer span (teacher forcing).
    Works for left-padded packed sequences (no extra prefix outside input_ids).
    """
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



def stage2_finetune_step(args, epoch: int, step: int, global_step: int, batch_tasks, device, processor, qwen_model, base_model, mlp, optimizer, writer,
model_dtype, train_emb_normed, train_qids, qid_to_row, index, train_task_map, reprag_token_id: int, pbar=None):

    if not batch_tasks:
        return None, None, global_step

    assert processor.tokenizer.padding_side == "left", "Your packing slices assume left padding (valid tokens at the end)."

    # ---- 1) Build base messages + gold ----
    base_messages, gold_indices = build_messages_and_golds(batch_tasks)
    gold_indices = gold_indices.to(device)

    # ---- 2) Retrieve neighbors ----
    batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch(
        batch_tasks=batch_tasks,
        train_emb_normed=train_emb_normed,
        train_qids=train_qids,
        qid_to_row=qid_to_row,
        index=index,
        top_k=args.stage2_finetune_topk,
    )

    # ---- 3) Retrieval embeddings [B,K,H] ----
    with torch.no_grad():
        ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved(
            batch_neighbors_qids=batch_neighbors_qids,
            batch_neighbors_scores=batch_neighbors_scores,
            train_task_map=train_task_map,
            model=qwen_model,
            processor=processor,
            device=device,
            output_mode=args.embedding_output_mode,
            k_concat=args.stage2_finetune_topk,
        )  # [B,K,H]

    H = base_model.config.hidden_size
    assert ret_emb_batch.dim() == 3 and ret_emb_batch.size(-1) == H
    B, K, _ = ret_emb_batch.shape
    assert K == args.stage2_finetune_topk, f"K mismatch: got {K}, expected {args.stage2_finetune_topk}"

    # ---- 4) Tokenize gold answers ----
    answer_ids_list = build_answer_token_ids(processor, gold_indices)
    ans_lens = torch.tensor([len(x) for x in answer_ids_list], device=device, dtype=torch.long)

    # ---- 5) Teacher forward (optional) ----
    image_inputs, video_inputs = process_vision_info(base_messages)

    if not args.stage2_wo_kl:
        teacher_max_chars = int(getattr(args, "teacher_max_chars", 1200))

        teacher_evidence = build_teacher_evidence_texts(
            batch_neighbors_qids=batch_neighbors_qids,
            train_task_map=train_task_map,
            topk=K,
            max_chars=teacher_max_chars,
        )
        teacher_prompt_texts = build_teacher_prompt_texts(processor, base_messages, teacher_evidence)

        teacher_prompt_inputs = processor(text=teacher_prompt_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        teacher_prompt_input_ids = teacher_prompt_inputs["input_ids"]      # [B, Tt_prompt]
        teacher_prompt_attn_mask = teacher_prompt_inputs["attention_mask"]
        pad_id = processor.tokenizer.pad_token_id
        teacher_prompt_lens = teacher_prompt_attn_mask.sum(dim=1).long()

        # Pack teacher prompt + answer (LEFT padded, right-aligned content)
        teacher_total_lens = teacher_prompt_lens + ans_lens
        teacher_T_total = int(teacher_total_lens.max().item())

        teacher_input_ids = torch.full((B, teacher_T_total), pad_id, device=device, dtype=torch.long)
        teacher_attn_mask = torch.zeros((B, teacher_T_total), device=device, dtype=teacher_prompt_attn_mask.dtype)

        T_prompt_t = teacher_prompt_input_ids.size(1)
        for b in range(B):
            p_len = int(teacher_prompt_lens[b].item())
            a_len = int(ans_lens[b].item())
            total = p_len + a_len
            start = teacher_T_total - total

            if p_len > 0:
                prompt_tokens = teacher_prompt_input_ids[b, T_prompt_t - p_len : T_prompt_t]
                teacher_input_ids[b, start : start + p_len] = prompt_tokens
                teacher_attn_mask[b, start : start + p_len] = 1

            if a_len > 0:
                ans_tokens = torch.tensor(answer_ids_list[b], device=device, dtype=torch.long)
                teacher_input_ids[b, start + p_len : start + p_len + a_len] = ans_tokens
                teacher_attn_mask[b, start + p_len : start + p_len + a_len] = 1

        teacher_labels = make_labels_for_answer_span_packed(
            input_ids=teacher_input_ids,
            prompt_lens=teacher_prompt_lens,
            ans_lens=ans_lens,
            device=device,
        )

        # Teacher forward (no hook) - keep LLM eval & frozen
        base_model.eval()
        with torch.no_grad():
            teacher_forward_kwargs = {
                "input_ids": teacher_input_ids,
                "attention_mask": teacher_attn_mask,
                "return_dict": True,
            }
            for k_in, v_in in teacher_prompt_inputs.items():
                if k_in in ("input_ids", "attention_mask"):
                    continue
                teacher_forward_kwargs[k_in] = v_in

            teacher_outputs = base_model(**teacher_forward_kwargs)
            teacher_logits = teacher_outputs.logits  # [B, teacher_T_total, vocab]

    # ---- 6) Project injection embeddings ----
    mlp_dtype = next(mlp.parameters()).dtype
    inj_raw = mlp(ret_emb_batch.reshape(B*K, H).to(mlp_dtype)).reshape(B,K,H)
    inj_flat = inj_raw.to(model_dtype).reshape(B*K,H)

    # ---- 7) Student prompts with K <RepRAG> tokens ----
    prompt_texts = build_student_prompt_texts(processor, base_messages, k_prefix=K)
    prompt_inputs = processor(text=prompt_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    prompt_input_ids = prompt_inputs["input_ids"]      # [B, T_prompt]
    prompt_attn_mask = prompt_inputs["attention_mask"] # [B, T_prompt]
    pad_id = processor.tokenizer.pad_token_id
    prompt_lens = prompt_attn_mask.sum(dim=1).long()   # includes RepRAG tokens now

    # pack student prompt + answer
    total_lens = prompt_lens + ans_lens
    T_total = int(total_lens.max().item())

    input_ids = torch.full((B, T_total), pad_id, device=device, dtype=torch.long)
    attn_mask = torch.zeros((B, T_total), device=device, dtype=prompt_attn_mask.dtype)

    T_prompt = prompt_input_ids.size(1)
    for b in range(B):
        p_len = int(prompt_lens[b].item())
        a_len = int(ans_lens[b].item())
        total = p_len + a_len
        start = T_total - total  # right-align content

        if p_len > 0:
            prompt_tokens = prompt_input_ids[b, T_prompt - p_len : T_prompt]
            input_ids[b, start : start + p_len] = prompt_tokens
            attn_mask[b, start : start + p_len] = 1

        if a_len > 0:
            ans_tokens = torch.tensor(answer_ids_list[b], device=device, dtype=torch.long)
            input_ids[b, start + p_len : start + p_len + a_len] = ans_tokens
            attn_mask[b, start + p_len : start + p_len + a_len] = 1

    # ---- 7) Labels (only answer span) ----
    labels = make_labels_for_answer_span_packed(input_ids=input_ids, prompt_lens=prompt_lens, ans_lens=ans_lens, device=device)

    if not args.stage2_wo_kl:
        t_sup = int((teacher_labels != -100).sum().item())
        s_sup = int((labels != -100).sum().item())
        if t_sup != s_sup:
            raise RuntimeError(
                f"Teacher/student supervised token mismatch: {t_sup} vs {s_sup}. "
                f"Likely teacher prompt too long / truncation."
            )

    # ---- 8) RepRAG-style injection (no gate scaling) ----
    if writer is not None:
        writer.add_scalar("stage2/inj_norm", float(inj_raw.norm(dim=-1).mean().item()), global_step)

    # ensure exactly K RepRAG tokens per sample
    xmask = (input_ids == reprag_token_id)
    counts = xmask.sum(dim=1).tolist()
    assert all(c == K for c in counts), f"Each sample must have K <RepRAG> tokens, got {counts}"
    assert int(xmask.sum().item()) == B*K, "Total <RepRAG> slots mismatch."
    assert reprag_token_id != processor.tokenizer.unk_token_id

    # ---- 9) Register embedding hook to overwrite <RepRAG> ----
    emb_layer = base_model.get_input_embeddings()

    def _inject_hook(module, inputs, output):
        ids = inputs[0]  # [B,T]
        mask = (ids == reprag_token_id)
        if not mask.any():
            return output
        out = output.clone()
        out[mask] = inj_flat.to(out.dtype)
        return out

    handle = emb_layer.register_forward_hook(_inject_hook)

    try:
        # ---- 10) Forward (use input_ids so VL vision overwrite works) ----
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }
        for k_in, v_in in prompt_inputs.items():
            if k_in in ("input_ids", "attention_mask"):
                continue
            forward_kwargs[k_in] = v_in

        base_model.eval()
        outputs = base_model(**forward_kwargs)
        loss_main = outputs.loss
        student_logits = outputs.logits

        # ---- 11) KL distillation (RepRAG-style) ----
        if not args.stage2_wo_kl:
            alpha_kl = float(getattr(args, "alpha_kl", 1.0))
            T_kl = float(getattr(args, "kl_temperature", 1.0))
            distill_topk = getattr(args, "distill_topk", None)

            loss_kl = get_kl_loss_reprag_style(
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

        # ---- 12) Training accuracy (exact match on supervised answer span) ----
        with torch.no_grad():
            sup_mask = (labels != -100)                 # [B, T]
            # set non-sup positions to large number, then take min
            idxs = torch.arange(labels.size(1), device=labels.device).unsqueeze(0).expand_as(labels)
            masked = idxs.masked_fill(~sup_mask, 10**9)
            first_pos = masked.min(dim=1).values        # [B]
            # if a sample has no supervised tokens, first_pos will be 1e9; clamp it
            first_pos = first_pos.clamp(max=labels.size(1) - 1).to(torch.long)


            # 2) logits at that position
            # Note: logits are for predicting token at position t (next-token shift in CausalLM),
            # but HF loss handles shift internally. For choice scoring, we should align with labels:
            # easiest: score at positions where labels != -100 using logits at same positions.
            # We'll score using the logits at first supervised position.
            logits_at = student_logits[torch.arange(labels.size(0), device=labels.device), first_pos]  # [B, vocab]

            # 3) build candidate token id for each letter " A", " B", ...
            # IMPORTANT: must match training exactly (leading space)
            cand_texts = [" " + LETTERS[i] for i in range(max(len(t["choices"]) for t in batch_tasks))]
            cand_ids = processor.tokenizer(cand_texts, add_special_tokens=False, padding=False)["input_ids"]

            # cand_ids[i] may be >1 token; handle simplest case (1 token) robustly:
            # If multi-token, you'd need sum logprobs across tokens (see Option B below).
            # For now, we take the last token id (often works if tokenization splits).
            cand_last_ids = torch.tensor([ids[-1] for ids in cand_ids], device=labels.device, dtype=torch.long)  # [C]

            logp = F.log_softmax(logits_at, dim=-1)  # [B, vocab]
            scores = logp[:, cand_last_ids]          # [B, C]

            # clamp C per sample by number of choices
            pred = scores.argmax(dim=1)             # [B]
            gold = gold_indices                      # [B]
            acc = float((pred == gold).float().mean().item())

        if writer is not None:
            writer.add_scalar("stage2/train_loss_ce", float(loss_main.item()), global_step)
            writer.add_scalar("stage2/train_loss_total", float(loss.item()), global_step)
            writer.add_scalar("stage2/train_acc", acc, global_step)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()


        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()

        global_step += 1

        loss_val = float(loss.item())
        loss_ce_val = float(loss_main.item())

        if pbar is not None:
            if not args.stage2_wo_kl:
                loss_kl_val = float(loss_kl.item())
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    ce=f"{loss_ce_val:.4f}",
                    kl=f"{loss_kl_val:.4f}",
                    acc=f"{acc:.4f}",
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    ce=f"{loss_ce_val:.4f}",
                    acc=f"{acc:.4f}",
                )
            pbar.update(1)

        if (step + 1) % 20 == 0 and pbar is None:
            print(f"[Stage2][Epoch {epoch+1}/train] step {step+1} | batch loss={loss_val:.4f} | acc={acc:.4f}")

        return loss_val, acc, global_step

    finally:
        handle.remove()


def stage2_eval_step(args, epoch: int, step: int, global_step: int, batch_tasks, device, processor, qwen_model, base_model, mlp, 
writer, model_dtype, train_emb_normed, train_qids, qid_to_row, index, train_task_map, reprag_token_id: int, eval_prefix: str = "stage2/eval", pbar=None):
    
    if not batch_tasks:
        return None, None

    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    try:
        with torch.no_grad():
            # 1) messages + gold
            base_messages, gold_indices = build_messages_and_golds(batch_tasks)
            gold_indices = gold_indices.to(device)

            # 2) K
            K = int(getattr(args, "stage2_finetune_topk", getattr(args, "top_k", 1)))

            # 3) retrieval
            batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch(
                batch_tasks=batch_tasks,
                train_emb_normed=train_emb_normed,
                train_qids=train_qids,
                qid_to_row=qid_to_row,
                index=index,
                top_k=K,
            )

            # 4) retrieval embs [B,K,H]
            ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved(
                batch_neighbors_qids=batch_neighbors_qids,
                batch_neighbors_scores=batch_neighbors_scores,
                train_task_map=train_task_map,
                model=qwen_model,
                processor=processor,
                device=device,
                output_mode=args.embedding_output_mode,
                k_concat=K,
            )
            H = base_model.config.hidden_size
            B, K2, _ = ret_emb_batch.shape
            assert K2 == K and ret_emb_batch.size(-1) == H

            # 5) project injections (compute in mlp dtype, inject in model dtype)
            mlp_dtype = next(mlp.parameters()).dtype
            inj_raw = mlp(ret_emb_batch.reshape(B*K, H).to(mlp_dtype)).reshape(B, K, H)
            inj_flat = inj_raw.to(dtype=model_dtype).reshape(B*K, H)

            # 6) build prompt inputs (for generation)
            prompt_texts = build_student_prompt_texts(processor, base_messages, k_prefix=K)
            image_inputs, video_inputs = process_vision_info(base_messages)

            prompt_inputs = processor(text=prompt_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

            input_ids = prompt_inputs["input_ids"]
            attn_mask = prompt_inputs["attention_mask"]

            # sanity: exactly K <RepRAG>
            xmask = (input_ids == reprag_token_id)
            counts = xmask.sum(dim=1).tolist()
            if not all(c == K for c in counts):
                raise RuntimeError(f"[Eval] expected K={K} <RepRAG> tokens per sample, got {counts}")

            # 7) hook (prefill-only)
            emb_layer = base_model.get_input_embeddings()
            fired = {"done": False}

            def _inject_hook(module, hook_inputs, output):
                if fired["done"]:
                    return output
                ids = hook_inputs[0]
                mask = (ids == reprag_token_id)
                if not mask.any():
                    return output
                out = output.clone()
                out[mask] = inj_flat.to(out.dtype)
                fired["done"] = True
                return out

            handle = emb_layer.register_forward_hook(_inject_hook)

            try:
                # 8) generate
                max_new = int(getattr(args, "eval_max_new_tokens", 8))
                min_new = int(getattr(args, "eval_min_new_tokens", 1))
                num_beams = int(getattr(args, "eval_num_beams", 1))
                do_sample = bool(getattr(args, "eval_do_sample", False))

                prompt_len = input_ids.shape[1]
                min_length = prompt_len + min_new

                gen_args = dict(
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    min_length=min_length,
                )
                if do_sample:
                    temperature = getattr(args, "eval_temperature", None)
                    top_p = getattr(args, "eval_top_p", None)
                    if temperature is not None:
                        gen_args["temperature"] = float(temperature)
                    if top_p is not None:
                        gen_args["top_p"] = float(top_p)

                generate_kwargs = {"input_ids": input_ids, "attention_mask": attn_mask}
                for k_in, v_in in prompt_inputs.items():
                    if k_in in ("input_ids", "attention_mask"):
                        continue
                    generate_kwargs[k_in] = v_in

                _dynamo_was_disabled = torch._dynamo.config.disable
                torch._dynamo.config.disable = True
                try:
                    out = base_model.generate(**generate_kwargs, **gen_args)
                finally:
                    torch._dynamo.config.disable = _dynamo_was_disabled

                trimmed = [seq[prompt_len:] if seq.shape[0] > prompt_len else seq for seq in out]
                decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                preds = [parse_option_letter(txt, 26) for txt in decoded]

                choice_counts = [len(t["choices"]) for t in batch_tasks]  # clamp by #choices
                preds = [p if (0 <= p < nC) else -1 for p, nC in zip(preds, choice_counts)]
                gold = [int(t["gold"]) for t in batch_tasks]
                correct = sum(int(p == g) for p, g in zip(preds, gold) if p >= 0)
                total = len(batch_tasks)  # always B

                acc = correct / total if total > 0 else 0.0

                # 9) optional CE (tokenization-consistent with stage2 prompt format)
                ce_loss_val = None
                if bool(getattr(args, "eval_log_ce", True)):
                    # Build the exact same student prompts, then append the gold answer letter
                    base_prompts = build_student_prompt_texts(processor, base_messages, k_prefix=K)
                    gold_texts = [p + (" " + LETTERS[g]) for p, g in zip(base_prompts, gold)]

                    ce_inputs = processor(
                        text=gold_texts,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

                    ce_ids = ce_inputs["input_ids"]
                    ce_mask = ce_inputs["attention_mask"]

                    # labels: supervise only the final non-pad token (the answer token)
                    ce_labels = torch.full_like(ce_ids, -100)
                    lengths = ce_mask.sum(dim=1).long()
                    for b in range(B):
                        last = int(lengths[b].item()) - 1
                        ce_labels[b, last] = ce_ids[b, last]

                    # reset hook so injection runs for this forward too
                    fired["done"] = False

                    fw = {"input_ids": ce_ids, "attention_mask": ce_mask, "labels": ce_labels, "return_dict": True}
                    for k_in, v_in in ce_inputs.items():
                        if k_in in ("input_ids", "attention_mask"):
                            continue
                        fw[k_in] = v_in

                    ce_out = base_model(**fw)
                    ce_loss_val = float(ce_out.loss.item())

                if pbar is not None:
                    if ce_loss_val is not None:
                        pbar.set_postfix(
                            acc=f"{acc:.4f}",
                            ce=f"{ce_loss_val:.4f}",
                        )
                    else:
                        pbar.set_postfix(
                            acc=f"{acc:.4f}",
                        )
                    pbar.update(1)

                if (step + 1) % 50 == 0 and pbar is None:
                    msg = f"[{eval_prefix}] epoch={epoch+1} step={step+1} batch acc={acc:.4f}"
                    if ce_loss_val is not None:
                        msg += f" batch ce_loss={ce_loss_val:.4f}"
                    print(msg)

                return ce_loss_val, acc

            finally:
                handle.remove()

    finally:
        base_model.eval()
        mlp.train(mlp_was_training)
