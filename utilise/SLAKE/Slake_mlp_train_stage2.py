import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from qwen_vl_utils import process_vision_info


def _is_gemma_model(model):
    base = model.module if hasattr(model, "module") else model
    return "gemma" in type(base).__name__.lower()


def _processor_call(processor, model, text, messages, **kwargs):
    """Call processor with or without videos depending on model type."""
    image_inputs, video_inputs = process_vision_info(messages)
    if _is_gemma_model(model):
        return processor(text=text, images=image_inputs or None, **kwargs)
    return processor(text=text, images=image_inputs, videos=video_inputs, **kwargs)

from VQA.utilise.retrieval_helper import retrieve_neighbors_for_batch, retrieve_neighbors_for_batch_by_embedding

from VQA.utilise.SLAKE.Slake_data_process import load_pil_or_none
from VQA.utilise.SLAKE.Slake_retrieval_utilise import build_prefix_last_layer_for_batch_from_retrieved
from VQA.utilise.SLAKE.Slake_prompt_message import build_slake_vqa_message
from VQA.utilise.SLAKE.Slake_eval_utils import compute_slake_batch_acc, strip_answer_prefix
from VQA.utilise.SLAKE.Slake_mlp_utilise import build_teacher_evidence_texts, build_teacher_prompt_texts
from VQA.utilise.SLAKE.Slake_mlp_utilise import build_student_prompt_texts, get_kl_loss_reprag_style


REPRAG_TOKEN = "<RepRAG>"


# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def build_messages_and_golds_slake(batch_tasks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    base_messages = []
    golds = []
    for t in batch_tasks:
        img = load_pil_or_none(t.get("image_path"))
        msg = build_slake_vqa_message(
            question=t["question"],
            image=img,
            answer_type=t.get("answer_type"),
            base_type=t.get("base_type"),
            triple=t.get("triple"),
        )
        base_messages.append(msg)
        golds.append(t.get("gold_answer", t.get("answer", "")) or "")
    return base_messages, golds


def build_answer_token_ids_slake(processor, gold_texts: List[str]) -> List[List[int]]:
    answers = [" " + (g.strip() if g is not None else "") for g in gold_texts]
    tok = processor.tokenizer(answers, add_special_tokens=False, padding=False)
    return tok["input_ids"]


def make_labels_for_answer_span_packed(input_ids: torch.Tensor, prompt_lens: torch.Tensor, ans_lens: torch.Tensor, 
device: torch.device) -> torch.Tensor:
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



def stage2_finetune_step(args, epoch: int, step: int, global_step: int, batch_tasks, device, processor, qwen_model, base_model, mlp,
optimizer, writer, model_dtype, train_emb_normed, train_qids, qid_to_row, index, train_task_map, reprag_token_id: int, pbar=None):

    if not batch_tasks:
        return None, None, global_step

    assert processor.tokenizer.padding_side == "left", "Your packing slices assume left padding (valid tokens at the end)."

    # ---- 1) Build base messages + gold ----
    base_messages, gold_texts = build_messages_and_golds_slake(batch_tasks)

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
        )

    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    assert ret_emb_batch.dim() == 3 and ret_emb_batch.size(-1) == H
    B, K, _ = ret_emb_batch.shape
    assert K == args.stage2_finetune_topk, f"K mismatch: got {K}, expected {args.stage2_finetune_topk}"

    # ---- 4) Tokenize gold answers ----
    answer_ids_list = build_answer_token_ids_slake(processor, gold_texts)
    ans_lens = torch.tensor([len(x) for x in answer_ids_list], device=device, dtype=torch.long)

    # ---- 5) Teacher forward (optional) ----
    if not args.stage2_wo_kl:
        teacher_max_chars = int(getattr(args, "teacher_max_chars", 5000))

        teacher_evidence = build_teacher_evidence_texts(
            batch_neighbors_qids=batch_neighbors_qids,
            train_task_map=train_task_map,
            topk=K,
            max_chars=teacher_max_chars,
        )
        teacher_prompt_texts = build_teacher_prompt_texts(processor, base_messages, teacher_evidence)

        teacher_prompt_inputs = _processor_call(
            processor, qwen_model, teacher_prompt_texts, base_messages,
            padding=True, return_tensors="pt"
        ).to(device)

        teacher_prompt_input_ids = teacher_prompt_inputs["input_ids"]
        teacher_prompt_attn_mask = teacher_prompt_inputs["attention_mask"]
        pad_id = processor.tokenizer.pad_token_id
        teacher_prompt_lens = teacher_prompt_attn_mask.sum(dim=1).long()

        teacher_total_lens = teacher_prompt_lens + ans_lens
        teacher_T_total = int(teacher_total_lens.max().item())

        teacher_input_ids = torch.full((B, teacher_T_total), pad_id, device=device, dtype=torch.long)
        teacher_attn_mask = torch.zeros((B, teacher_T_total), device=device, dtype=teacher_prompt_attn_mask.dtype)

        teacher_prompt_token_type_ids = teacher_prompt_inputs.get("token_type_ids", None)
        teacher_token_type_ids = None
        if teacher_prompt_token_type_ids is not None:
            teacher_token_type_ids = torch.zeros((B, teacher_T_total), device=device, dtype=teacher_prompt_token_type_ids.dtype)

        T_prompt_t = teacher_prompt_input_ids.size(1)
        for b in range(B):
            p_len = int(teacher_prompt_lens[b].item())
            a_len = int(ans_lens[b].item())
            total = p_len + a_len
            start = teacher_T_total - total

            if p_len > 0:
                prompt_tokens = teacher_prompt_input_ids[b, T_prompt_t - p_len:T_prompt_t]
                teacher_input_ids[b, start:start + p_len] = prompt_tokens
                teacher_attn_mask[b, start:start + p_len] = 1
                if teacher_token_type_ids is not None:
                    teacher_token_type_ids[b, start:start + p_len] = teacher_prompt_token_type_ids[b, T_prompt_t - p_len:T_prompt_t]

            if a_len > 0:
                ans_tokens = torch.tensor(answer_ids_list[b], device=device, dtype=torch.long)
                teacher_input_ids[b, start + p_len:start + p_len + a_len] = ans_tokens
                teacher_attn_mask[b, start + p_len:start + p_len + a_len] = 1

        teacher_labels = make_labels_for_answer_span_packed(
            input_ids=teacher_input_ids,
            prompt_lens=teacher_prompt_lens,
            ans_lens=ans_lens,
            device=device,
        )

        base_model.eval()
        with torch.no_grad():
            teacher_forward_kwargs = {
                "input_ids": teacher_input_ids,
                "attention_mask": teacher_attn_mask,
                "return_dict": True,
            }
            if teacher_token_type_ids is not None:
                teacher_forward_kwargs["token_type_ids"] = teacher_token_type_ids
            for k_in, v_in in teacher_prompt_inputs.items():
                if k_in in ("input_ids", "attention_mask", "token_type_ids"):
                    continue
                teacher_forward_kwargs[k_in] = v_in

            teacher_outputs = base_model(**teacher_forward_kwargs)
            teacher_logits = teacher_outputs.logits

    # ---- 6) Project injection embeddings ----
    mlp_dtype = next(mlp.parameters()).dtype
    inj_raw = mlp(ret_emb_batch.reshape(B * K, H).to(mlp_dtype)).reshape(B, K, H)
    inj_flat = inj_raw.to(model_dtype).reshape(B * K, H)

    if writer is not None:
        writer.add_scalar("stage2/inj_norm", float(inj_raw.norm(dim=-1).mean().item()), global_step)

    # ---- 7) Student prompts with K <RepRAG> tokens ----
    prompt_texts = build_student_prompt_texts(processor, base_messages, k_prefix=K)
    prompt_inputs = _processor_call(
        processor, qwen_model, prompt_texts, base_messages,
        padding=True, return_tensors="pt"
    ).to(device)

    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attn_mask = prompt_inputs["attention_mask"]
    pad_id = processor.tokenizer.pad_token_id
    prompt_lens = prompt_attn_mask.sum(dim=1).long()

    total_lens = prompt_lens + ans_lens
    T_total = int(total_lens.max().item())

    input_ids = torch.full((B, T_total), pad_id, device=device, dtype=torch.long)
    attn_mask = torch.zeros((B, T_total), device=device, dtype=prompt_attn_mask.dtype)

    prompt_token_type_ids = prompt_inputs.get("token_type_ids", None)
    student_token_type_ids = None
    if prompt_token_type_ids is not None:
        student_token_type_ids = torch.zeros((B, T_total), device=device, dtype=prompt_token_type_ids.dtype)

    T_prompt = prompt_input_ids.size(1)
    for b in range(B):
        p_len = int(prompt_lens[b].item())
        a_len = int(ans_lens[b].item())
        total = p_len + a_len
        start = T_total - total

        if p_len > 0:
            prompt_tokens = prompt_input_ids[b, T_prompt - p_len:T_prompt]
            input_ids[b, start:start + p_len] = prompt_tokens
            attn_mask[b, start:start + p_len] = 1
            if student_token_type_ids is not None:
                student_token_type_ids[b, start:start + p_len] = prompt_token_type_ids[b, T_prompt - p_len:T_prompt]

        if a_len > 0:
            ans_tokens = torch.tensor(answer_ids_list[b], device=device, dtype=torch.long)
            input_ids[b, start + p_len:start + p_len + a_len] = ans_tokens
            attn_mask[b, start + p_len:start + p_len + a_len] = 1

    labels = make_labels_for_answer_span_packed(input_ids=input_ids, prompt_lens=prompt_lens, ans_lens=ans_lens, device=device)

    xmask = (input_ids == reprag_token_id)
    counts = xmask.sum(dim=1).tolist()
    assert all(c == K for c in counts), f"Each sample must have K <RepRAG> tokens, got {counts}"
    assert int(xmask.sum().item()) == B * K, "Total <RepRAG> slots mismatch."
    assert reprag_token_id != processor.tokenizer.unk_token_id

    if not args.stage2_wo_kl:
        t_sup = int((teacher_labels != -100).sum().item())
        s_sup = int((labels != -100).sum().item())
        if t_sup != s_sup:
            raise RuntimeError(
                f"Teacher/student supervised token mismatch: {t_sup} vs {s_sup}. "
                f"Likely teacher prompt too long / truncation."
            )

    # ---- 8) Register embedding hook ----
    emb_layer = base_model.get_input_embeddings()

    def _inject_hook(module, inputs, output):
        ids = inputs[0]
        mask = (ids == reprag_token_id)
        if not mask.any():
            return output
        out = output.clone()
        out[mask] = inj_flat.to(out.dtype)
        return out

    handle = emb_layer.register_forward_hook(_inject_hook)

    try:
        # ---- 9) Forward ----
        # For Gemma3: pixel_values in prompt_inputs are sized for T_prompt only, but
        # input_ids here has length T_total = T_prompt + ans_len. Passing both together
        # causes a causal mask size mismatch inside Gemma3, producing wrong gradients
        # (the MLP appears to learn initially but diverges as noisy gradients accumulate).
        # Fix: call the embedding layer directly so the hook fires and replaces <RepRAG>
        # tokens, then pass inputs_embeds (no pixel_values) to the model.
        base_model.eval()
        if _is_gemma_model(base_model):
            train_embeds = emb_layer(input_ids)  # hook fires here
            forward_kwargs = {
                "inputs_embeds": train_embeds,
                "attention_mask": attn_mask,
                "labels": labels,
                "return_dict": True,
            }
        else:
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels,
                "return_dict": True,
            }
            if student_token_type_ids is not None:
                forward_kwargs["token_type_ids"] = student_token_type_ids
            for k_in, v_in in prompt_inputs.items():
                if k_in in ("input_ids", "attention_mask", "token_type_ids"):
                    continue
                forward_kwargs[k_in] = v_in

        outputs = base_model(**forward_kwargs)
        loss_main = outputs.loss
        student_logits = outputs.logits

        # ---- 10) KL distillation ----
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

        # ---- 11) Generate periodic train acc ----
        acc = None
        acc_every = int(getattr(args, "acc_every", 20))
        if (global_step % acc_every) == 0:
            gen_prompt_len = prompt_inputs["input_ids"].shape[1]
            max_new = int(getattr(args, "eval_max_new_tokens", 8))
            min_new = int(getattr(args, "eval_min_new_tokens", 1))
            num_beams = int(getattr(args, "eval_num_beams", 1))
            do_sample = bool(getattr(args, "eval_do_sample", False))
            min_length = gen_prompt_len + min_new

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

            gen_kwargs = {
                "input_ids": prompt_inputs["input_ids"],
                "attention_mask": prompt_inputs["attention_mask"],
            }
            for k_in, v_in in prompt_inputs.items():
                if k_in in ("input_ids", "attention_mask"):
                    continue
                gen_kwargs[k_in] = v_in

            _dynamo_was_disabled = torch._dynamo.config.disable
            torch._dynamo.config.disable = True
            try:
                with torch.no_grad():
                    out = base_model.generate(**gen_kwargs, **gen_args)
            finally:
                torch._dynamo.config.disable = _dynamo_was_disabled

            trimmed = [seq[gen_prompt_len:] if seq.shape[0] > gen_prompt_len else seq for seq in out]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            pred_texts = [strip_answer_prefix(d) for d in decoded]
            acc, breakdown, state = compute_slake_batch_acc(batch_tasks, pred_texts, return_breakdown=True)

        if writer is not None:
            writer.add_scalar("stage2/train_loss_ce", float(loss_main.item()), global_step)
            writer.add_scalar("stage2/train_loss_total", float(loss.item()), global_step)
            if acc is not None:
                writer.add_scalar("stage2/train_acc", float(acc), global_step)

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
                if acc is not None:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", ce=f"{loss_ce_val:.4f}", kl=f"{loss_kl_val:.4f}", acc=f"{acc:.4f}")
                else:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", ce=f"{loss_ce_val:.4f}", kl=f"{loss_kl_val:.4f}")
            else:
                if acc is not None:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", ce=f"{loss_ce_val:.4f}", acc=f"{acc:.4f}")
                else:
                    pbar.set_postfix(loss=f"{loss_val:.4f}", ce=f"{loss_ce_val:.4f}")
            pbar.update(1)

        if (step + 1) % 20 == 0 and pbar is None:
            msg = f"[Stage2][SLAKE][Epoch {epoch+1}] step {step+1} | loss={loss_val:.4f} | ce={loss_ce_val:.4f}"
            if not args.stage2_wo_kl:
                msg += f" | kl={float(loss_kl.item()):.4f}"
            if acc is not None:
                msg += f" | acc={acc:.4f}"
            print(msg)

        return loss_val, (float(acc) if acc is not None else None), global_step

    finally:
        handle.remove()


def stage2_eval_step(args, epoch: int, step: int, global_step: int, batch_tasks, device, processor, qwen_model, base_model, mlp,
writer, model_dtype, train_emb_normed, train_qids, qid_to_row, index, train_task_map, reprag_token_id: int, 
eval_prefix: str = "stage2/eval", pbar=None):

    if not batch_tasks:
        return None, None

    mlp_was_training = mlp.training
    base_model.eval()
    mlp.eval()

    try:
        with torch.no_grad():
            # 1) messages + gold
            base_messages, gold_texts = build_messages_and_golds_slake(batch_tasks)

            # 2) K
            K = int(getattr(args, "stage2_finetune_topk", getattr(args, "top_k", 1)))

            # 3) retrieval
            batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch_by_embedding(
                batch_messages=base_messages,
                model=qwen_model,
                processor=processor,
                device=device,
                faiss_index=index,
                train_qids=train_qids,
                top_k=50,
            )
            assert batch_neighbors_scores is not None, "retrieve_neighbors_for_batch_by_embedding must return scores."

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
            cfg = base_model.config
            H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            B, K2, _ = ret_emb_batch.shape
            assert K2 == K and ret_emb_batch.size(-1) == H

            # 5) project injections
            mlp_dtype = next(mlp.parameters()).dtype
            inj_raw = mlp(ret_emb_batch.reshape(B * K, H).to(mlp_dtype)).reshape(B, K, H)
            inj_flat = inj_raw.to(dtype=model_dtype).reshape(B * K, H)

            # 6) build prompt inputs
            prompt_texts = build_student_prompt_texts(processor, base_messages, k_prefix=K)
            prompt_inputs = _processor_call(
                processor, qwen_model, prompt_texts, base_messages,
                padding=True, return_tensors="pt",
            ).to(device)

            input_ids = prompt_inputs["input_ids"]
            attn_mask = prompt_inputs["attention_mask"]

            xmask = (input_ids == reprag_token_id)
            counts = xmask.sum(dim=1).tolist()
            if not all(c == K for c in counts):
                raise RuntimeError(f"[Eval] expected K={K} <RepRAG> tokens per sample, got {counts}")

            # 7) hook
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
                pred_texts = [strip_answer_prefix(d) for d in decoded]

                acc, breakdown, state = compute_slake_batch_acc(batch_tasks, pred_texts, return_breakdown=True)

                ce_loss_val = None
                if bool(getattr(args, "eval_log_ce", True)):
                    answer_ids_list = build_answer_token_ids_slake(processor, gold_texts)
                    ans_lens = torch.tensor([len(x) for x in answer_ids_list], device=device, dtype=torch.long)

                    prompt_input_ids = prompt_inputs["input_ids"]
                    prompt_attn_mask = prompt_inputs["attention_mask"]
                    prompt_lens = prompt_attn_mask.sum(dim=1).long()

                    pad_id = processor.tokenizer.pad_token_id
                    total_lens = prompt_lens + ans_lens
                    T_total = int(total_lens.max().item())

                    ce_ids = torch.full((B, T_total), pad_id, device=device, dtype=torch.long)
                    ce_mask = torch.zeros((B, T_total), device=device, dtype=prompt_attn_mask.dtype)

                    T_prompt = prompt_input_ids.size(1)
                    for b in range(B):
                        p_len = int(prompt_lens[b].item())
                        a_len = int(ans_lens[b].item())
                        total = p_len + a_len
                        start = T_total - total

                        if p_len > 0:
                            prompt_tokens = prompt_input_ids[b, T_prompt - p_len:T_prompt]
                            ce_ids[b, start:start + p_len] = prompt_tokens
                            ce_mask[b, start:start + p_len] = 1

                        if a_len > 0:
                            ans_tokens = torch.tensor(answer_ids_list[b], device=device, dtype=torch.long)
                            ce_ids[b, start + p_len:start + p_len + a_len] = ans_tokens
                            ce_mask[b, start + p_len:start + p_len + a_len] = 1

                    ce_labels = make_labels_for_answer_span_packed(
                        input_ids=ce_ids,
                        prompt_lens=prompt_lens,
                        ans_lens=ans_lens,
                        device=device,
                    )

                    fired["done"] = False
                    # Use inputs_embeds instead of input_ids + vision keys to avoid
                    # Gemma3's causal mask size mismatch when answer tokens are appended
                    # (ce_ids has T_total = T_prompt + ans_len, but pixel_values/
                    # image_grid_thw are sized for T_prompt only).
                    ce_embeds = base_model.get_input_embeddings()(ce_ids)
                    fw = {
                        "inputs_embeds": ce_embeds,
                        "attention_mask": ce_mask,
                        "labels": ce_labels,
                        "return_dict": True,
                    }

                    ce_out = base_model(**fw)
                    ce_loss_val = float(ce_out.loss.item())

                if pbar is not None:
                    if ce_loss_val is not None:
                        pbar.set_postfix(acc=f"{acc:.4f}", ce=f"{ce_loss_val:.4f}")
                    else:
                        pbar.set_postfix(acc=f"{acc:.4f}")
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