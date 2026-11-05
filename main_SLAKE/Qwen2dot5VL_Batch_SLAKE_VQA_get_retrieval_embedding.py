import os
import sys
import json
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import pandas as pd

import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
from qwen_vl_utils import process_vision_info

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from VQA.utilise.model_loading import load_model_with_quantization, load_model_with_deepspeed, load_model_standard, load_gemma3_standard
from VQA.utilise.data_process import load_image, load_pil_or_none
from VQA.utilise.SLAKE.Slake_retrieval_arg import parse_args
from VQA.utilise.SLAKE.Slake_data_process import load_slake_items, build_tasks_from_slake
from VQA.utilise.SLAKE.Slake_post_process import normalize_answer_with_numbers, answers_match
from VQA.utilise.SLAKE.Slake_result_record import write_final_results_txt, make_stats_bucket, norm_cat, update_stats
from VQA.utilise.SLAKE.Slake_prompt_message import build_slake_vqa_message



# ----------------------------------------------------------------
# --------------------- Gemma helper ----------------------------
# ----------------------------------------------------------------
def _to_gemma_messages(messages: List[Dict]) -> tuple:
    """Strip PIL images from content dicts (Gemma format) and return (gemma_msgs, flat_pil_list)."""
    gemma_msgs, all_pils = [], []
    for msg in messages:
        new_content = []
        for item in msg.get("content", []):
            if item.get("type") == "image":
                pil = item.get("image")
                if isinstance(pil, PILImage):
                    all_pils.append(pil)
                new_content.append({"type": "image"})
            else:
                new_content.append(item)
        gemma_msgs.append({"role": msg["role"], "content": new_content})
    return gemma_msgs, all_pils

# ----------------------------------------------------------------
# --------------------- Decode cleanup ---------------------------
# ----------------------------------------------------------------
def _clean_first_answer_line(text: str) -> str:
    """
    Prevent 'assistant' / 'Assistant:' being returned as the first line.
    Keep the first non-empty line that is not a role marker.
    """
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    # split lines and drop empty
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    # drop leading "assistant" markers
    while lines and (lines[0].lower() == "assistant" or lines[0].lower().startswith("assistant:")):
        lines = lines[1:]

    return lines[0] if lines else ""

# ----------------------------------------------------------------
# --------------------- Batched generate -------------------------
# ----------------------------------------------------------------
@torch.no_grad()
def batch_vqa_answer_text(messages: List[Dict], model, processor, device, max_new_tokens: int = 16, min_new_tokens: int = 1,
num_beams: int = 1, do_sample: bool = False, temperature: Optional[float] = None, top_p: Optional[float] = None,
is_gemma: bool = False) -> List[str]:
    """
    Follow your original SLAKE inference code path, but:
      - trim by per-sample attention_mask.sum() (correct for left padding)
      - robustly clean decoded outputs to avoid 'assistant'
      - avoid DS eos_token_id duplication by not overriding eos/pad for deepspeed engine
    """
    if is_gemma:
        gemma_msgs, pil_list = _to_gemma_messages(messages)
        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in gemma_msgs]
        inputs = processor(text=texts, images=pil_list if pil_list else None, padding=True, return_tensors="pt").to(device)
    else:
        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    prompt_len = inputs.input_ids.shape[1]
    min_length = prompt_len + min_new_tokens

    gen_args = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        min_length=min_length,
    )

    if do_sample:
        if temperature is not None:
            gen_args["temperature"] = float(temperature)
        if top_p is not None:
            gen_args["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_args)

    trimmed = [seq[prompt_len:] for seq in out]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = [d.strip().splitlines()[0].strip() for d in decoded]

    print("---------------------")
    print("********* decoded answers:", decoded)
    print("---------------------")
    
    return decoded

# ----------------------------------------------------------------
# --------------------- Embedding extraction ---------------------
# ----------------------------------------------------------------
@torch.no_grad()
def compute_batch_embeddings(messages: List[Dict], model, processor, device, is_gemma: bool = False) -> torch.Tensor:
    """
    Returns: [B, D] on CPU, L2-normalized.
    """
    if is_gemma:
        gemma_msgs, pil_list = _to_gemma_messages(messages)
        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in gemma_msgs]
        inputs = processor(text=texts, images=pil_list if pil_list else None, padding=True, return_tensors="pt").to(device)
    else:
        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = outputs.hidden_states[-1]  # [B, T, D]

    attn = inputs.attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
    summed = (hidden * attn).sum(dim=1)                          # [B, D]
    lengths = attn.sum(dim=1).clamp(min=1e-6)                    # [B, 1]
    emb = summed / lengths                                       # [B, D]
    emb = F.normalize(emb, p=2, dim=1)                           # L2

    return emb.detach().cpu()


@torch.no_grad()
def debug_generate(messages: List[Dict], model, processor, device,
                   max_new_tokens=16, do_sample=False, temperature=None, is_gemma: bool = False) -> List[str]:
    if is_gemma:
        gemma_msgs, pil_list = _to_gemma_messages(messages)
        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in gemma_msgs]
        inputs = processor(text=texts, images=pil_list if pil_list else None, padding=True, return_tensors="pt").to(device)
    else:
        texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

    # generation args
    gen_args = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=1,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    if do_sample and temperature is not None:
        gen_args["temperature"] = float(temperature)

    out = model.generate(**inputs, **gen_args)

    # trim by true (unpadded) prompt length per sample
    input_lens = inputs.attention_mask.sum(dim=1).tolist()
    trimmed = [seq[int(L):] for seq, L in zip(out, input_lens)]

    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = [d.strip().splitlines()[0].strip() for d in decoded]
    return decoded


# ----------------------------------------------------------------
# ----------------------------- Main -----------------------------
# ----------------------------------------------------------------
def main():
    args = parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

     # save args
    arg_file = os.path.join(out_dir, "argument.txt")
    with open(arg_file, "w") as f:
        f.write("===== Parsed arguments =====\n")
        for k in sorted(vars(args)):
            f.write(f"{k}: {getattr(args, k)}\n")
        f.write("================================\n")
    print(f"[Args] saved to {arg_file}")

    # model loading
    model_map = {
        "qwen2_5-vl-instruct:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2_5-vl-instruct:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2_5-vl-instruct:72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        "gemma3:4b": "google/gemma-3-4b-it",
        "gemma3:12b": "google/gemma-3-12b-it",
    }
    model_name = model_map[args.model]
    is_gemma = args.model.startswith("gemma3")

    if is_gemma:
        processor, model = load_gemma3_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.use_deepspeed:
        ret = load_model_with_deepspeed(model_name)
        processor, model = ret[0], ret[1]
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    elif args.use_quantization:
        processor, model = load_model_with_quantization(model_name, args.use_quantization)
        device = "cuda"
    else:
        processor, model = load_model_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    processor.tokenizer.padding_side = "left"

    # load SLAKE items -> tasks
    items = load_slake_items(args.problems_json)
    tasks = build_tasks_from_slake(items, images_root=(args.images_root if args.images_root else None), split_name=args.split, 
    lang_filter=args.lang)
    tasks.sort(key=lambda x: x["qid"])
    print(f"*** Loaded {len(tasks)} SLAKE items for split={args.split}, lang={args.lang}")

    # shard
    if args.num_parts > 1:
        tasks = [t for j, t in enumerate(tasks) if (j % args.num_parts) == args.idx]
        print(f"*** Shard idx={args.idx}/{args.num_parts}: {len(tasks)} items")

    if not args.store_embeddings:
        print("[WARN] --store_embeddings not set; nothing to do.")
        return

    # stats + outputs (reuse your result_record)
    rows = []
    correct_all = seen_all = 0
    correct_open = seen_open = 0
    correct_closed = seen_closed = 0

    stats = {
        "modality":     {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "location":     {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "content_type": {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "base_type":    {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
    }

    all_embs = []
    all_qids = []
    all_meta = []

    total = len(tasks)
    for i in range(0, total, args.batch_size):
        batch = tasks[i:i + args.batch_size]

        messages = []
        for t in batch:
            img = load_pil_or_none(t.get("image_path"))
            msg = build_slake_vqa_message(
                question=t["question"],
                image=img,
                answer_type=t.get("answer_type"),
                base_type=t.get("base_type"),
                triple=t.get("triple"),
                gold_answer=t.get("gold_answer"),
                include_gold=getattr(args, "embeddings_w_gold", False),
            )
            messages.append(msg)

        embs = compute_batch_embeddings(messages, model, processor, device, is_gemma=is_gemma)  # [B, D] CPU
        all_embs.append(embs)

        for t in batch:
            all_qids.append(t["qid"])
            all_meta.append({
                "qid": t["qid"],
                "split": t.get("split", ""),
                "question": t.get("question", ""),
                "answer_type": t.get("answer_type", ""),
                "base_type": t.get("base_type", ""),
                "content_type": t.get("content_type", ""),
                "modality": t.get("modality", ""),
                "location": t.get("location", ""),
                "image_path": t.get("image_path", ""),
                "gold_answer": t.get("gold_answer", ""),
            })

        # debug inference (print + record)
        if getattr(args, "debug_generate", False):
            preds = batch_vqa_answer_text(
                messages, model, processor, device,
                max_new_tokens=getattr(args, "max_new_tokens", 16),
                min_new_tokens=1,
                num_beams=1,
                do_sample=getattr(args, "do_sample", False),
                temperature=getattr(args, "temperature", None) if getattr(args, "do_sample", False) else None,
                top_p=getattr(args, "top_p", None) if getattr(args, "do_sample", False) else None,
                is_gemma=is_gemma,
            )

            for t, pred in zip(batch, preds):
                gold = t.get("gold_answer", "")
                pred_n = normalize_answer_with_numbers(pred)
                gold_n = normalize_answer_with_numbers(gold)
                is_correct = int(answers_match(pred, gold)) if gold is not None else 0

                at = (t.get("answer_type") or "").upper()
                m  = norm_cat(t.get("modality"))
                loc = norm_cat(t.get("location"))
                ct  = norm_cat(t.get("content_type"))
                bt  = norm_cat(t.get("base_type"))

                update_stats(stats["modality"]["all"], m, is_correct)
                update_stats(stats["location"]["all"], loc, is_correct)
                update_stats(stats["content_type"]["all"], ct, is_correct)
                update_stats(stats["base_type"]["all"], bt, is_correct)

                if at == "OPEN":
                    update_stats(stats["modality"]["open"], m, is_correct)
                    update_stats(stats["location"]["open"], loc, is_correct)
                    update_stats(stats["content_type"]["open"], ct, is_correct)
                    update_stats(stats["base_type"]["open"], bt, is_correct)
                elif at == "CLOSED":
                    update_stats(stats["modality"]["closed"], m, is_correct)
                    update_stats(stats["location"]["closed"], loc, is_correct)
                    update_stats(stats["content_type"]["closed"], ct, is_correct)
                    update_stats(stats["base_type"]["closed"], bt, is_correct)

                correct_all += is_correct
                seen_all += 1
                if at == "OPEN":
                    correct_open += is_correct
                    seen_open += 1
                elif at == "CLOSED":
                    correct_closed += is_correct
                    seen_closed += 1

                print(f"[qid={t['qid']}] type={t.get('answer_type','')} | Q: {t['question']}")
                print(f"  pred : '{pred}'   (norm='{pred_n}')")
                print(f"  gold : '{gold}'   (norm='{gold_n}')")
                print(f"  ok?  : {is_correct}")

                rows.append({
                    "qid": t["qid"],
                    "split": t.get("split", ""),
                    "question": t.get("question", ""),
                    "image_path": t.get("image_path", ""),
                    "answer_type": t.get("answer_type", ""),
                    "base_type": t.get("base_type", ""),
                    "content_type": t.get("content_type", ""),
                    "modality": t.get("modality", ""),
                    "location": t.get("location", ""),
                    "pred_answer": pred,
                    "pred_norm": pred_n,
                    "gold_answer": gold,
                    "gold_norm": gold_n,
                    "correct": is_correct,
                })

            if seen_all > 0:
                acc_all = 100.0 * correct_all / max(seen_all, 1)
                acc_open = 100.0 * correct_open / max(seen_open, 1)
                acc_closed = 100.0 * correct_closed / max(seen_closed, 1)
                print(f"[DEBUG] seen={seen_all} correct={correct_all}/{seen_all} acc={acc_all:.2f}% "
                      f"(open={correct_open}/{seen_open}={acc_open:.2f}%, "
                      f"closed={correct_closed}/{seen_closed}={acc_closed:.2f}%)")

        done = min(i + args.batch_size, total)
        print(f"[Emb] {done}/{total} done")
        torch.cuda.empty_cache()

    # save embeddings
    emb_tensor = torch.cat(all_embs, dim=0)
    split_name = args.split or "all"

    if args.num_parts > 1:
        out_pt = os.path.join(out_dir, f"slake_embeddings_{split_name}_part{args.idx}_of{args.num_parts}.pt")
    else:
        out_pt = os.path.join(out_dir, f"slake_embeddings_{split_name}.pt")

    torch.save({"embeddings": emb_tensor, "qids": all_qids, "meta": all_meta}, out_pt)
    print(f"[Saved] {out_pt} | emb shape={tuple(emb_tensor.shape)}")

    # optional: save debug predictions csv + final_results.txt
    if rows:
        out_csv = os.path.join(out_dir, f"slake_debug_preds_{split_name}_{args.lang}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")

        write_final_results_txt(
            out_dir=out_dir,
            args=args,
            split=args.split,
            lang=args.lang,
            total_questions=total,
            correct_all=correct_all,
            seen_all=seen_all,
            correct_open=correct_open,
            seen_open=seen_open,
            correct_closed=correct_closed,
            seen_closed=seen_closed,
            out_csv_path=out_csv,
            stats=stats,
        )


if __name__ == "__main__":
    main()