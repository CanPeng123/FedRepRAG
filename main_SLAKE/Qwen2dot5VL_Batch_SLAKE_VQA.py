import torch
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import os
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json
import pandas as pd
import datetime

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # go up to repo root
sys.path.append(str(ROOT))

from VQA.utilise.model_loading import load_model_with_quantization, load_model_with_deepspeed, load_model_standard

from VQA.utilise.SLAKE.Slake_arg import parse_args
from VQA.utilise.SLAKE.Slake_data_process import load_slake_items, build_tasks_from_slake
from VQA.utilise.SLAKE.Slake_post_process import normalize_answer_with_numbers, answers_match, normalize_answer
from VQA.utilise.SLAKE.Slake_result_record import write_final_results_txt, make_stats_bucket, norm_cat, update_stats


# ----------------------------------------------------------------
# --------------------- Prompt + generation ----------------------
# ----------------------------------------------------------------
def build_slake_vqa_message(question: str, image: Optional[PILImage], answer_type: Optional[str] = None, base_type: Optional[str] = None, 
triple: Optional[Any] = None):    
    """
    SLAKE: 1 image + 1 question -> short text answer.
    Optionally include (relation, tail) from triple for kvqa questions (won't leak the answer).
    """
    lines = [
        "You are a careful medical VQA assistant.",
        "Answer the question based on the radiology image.",
        "Reply with a short, direct answer (a single word or short phrase).",
        "If it is a yes/no question, answer exactly: Yes or No.",
        f"Question: {question}",
    ]

    if answer_type:
        lines.append(f"Answer type: {answer_type}")  # OPEN / CLOSED

    # Optional: hint the KG query structure if it's kvqa; triple often looks like [vhead, relation, tail]
    if base_type and base_type.lower() == "kvqa" and isinstance(triple, (list, tuple)) and len(triple) >= 3:
        rel = str(triple[1])
        tail = str(triple[2])
        lines.append(f"Knowledge query (may help): relation={rel}, tail={tail}")

    lines.append("Rules:\n- Do not output explanations.\n- Output ONLY the answer text.")

    content = [{"type": "text", "text": "\n".join(lines)}]
    if image is not None:
        content.append({"type": "image", "image": image})

    return {"role": "user", "content": content}


# ----------------------------------------------------------------
# ---- Batched VQA over (image, question, choices[, context]) ----
# ----------------------------------------------------------------
def _load_pil_or_none(path: Optional[str]) -> Optional[PILImage]:
    if not path:
        return None
    try:
        return PILImageModule.open(path).convert("RGB")
    except Exception:
        return None
    

def batch_vqa_answer_text(messages: List[Dict], model, processor, device, max_new_tokens: int = 16, min_new_tokens: int = 1, 
num_beams: int = 1, do_sample: bool = False, temperature: Optional[float] = None, top_p: Optional[float] = None) -> List[str]:
    """
    Run batched VQA inference and return decoded text answers (first line only).

    NOTE:
      - This version trims generation by the padded prompt_len (inputs.input_ids.shape[1]).
      - For more robust trimming under left-padding, use attention_mask.sum() per sample.
      - If using DeepSpeed InferenceEngine, avoid passing eos_token_id twice elsewhere.
    """
    texts = [
        processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    prompt_len = inputs.input_ids.shape[1]
    min_length = prompt_len + min_new_tokens

    gen_args = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        eos_token_id=processor.tokenizer.eos_token_id,
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


def process_batch_vqa_from_slake_json(args, model, processor, device, slake_json: str, images_root: Optional[str] = None, 
out_dir: str = "VQA_preds", batch_size: int = 8, split: Optional[str] = None, lang: str = "en"):
    os.makedirs(out_dir, exist_ok=True)

    items = load_slake_items(slake_json)
    tasks = build_tasks_from_slake(items, images_root=images_root, split_name=split, lang_filter=lang)
    tasks.sort(key=lambda x: x["qid"])
    print("task keys:", tasks[0].keys())
    print("example:", tasks[0])

    total = len(tasks)
    print(f"*** Loaded {total} SLAKE QA items (lang={lang})" + (f" for split={split}" if split else ""))

    rows = []
    correct_all = 0
    seen_all = 0
    correct_open = 0
    seen_open = 0
    correct_closed = 0
    seen_closed = 0

    # per-category breakdowns for ALL / OPEN / CLOSED
    stats = {
        "modality":     {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "location":     {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "content_type": {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "base_type":    {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
    }

    for i in range(0, total, batch_size):
        batch_tasks = tasks[i:i+batch_size]

        messages = []
        for t in batch_tasks:
            img = _load_pil_or_none(t["image_path"])
            msg = build_slake_vqa_message(question=t["question"], image=img, answer_type=t.get("answer_type"), 
            base_type=t.get("base_type"), triple=t.get("triple"))
            messages.append(msg)

        preds = batch_vqa_answer_text(
            messages, model, processor, device,
            max_new_tokens=16,
            min_new_tokens=1,
            num_beams=1,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
        )

        for t, pred in zip(batch_tasks, preds):
            gold = t["gold_answer"]
            pred_n = normalize_answer_with_numbers(pred)
            gold_n = normalize_answer_with_numbers(gold)
            is_correct = int(answers_match(pred, gold))

            at = (t.get("answer_type") or "").upper()  # OPEN / CLOSED / others

            m  = norm_cat(t.get("modality"))
            loc = norm_cat(t.get("location"))
            ct  = norm_cat(t.get("content_type"))
            bt  = norm_cat(t.get("base_type"))

            # always update ALL
            update_stats(stats["modality"]["all"], m, is_correct)
            update_stats(stats["location"]["all"], loc, is_correct)
            update_stats(stats["content_type"]["all"], ct, is_correct)
            update_stats(stats["base_type"]["all"], bt, is_correct)

            # update OPEN / CLOSED buckets
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

            print(f"[qid={t['qid']}] type={t.get('answer_type','')} | Q: {t['question']}")
            print(f"  pred : '{pred}'   (norm='{pred_n}')")
            print(f"  gold : '{gold}'   (norm='{gold_n}')")
            print(f"  ok?  : {is_correct}")

            correct_all += is_correct
            seen_all += 1

            if at == "OPEN":
                correct_open += is_correct
                seen_open += 1
            elif at == "CLOSED":
                correct_closed += is_correct
                seen_closed += 1

            rows.append({
                "qid": t["qid"],
                "split": t.get("split", ""),
                "question": t["question"],
                "image_path": t["image_path"],
                "answer_type": t.get("answer_type", ""),
                "base_type": t.get("base_type", ""),
                "content_type": t.get("content_type", ""),
                "modality": t.get("modality", ""),
                "location": t.get("location", ""),
                "triple": json.dumps(t.get("triple", None), ensure_ascii=False),

                "pred_answer": pred,
                "pred_norm": pred_n,
                "gold_answer": gold,
                "gold_norm": gold_n,
                "correct": is_correct,
            })

        done = min(i + batch_size, total)
        acc_all = 100.0 * correct_all / max(seen_all, 1)
        acc_open = 100.0 * correct_open / max(seen_open, 1)
        acc_closed = 100.0 * correct_closed / max(seen_closed, 1)
        print(
            f"Processed {done}/{total} | "
            f"correct={correct_all}/{seen_all} | "
            f"acc={acc_all:.2f}% (open={correct_open}/{seen_open}={acc_open:.2f}%, "
            f"closed={correct_closed}/{seen_closed}={acc_closed:.2f}%)"
        )
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f"slake_preds_{split or 'all'}_{lang}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    write_final_results_txt(out_dir=out_dir, args=args, split=split, lang=lang, total_questions=total, correct_all=correct_all, 
    seen_all=seen_all, correct_open=correct_open, seen_open=seen_open, correct_closed=correct_closed, seen_closed=seen_closed, 
    out_csv_path=out_csv, stats=stats)


# ----------------------------------------------------------------
# ----------------------------- Main -----------------------------
# ----------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()

    # Append creation time in YYYYmmdd_HHMMSS format
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Print and Save all parsed arguments (aligned) ----
    arg_dict = vars(args)
    lines = []
    print("\n===== Parsed arguments =====")
    width = max(len(k) for k in arg_dict) if arg_dict else 0
    for k in sorted(arg_dict):
        line = f"{k:<{width}} : {arg_dict[k]}"
        print(line)
        lines.append(line)
    print("================================\n")

    arg_file = os.path.join(output_dir, "argument.txt")
    with open(arg_file, "w") as f:
        f.write("===== Parsed arguments =====\n")
        f.write("\n".join(lines))
        f.write("\n================================\n")
    print(f"Arguments saved to {arg_file}")

    # ---------------------------------------------
    model_map = {
        'qwen2_5-vl-instruct:2b': "Qwen/Qwen2.5-VL-2B-Instruct",
        'qwen2_5-vl-instruct:7b': "Qwen/Qwen2.5-VL-7B-Instruct",
        'qwen2_5-vl-instruct:72b': "Qwen/Qwen2.5-VL-72B-Instruct"
    }
    model_name = model_map[args.model]

    if args.use_deepspeed:
        processor, model, _ = load_model_with_deepspeed(model_name)
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    elif args.use_quantization:
        processor, model = load_model_with_quantization(model_name, args.use_quantization)
        device = "cuda"
    else:
        processor, model = load_model_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    # Ensure left padding for Qwen2.5 tokenizer (safer for batched generate)
    processor.tokenizer.padding_side = "left"

    if not args.problems_json:
        raise SystemExit("--problems_json is required for VQA task.")
    
    process_batch_vqa_from_slake_json(args, model, processor, device, slake_json=args.problems_json, 
    images_root=args.images_root if args.images_root else None, out_dir=output_dir, batch_size=args.batch_size,
    split=args.split, lang=getattr(args, "lang", "en"))