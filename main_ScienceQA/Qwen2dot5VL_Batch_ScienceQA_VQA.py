import torch
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import os
from qwen_vl_utils import process_vision_info
from typing import List, Dict
import json
import pandas as pd
import datetime

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # go up to repo root
sys.path.append(str(ROOT))

from VQA.utilise.model_loading import load_model_with_quantization, load_model_with_deepspeed, load_model_standard
from VQA.utilise.ScienceQA_arg import parse_args
from VQA.utilise.ScienceQA_data_process import load_scienceqa_problems, build_tasks_from_scienceqa, load_image
from VQA.utilise.ScienceQA_post_process import parse_option_letter, write_per_split_outputs


# -----------------------------
# Prompt + generation
# -----------------------------
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def build_multi_image_vqa_message(question: str, choices: list[str], question_img: PILImage | None, choice_imgs: list[PILImage | None], 
                                  context: str | None = None):

    lines = [
        "You are a helpful science VQA assistant.",
        "You will be given one question image and images for some answer choices.",
        "Look at all the images and the question text, then select exactly ONE letter for the correct answer.",
        f"Question: {question}"
    ]
    if context:
        lines.append(f"Context: {context}")
    lines.append("Options:")
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c if c else '(see corresponding image)'}")
    lines.append("\nRules:" \
    "\n- The first image is the question image (if provided)." \
    "\n- The next images correspond to choices in order A, B, C, ..." \
    "\n- Answer with ONLY the option letter (e.g., A).")

    content = [{"type": "text", "text": "\n".join(lines)}]

    if question_img is not None:
        content.append({"type": "text", "text": "Question Image:"})
        content.append({"type": "image", "image": question_img})

    for i, im in enumerate(choice_imgs):
        if im is None:
            continue
        content.append({"type": "text", "text": f"Choice {LETTERS[i]} Image:"})
        content.append({"type": "image", "image": im})

    return {"role": "user", "content": content}


# ---------- Image caption generation ----------
def batch_caption(images: List[PILImage], model, processor, device, caption_prompt: str = "Describe the image briefly and factually.",
                  max_new_tokens: int = 48) -> List[str]:
    """Deterministic, short captions."""
    messages = []
    for img in images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": caption_prompt},
                {"type": "image", "image": img},
            ],
        })
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
             for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             min_new_tokens=1, num_beams=1, do_sample=False,
                             pad_token_id=pad_id,
                             eos_token_id=getattr(getattr(model, "generation_config", object()), "eos_token_id",
                                                  processor.tokenizer.eos_token_id))
    input_lens = inputs.attention_mask.sum(dim=1).tolist()
    trimmed = [seq[int(L):] for seq, L in zip(out, input_lens)]
    return [s.strip() for s in processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)]


# ----------------------------------------------------------------
# ---- Batched VQA over (image, question, choices[, context]) ----
# ----------------------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def _discover_q_choice_by_convention(image_dir: str | None, num_choices: int) -> tuple[str | None, list[str | None]]:
    """
    Deterministically find:
      - question image: 'image' + ext
      - choice images:  'choice_0'..'choice_{n-1}' + ext
    Returns (q_path_or_None, [choice0_path_or_None, ...]).
    """
    if not image_dir or not os.path.isdir(image_dir):
        return None, [None] * num_choices

    # question image
    q_path = None
    for ext in IMG_EXTS:
        p = os.path.join(image_dir, f"image{ext}")
        if os.path.isfile(p):
            q_path = p
            break

    # choices
    choice_paths: list[str | None] = []
    for i in range(num_choices):
        found = None
        for ext in IMG_EXTS:
            p = os.path.join(image_dir, f"choice_{i}{ext}")
            if os.path.isfile(p):
                found = p
                break
        choice_paths.append(found)

    return q_path, choice_paths

def _load_pil_or_none(path: str | None) -> PILImage | None:
    if not path:
        return None
    try:
        return PILImageModule.open(path).convert("RGB")
    except Exception:
        return None
    

def _load_captions_map(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow list of dicts or flat dict
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    if isinstance(data, list):
        # expect [{"qid": "...", "caption": "..."}]
        m = {}
        for x in data:
            qid = str(x.get("qid"))
            cap = str(x.get("caption", ""))
            if qid:
                m[qid] = cap
        return m
    return {}


def batch_vqa_answer(messages: list[dict], model, processor, device, max_new_tokens: int = 8, min_new_tokens: int = 1, num_beams: int = 1, 
                     do_sample: bool = False, temperature: float | None = None, top_p: float | None = None) -> tuple[list[int], list[str]]:
    """
    messages: list of chat dicts (each may include multiple images via build_multi_image_vqa_message)
    returns: (pred_indices_by_letter, raw_outputs)
    """
    # 1) Chat-template strings and vision inputs
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    
    # padding=True argument tells the tokenizer inside the processor to pad every sequence in the batch to the same length
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    # 2) Deterministic generation (don’t pass temperature if do_sample=False)
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
        if temperature is not None: gen_args["temperature"] = float(temperature)
        if top_p is not None:       gen_args["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_args)

    # 3) Trim each sequence using its own prompt length
    trimmed = [seq[prompt_len:] for seq in out]

    # 4) Decode + cleanup of any leaked template text
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # 5) parse A/B/C/... (we clamp later to #choices in the runner)
    preds = [parse_option_letter(txt, 26) for txt in decoded]

    print("---------------------")
    print("********* preds: {0}".format(preds))
    print("********* decoded: {0}".format(decoded))
    print("---------------------")

    return preds, decoded


def process_batch_vqa_from_json(args, model, processor, device, problems_json: str, images_root: str | None = None,
                                out_dir: str = "VQA_preds", batch_size: int = 8, split: str | None = None):
    os.makedirs(out_dir, exist_ok=True)
    probs = load_scienceqa_problems(problems_json)
    tasks = build_tasks_from_scienceqa(probs, images_root=images_root, split_filter=split)
    tasks.sort(key=lambda x: (x.get("split",""), x["qid"]))
    total = len(tasks)
    print(f"*** Loaded {total} QA items" + (f" for split={split}" if split else ""))

    # Standard eval config
    mode = "standard"  # single mode
    rows = []
    correct_so_far = 0
    processed_so_far = 0  

    for i in range(0, total, batch_size):
        batch_tasks = tasks[i:i+batch_size]

        # Standard ScienceQA: use dataset hint (if any) as context;
        # include images if present (question and optional choice images).
        contexts = [(t.get("context") or "") for t in batch_tasks]

        # build multi-image messages by folder convention
        messages, choice_counts = [], []
        for t, ctx in zip(batch_tasks, contexts):
            image_dir = t.get("image_dir") or (os.path.dirname(t["image_path"]) if t.get("image_path") else None)
            q_path, choice_paths = _discover_q_choice_by_convention(image_dir, len(t["choices"]))
            q_img = _load_pil_or_none(q_path)
            choice_imgs = [_load_pil_or_none(p) for p in choice_paths]

            msg = build_multi_image_vqa_message(question=t["question"], choices=t["choices"], question_img=q_img, 
            choice_imgs=choice_imgs, context=ctx)
            messages.append(msg)
            choice_counts.append(len(t["choices"]))

        preds, raw = batch_vqa_answer(messages, model, processor, device, max_new_tokens=8, min_new_tokens=1, num_beams=1,
            do_sample=args.do_sample, temperature=args.temperature if args.do_sample else None)

        for t, p, r, ctx, nC in zip(batch_tasks, preds, raw, contexts, choice_counts):
            # clamp parsed letter to valid range
            if not (0 <= p < nC):
                p = -1

            # compute correctness once to reuse below and for counters
            is_correct = int(p == t["gold"]) if (t["gold"] is not None and p >= 0) else 0
            correct_so_far += is_correct
            processed_so_far += 1            
            
            rows.append({
                "mode": mode,
                "qid": t["qid"],
                "split": t["split"],
                "subject": t.get("subject", ""),
                "grade": t.get("grade", None),
                "question": t["question"],
                "choices": " || ".join(t["choices"]),
                "context": ctx or "",
                "image_path": t.get("image_path", ""),
                "pred_idx": p,
                "pred_letter": (LETTERS[p] if p >= 0 else ""),
                "raw_output": r,
                "gold_idx": t["gold"],
                "gold_letter": (LETTERS[int(t["gold"])] if isinstance(t["gold"], int) and t["gold"] >= 0 else ""),
                "correct": int(p == t["gold"]) if t["gold"] is not None and p >= 0 else 0
            })

        processed = min(i + batch_size, total)
        acc_pct = (correct_so_far / processed_so_far * 100.0) if processed_so_far else 0.0
        print(f"[mode: {mode}] Processed {processed}/{total} | correct so far: {correct_so_far}/{processed_so_far} ({acc_pct:.2f}%)")
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    write_per_split_outputs(df, out_dir, split, mode)


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
    
    process_batch_vqa_from_json(args, model, processor, device, problems_json=args.problems_json, 
                                images_root=args.images_root if args.images_root else None, out_dir=output_dir, 
                                batch_size=args.batch_size, split=args.split)
    