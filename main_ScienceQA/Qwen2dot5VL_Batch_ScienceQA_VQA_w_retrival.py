import torch
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import os
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Union, Optional
import json
import pandas as pd
import datetime
import torch.nn.functional as F     
import numpy as np                  
import faiss                        

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # go up to repo root
sys.path.append(str(ROOT))

from VQA.arg_ScienceQA.ScienceQA_retrival_arg import parse_args

from VQA.utilise.model_loading import load_model_with_quantization, load_model_with_deepspeed, load_model_standard
from VQA.utilise.retrieval_architecture import PrefixProjector
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index, retrieve_neighbors_for_batch_by_embedding
from VQA.utilise.eval_utils import estimate_gflops, get_batch_token_count, record_efficiency_to_txt

from VQA.utilise.data_process import load_image, load_pil_or_none
from VQA.utilise.ScienceQA.ScienceQA_data_process import load_scienceqa_problems, build_tasks_from_scienceqa
from VQA.utilise.ScienceQA.ScienceQA_data_process import discover_q_choice_by_convention, load_captions
from VQA.utilise.ScienceQA.ScienceQA_post_process import parse_option_letter, write_per_split_outputs
from VQA.utilise.ScienceQA.ScienceQA_prompt_message import build_multi_image_vqa_message, build_retrieval_augmented_message
from VQA.utilise.ScienceQA.ScienceQA_retrival_utilise import build_prefix_layer0_for_batch_from_retrieved

from VQA.utilise.ScienceQA.ScienceQA_mlp_train_stage1 import build_prefix_last_layer_for_batch_from_retrieved

from VQA.utilise.mlp_utilise.mlp_train_stage2_utilise import build_student_prompt_texts


REPRAG_TOKEN = "<RepRAG>"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ----------------------------------------------------------------
# ---- Batched VQA over (image, question, choices[, context]) ----
# ----------------------------------------------------------------
def get_eval_json_path(args) -> str:
    eval_on = getattr(args, "eval_on", "test").lower()
    if eval_on == "train":
        if not args.train_json:
            raise ValueError("--train_json is required when --eval_on train")
        return args.train_json
    elif eval_on == "val":
        if not args.val_json:
            raise ValueError("--val_json is required when --eval_on val")
        return args.val_json
    elif eval_on == "test":
        if not args.test_json:
            raise ValueError("--test_json is required when --eval_on test")
        return args.test_json
    else:
        raise ValueError(f"Unsupported eval_on={eval_on}. Use train / val / test.")

# ----------------------------------------------------------------
# ---- Batched VQA over (image, question, choices[, context]) ----
# ----------------------------------------------------------------
def batch_vqa_answer(messages: list[dict], model, processor, device, max_new_tokens: int = 8, min_new_tokens: int = 1, num_beams: int = 1, 
do_sample: bool = False, temperature: float | None = None, top_p: float | None = None) -> tuple[list[int], list[str]]:
    """
    messages: list of chat dicts (each may include multiple images via build_multi_image_vqa_message)
    returns: (pred_indices_by_letter, raw_outputs)
    """
    # 1) Chat-template strings and vision inputs
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    # padding=True pads every sequence in the batch to same length
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    # 2) Deterministic generation (don’t pass temperature if do_sample=False)
    prompt_len = inputs.input_ids.shape[1]
    min_length = prompt_len + min_new_tokens

    gen_args = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams, 
    eos_token_id=processor.tokenizer.eos_token_id, min_length=min_length)
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

    return preds, decoded


def batch_vqa_answer_with_prefix(messages: list[dict], model, processor, device, prefix_embs: torch.Tensor, max_new_tokens: int = 8, min_new_tokens: int = 1, 
num_beams: int = 1, do_sample: bool = False, temperature: float | None = None, top_p: float | None = None) -> tuple[list[int], list[str]]:
    """
    Same as batch_vqa_answer, but injects a sequence of prefix embeddings
    (e.g., from retrieval) before the text tokens via inputs_embeds.
    """
    # 0) Get underlying HF model (unwrap Deepspeed InferenceEngine if needed)
    base_model = model.module if hasattr(model, "module") else model
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype

    # 1) Build standard inputs via processor
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    # padding=True pads every sequence in the batch to same length
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    # Move inputs to the model's device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    B, T = input_ids.shape

    # 2) Text embedding-layer output (NOTE: image/video placeholders are still placeholders here)
    with torch.no_grad():
        embed_layer = base_model.get_input_embeddings()  # nn.Embedding
        base_embs = embed_layer(input_ids)              # [B, T, H], text_only
        base_embs = base_embs.to(dtype=model_dtype)
    print("--- batch_vqa_answer_with_prefix | base_embs size: {0}".format(base_embs.shape))

    # 3) Concatenate prefix embeddings (if any)
    if prefix_embs is not None and prefix_embs.size(1) > 0:
        assert prefix_embs.shape[0] == B, "prefix_embs batch mismatch"
        prefix_embs = prefix_embs.to(device=model_device, dtype=model_dtype)
        print("--- batch_vqa_answer_with_prefix | prefix_embs size: {0}".format(prefix_embs.shape))
        L = prefix_embs.shape[1]  # prefix length
        
        new_embs = torch.cat([prefix_embs, base_embs], dim=1)  # [B, L+T, H]
        print("--- batch_vqa_answer_with_prefix | new_embs size: {0}".format(new_embs.shape))

        prefix_mask = torch.ones((B, L), device=model_device, dtype=attn_mask.dtype)
        new_attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)  # [B, L+T]

        # IMPORTANT: build matching-length input_ids for the prefix positions
        tok = processor.tokenizer
        dummy_id = tok.encode(".", add_special_tokens=False)[0]  # Try to use a normal text token that is almost certainly not a vision placeholder.
        special_ids = set(tok.all_special_ids)  # As a safety net, avoid any special token ids
        if dummy_id in special_ids:  # If "." somehow maps to a special id (unlikely), fall back to another common token
            dummy_id = tok.encode("the", add_special_tokens=False)[0]
        if dummy_id in special_ids:  # Final fallback: pick the first non-special id in vocab
            for cand in range(tok.vocab_size):
                if cand not in special_ids:
                    dummy_id = cand
                    break
        dummy_prefix_ids = torch.full((B, L), fill_value=dummy_id, device=model_device, dtype=input_ids.dtype)
        new_input_ids = torch.cat([dummy_prefix_ids, input_ids], dim=1)  # [B, L+T]

        # count image placeholder tokens before/after
        tok = processor.tokenizer
        # Try common Qwen-VL image tokens (depends on template); adjust if you know the exact string
        candidate_tokens = ["<image>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
        candidate_ids = {t: tok.convert_tokens_to_ids(t) for t in candidate_tokens}
        print("candidate_ids:", candidate_ids)

        # Count how many times any of those ids appear
        for t, tid in candidate_ids.items():
            if tid is None or tid < 0:
                continue
            print(f"count {t} in input_ids:", int((input_ids == tid).sum().item()))
            print(f"count {t} in new_input_ids:", int((new_input_ids == tid).sum().item()))

        # Also print the dummy id and whether it appears in original input_ids
        print("dummy_id:", int(dummy_id))
        print("dummy_id appears in input_ids:", bool((input_ids == dummy_id).any().item()))
    else:
        L = 0
        new_embs = base_embs
        new_attn_mask = attn_mask
        new_input_ids = input_ids
        
    prompt_len = new_embs.shape[1]  # L + T

    # 4) Generation args – keep it simple
    gen_args = dict(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, do_sample=do_sample, num_beams=num_beams, 
    eos_token_id=processor.tokenizer.eos_token_id, pad_token_id=processor.tokenizer.pad_token_id)
    if do_sample:
        if temperature is not None: gen_args["temperature"] = float(temperature)
        if top_p is not None:       gen_args["top_p"] = float(top_p)

    # 5) Build generate kwargs
    # CRITICAL: pass input_ids alongside inputs_embeds for Qwen2.5-VL multimodal masking/replacement.
    generate_kwargs = {
        "inputs_embeds": new_embs,  # model uses the text embeddings provided in new_embs as-is
        "input_ids": new_input_ids, # uses new_input_ids to find the visual placeholder slots and overwrite those slots with real vision embeddings
        "attention_mask": new_attn_mask,
    }
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        generate_kwargs[k] = v

    with torch.no_grad():
        out = model.generate(**generate_kwargs, **gen_args)

    # print(">>> out.shape:", out.shape)
    # print(">>> prompt_len:", prompt_len)
    full_decoded = processor.batch_decode(out, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # print(">>> FULL decoded[0]:", repr(full_decoded[0]))

    # If out is shorter than or equal to prompt_len, it clearly does NOT contain the prompt.
    # In that case, just use out as-is.
    trimmed = []
    for seq in out:
        if seq.shape[0] > prompt_len:
            trimmed.append(seq[prompt_len:])
        else:
            trimmed.append(seq)          # <- NO TRIM, just use generated tokens

    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds = [parse_option_letter(txt, 26) for txt in decoded]

    print("---------------------")
    print("********* preds (prefix mode):", preds)
    print("********* decoded (prefix mode):", decoded)
    print("---------------------")

    return preds, decoded


def batch_vqa_answer_with_reprag_hook(
    messages: list[dict],
    model,
    processor,
    device,
    ret_emb_batch: torch.Tensor,   # [B, K, H] from build_prefix_last_layer_for_batch_from_retrieved
    mlp_model,                     # stage2-trained MLP projector
    reprag_token_id: int,
    K: int,
    max_new_tokens: int = 8,
    min_new_tokens: int = 1,
    num_beams: int = 1,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    inj_mult: float = 0.1,         # IMPORTANT: match training (0.1)
) -> tuple[list[int], list[str]]:

    # unwrap
    base_model = model.module if hasattr(model, "module") else model
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype

    B, K2, H = ret_emb_batch.shape
    assert K2 == K, f"K mismatch: ret_emb_batch has K={K2}, expected K={K}"

    # 1) Build prompt texts with K <RepRAG> tokens prepended (same as stage2 training)
    # 1) Build prompt texts EXACTLY like stage-2 training
    texts = build_student_prompt_texts(processor, messages, k_prefix=K)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    # sanity: ensure exactly K RepRAG tokens per sample
    xmask = (input_ids == reprag_token_id)
    counts = xmask.sum(dim=1).tolist()
    print(f"[REPRAG] counts per sample: {counts} (expect {K})")
    if not all(c == K for c in counts):
        raise RuntimeError(f"Each sample must have exactly K={K} <RepRAG> tokens, got counts={counts}. "
                           f"Likely <RepRAG> is not a single token or template changed.")
    assert int(xmask.sum().item()) == B * K


    # 5) project injections (compute in mlp dtype, inject in model dtype)
    with torch.no_grad():
        mlp_dtype = next(mlp_model.parameters()).dtype
        inj_raw = mlp_model(ret_emb_batch.reshape(B*K, H).to(mlp_dtype)).reshape(B, K, H)
        inj_used = inj_raw.to(dtype=model_dtype)   # [B, K, H]
  
    # 3) Hook: overwrite embeddings at <RepRAG> positions
    fired = {"done": False}
    def _inject_hook(module, hook_inputs, output):
        if fired["done"]:
            return output

        ids = hook_inputs[0]              # [B,T]
        xmask = (ids == reprag_token_id)    # [B,T]
        if not xmask.any():
            return output

        # (optional) strict sanity: exactly K per sample
        counts = xmask.sum(dim=1).tolist()
        if not all(c == K for c in counts):
            print(f"[REPRAG][WARN] counts per sample = {counts} (expect {K}); skipping injection")
            return output

        out = output.clone()
        inj_flat = inj_used.reshape(-1, inj_used.size(-1))   # [B*K, H]
        out[xmask] = inj_flat.to(out.dtype)                  # exactly stage2
        fired["done"] = True
        return out

    handle = base_model.get_input_embeddings().register_forward_hook(_inject_hook)

    try:
        prompt_len = input_ids.shape[1]
        min_length = prompt_len + min_new_tokens

        gen_args = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            eos_token_id=processor.tokenizer.eos_token_id,
            min_length=min_length,
            pad_token_id=processor.tokenizer.pad_token_id,
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
        preds = [parse_option_letter(txt, 26) for txt in decoded]
        return preds, decoded

    finally:
        handle.remove()


# ----------------------------------------------------------------
# ---- VQA over (image, question, choices[, context]) ----
# ----------------------------------------------------------------
def process_batch_vqa_from_json(args, model, processor, device, eval_json: str, train_json: str, images_root: str | None = None, 
out_dir: str = "VQA_preds", batch_size: int = 8, split: str | None = None, reprag_token_id: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    
    eval_probs = load_scienceqa_problems(eval_json)
    tasks = build_tasks_from_scienceqa(eval_probs, images_root=images_root, split_filter=split)
    tasks.sort(key=lambda x: (x.get("split",""), x["qid"]))
    total = len(tasks)
    print(f"*** Loaded {total} QA items" + (f" for split={split}" if split else ""))

    # Standard eval config
    mode = "standard"  # single mode
    rows = []
    correct_so_far = 0
    processed_so_far = 0  

    # --- Efficiency Accumulators ---
    total_overhead_tokens = 0
    total_batch_gflops = 0
    total_queries = 0
    
    # Get model params once for GFLOPs
    model_params = sum(p.numel() for p in model.parameters())

    # ------------- Retrieval setup -------------
    retrieval_mode = getattr(args, "retrieval_mode", "none")
    use_retrieval = retrieval_mode in ("incontext", "embed_prefix_no_train", "embed_prefix_w_train")
    print(f"[RAG] retrieval_mode={retrieval_mode} use_retrieval={use_retrieval}")

    faiss_index = None
    train_emb = None
    train_qids = None
    train_task_map: Dict[str, dict] = {}

    if use_retrieval:
        if not getattr(args, "train_embeddings_path", None):
            raise ValueError("retrieval_mode is set but --train_embeddings_path is not provided.")
        ckpt = torch.load(args.train_embeddings_path, map_location="cpu")
        train_emb = ckpt["embeddings"].cpu().numpy().astype("float32")  # [N, D]
        train_qids = [int(q) for q in ckpt["qids"]]          # list[int]
        print("length of train_qids: {0}".format(len(train_qids)))
        N, D_emb = train_emb.shape

         # Normalize and build FAISS index
        train_emb_normed = l2_normalize(train_emb)
        faiss_index = build_faiss_ip_index(train_emb_normed)
        print(f"[Retrieval] FAISS index built with {faiss_index.ntotal} training embeddings (dim={D_emb}).")

        # Build retrieval source from train_json
        train_probs = load_scienceqa_problems(train_json)
        train_tasks = build_tasks_from_scienceqa(train_probs, images_root=images_root, split_filter=None)
        for tt in train_tasks:
            train_task_map[str(tt["qid"])] = tt
        print(f"[Retrieval] Loaded {len(train_task_map)} training tasks for retrieval (split='train').")

        # map qid -> row idx
        qid_to_row = {qid: i for i, qid in enumerate(train_qids)}  # dict[int, int]

        mlp_model = None

        if retrieval_mode == "embed_prefix_w_train":
            if not getattr(args, "retrieval_mlp_ckpt", None):
                raise ValueError("--retrieval_mlp_ckpt is required for retrieval_mode=embed_prefix_w_train")

            assert reprag_token_id is not None, "reprag_token_id must be provided for embed_prefix_w_train"

            # unwrap base HF model
            base_model = model.module if hasattr(model, "module") else model
            H = base_model.config.hidden_size

            mlp_model = PrefixProjector(input_size=H, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1).to(device)
            ckpt = torch.load(args.retrieval_mlp_ckpt, map_location="cpu")

            # --- pick the actual model weights dict ---
            if isinstance(ckpt, dict):
                if "mlp_state" in ckpt:                 # your case
                    state_dict = ckpt["mlp_state"]
                elif "state_dict" in ckpt:              # common in Lightning/others
                    state_dict = ckpt["state_dict"]
                elif "model_state_dict" in ckpt:        # another common convention
                    state_dict = ckpt["model_state_dict"]
                else:
                    # sometimes the checkpoint is already a raw state_dict
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # load weights
            mlp_model.load_state_dict(state_dict, strict=True)

            mlp_model.to(device)
            mlp_model.eval()
            print(f"[MLP] Loaded PrefixProjector from {args.retrieval_mlp_ckpt}")

    # ------------------------------------------------
    # ------------- Main evaluation loop -------------
    # ------------------------------------------------
    for i in range(0, total, batch_size):
        batch_tasks = tasks[i:i+batch_size]

        # Standard ScienceQA: use dataset hint (if any) as context;
        # include images if present (question and optional choice images).
        contexts = [(t.get("context") or "") for t in batch_tasks]

        # Build base messages (no retrieval) + keep images
        base_messages, choice_counts = [], []
        q_imgs, choice_imgs_list = [], []
        for t, ctx in zip(batch_tasks, contexts):
            image_dir = t.get("image_dir") or (os.path.dirname(t["image_path"]) if t.get("image_path") else None)
            q_path, choice_paths = discover_q_choice_by_convention(image_dir, len(t["choices"]))
            q_img = load_pil_or_none(q_path)
            choice_imgs = [load_pil_or_none(p) for p in choice_paths]

            msg = build_multi_image_vqa_message(question=t["question"], choices=t["choices"], question_img=q_img, 
            choice_imgs=choice_imgs, context=ctx)
            base_messages.append(msg)
            choice_counts.append(len(t["choices"]))
            q_imgs.append(q_img)
            choice_imgs_list.append(choice_imgs)

        # Efficiency: Get baseline token count for this batch
        base_tokens = get_batch_token_count(processor, base_messages)
        batch_overhead = 0

        # --- Retrieval & Inference ---
        if not use_retrieval or retrieval_mode == "none":
            # Baseline: just answer with base messages
            preds, raw = batch_vqa_answer(base_messages, model, processor, device, max_new_tokens=8, min_new_tokens=1,
            num_beams=1, do_sample=args.do_sample, temperature=args.temperature if args.do_sample else None)
        else:
            # Build neighbor qids per test item
            batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch_by_embedding(
                base_messages,              
                model=model,
                processor=processor,
                device=device,
                faiss_index=faiss_index,
                train_qids=train_qids,
                top_k=args.top_k,
            )
            if retrieval_mode == "incontext":
                retrieved_visual = getattr(args, "retrieved_visual", "none")

                # Build retrieval-augmented prompts
                aug_messages = []
                for bi, t in enumerate(batch_tasks):
                    neighbor_qids = batch_neighbors_qids[bi]
                    retrieved_tasks = []
                    for qid in neighbor_qids:
                        tt = train_task_map.get(str(qid))
                        if tt is not None:
                            retrieved_tasks.append(tt)
                    if not retrieved_tasks:  # Fallback: if nothing found, just use the base message
                        print("incontext | nothing found")
                        aug_messages.append(base_messages[bi])
                    else:
                        print("process_batch_vqa_from_json | retrieved_visual: {0}".format(retrieved_visual))
                        # ---------- OPTION A: add example IMAGES directly ----------
                        if retrieved_visual == "image":
                            msg = build_retrieval_augmented_message(
                                test_task=t,
                                retrieved_tasks=retrieved_tasks,
                                test_q_img=q_imgs[bi],
                                test_choice_imgs=choice_imgs_list[bi],
                                include_example_images=True,
                                example_captions=None,
                                retrieved_visual_w_choice_img=args.retrieved_visual_w_choice_img
                            )
                        # ---------- OPTION B: add example IMAGE CAPTIONS ----------
                        elif retrieved_visual == "caption":
                            example_captions = []
                            for ex in retrieved_tasks:
                                image_dir = ex.get("image_dir") or (
                                    os.path.dirname(ex["image_path"]) if ex.get("image_path") else None
                                )
                                if image_dir != None:
                                    caption_path = image_dir.replace("images", "GEN_text_Qwen2dot5VL")
                                    caption = load_captions(caption_path, image_only=not args.retrieved_visual_w_choice_img)
                                    example_captions.append(caption)
                                else:
                                    example_captions.append({})

                            msg = build_retrieval_augmented_message(
                                test_task=t,
                                retrieved_tasks=retrieved_tasks,
                                test_q_img=q_imgs[bi],
                                test_choice_imgs=choice_imgs_list[bi],
                                include_example_images=False,
                                example_captions=example_captions,
                                retrieved_visual_w_choice_img=args.retrieved_visual_w_choice_img
                            )
                        # ---------- OPTION C: add both example IMAGE and IMAGE CAPTIONS ----------
                        elif retrieved_visual == "image_n_caption":
                            example_captions = []
                            for ex in retrieved_tasks:
                                image_dir = ex.get("image_dir") or (
                                    os.path.dirname(ex["image_path"]) if ex.get("image_path") else None
                                )
                                if image_dir != None:
                                    caption_path = image_dir.replace("images", "GEN_text_Qwen2dot5VL")
                                    caption = load_captions(caption_path, image_only=True)
                                    example_captions.append(caption)
                                else:
                                    example_captions.append({})

                            msg = build_retrieval_augmented_message(
                                test_task=t,
                                retrieved_tasks=retrieved_tasks,
                                test_q_img=q_imgs[bi],
                                test_choice_imgs=choice_imgs_list[bi],
                                include_example_images=True,
                                example_captions=example_captions,
                                retrieved_visual_w_choice_img=args.retrieved_visual_w_choice_img
                            )
                        # ---------- Original behavior: text-only retrieved examples ----------
                        else:  # "none"
                            msg = build_retrieval_augmented_message(
                                test_task=t,
                                retrieved_tasks=retrieved_tasks,
                                test_q_img=q_imgs[bi],
                                test_choice_imgs=choice_imgs_list[bi],
                                include_example_images=False,
                                example_captions=None,
                            )
                        aug_messages.append(msg)
                aug_tokens = get_batch_token_count(processor, aug_messages)
                """
                print("----- sanity check -----")
                print("base_tokens:", base_tokens)
                print("aug_tokens :", aug_tokens)
                print("overhead   :", aug_tokens - base_tokens)
                print("base_text sample:")
                print(processor.apply_chat_template([base_messages[0]], tokenize=False, add_generation_prompt=True)[:1000])
                print("aug_text sample:")
                print(processor.apply_chat_template([aug_messages[0]], tokenize=False, add_generation_prompt=True)[:1000])
                print("------------------------")
                """
                batch_overhead = (aug_tokens - base_tokens)
                preds, raw = batch_vqa_answer(aug_messages, model, processor, device, max_new_tokens=8, min_new_tokens=1, 
                num_beams=1, do_sample=args.do_sample, temperature=args.temperature if args.do_sample else None)

            elif retrieval_mode == "embed_prefix_no_train":  # utilize first layer sample-wise full mbedding 
                
                # 1) Build prefix embeddings from retrieved examples' token embeddings
                prefix_embs = build_prefix_layer0_for_batch_from_retrieved(batch_neighbors_qids=batch_neighbors_qids, 
                train_task_map=train_task_map, model=model, processor=processor, device=device, max_tokens_per_example=65536)  # [B, Lmax, H]

                batch_overhead = prefix_embs.shape[1] * len(batch_tasks)

                # 2) Answer the base messages but with the prefix injected as extra tokens
                preds, raw = batch_vqa_answer_with_prefix(base_messages, model, processor, device, prefix_embs=prefix_embs, max_new_tokens=8, 
                min_new_tokens=1, num_beams=1, do_sample=args.do_sample, temperature=args.temperature if args.do_sample else None, top_p=None)
            
            elif retrieval_mode == "embed_prefix_w_train":  # utilize last layer sample-wise averaged mbedding 
                if mlp_model is None:
                    raise RuntimeError("MLP model not initialized for embed_prefix_w_train.")

                # 1) Build weighted pooled last-layer embeddings for retrieved examples → [B, H]
                ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved(batch_neighbors_qids=batch_neighbors_qids, 
                batch_neighbors_scores=batch_neighbors_scores, train_task_map=train_task_map, model=model, processor=processor, device=device, 
                output_mode=args.embedding_output_mode, k_concat=args.top_k)

                # sanity
                assert ret_emb_batch.dim() == 3
                assert ret_emb_batch.shape[0] == len(base_messages)
                assert ret_emb_batch.shape[1] == args.top_k

                batch_overhead = args.top_k * len(batch_tasks)

                preds, raw = batch_vqa_answer_with_reprag_hook(
                    messages=base_messages,
                    model=model,
                    processor=processor,
                    device=device,
                    ret_emb_batch=ret_emb_batch,
                    mlp_model=mlp_model,
                    reprag_token_id=reprag_token_id,
                    K=args.top_k,
                    max_new_tokens=8,
                    min_new_tokens=1,
                    num_beams=1,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=None,
                    inj_mult=getattr(args, "inj_mult", 0.1),   # IMPORTANT: default 0.1 like training
                )

        # --- Calculate GFLOPs ---
        # Total tokens = input tokens (base + overhead) + generated tokens
        # We estimate generated tokens as avg_new_tokens * batch_size
        est_gen_tokens = 8 * len(batch_tasks) # based on your max_new_tokens=8
        total_tokens_in_pass = base_tokens + batch_overhead + est_gen_tokens
        
        current_gflops = estimate_gflops(model, total_tokens_in_pass)

        # Accumulate
        total_overhead_tokens += batch_overhead
        total_batch_gflops += current_gflops
        total_queries += len(batch_tasks)

        # ----------------- Metrics + logging -----------------
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

    # --- Final Step: Save Report ---
    final_metrics = {
        "avg_overhead": total_overhead_tokens / total_queries if total_queries > 0 else 0,
        "avg_gflops": total_batch_gflops / total_queries if total_queries > 0 else 0,
        "total_queries": total_queries
    }
    record_efficiency_to_txt(out_dir, final_metrics, args)

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
        'qwen2_5-vl-instruct:3b': "Qwen/Qwen2.5-VL-3B-Instruct",
        'qwen2_5-vl-instruct:7b': "Qwen/Qwen2.5-VL-7B-Instruct",
        'qwen2_5-vl-instruct:72b': "Qwen/Qwen2.5-VL-72B-Instruct"
    }
    model_name = model_map[args.model]

    reprag_token_id = None
    if args.use_deepspeed:
        processor, model, reprag_token_id = load_model_with_deepspeed(model_name)
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    elif args.use_quantization:
        processor, model = load_model_with_quantization(model_name, args.use_quantization)
        device = "cuda"
    else:
        processor, model = load_model_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    if args.retrieval_mode == "embed_prefix_w_train" and reprag_token_id is None:
        if processor.tokenizer.add_tokens([REPRAG_TOKEN]):
            model.resize_token_embeddings(len(processor.tokenizer))
        reprag_token_id = processor.tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)

    # Ensure left padding for Qwen2.5 tokenizer (safer for batched generate)
    processor.tokenizer.padding_side = "left"

    if not args.train_json:
        raise SystemExit("--train_json is required.")
    if not args.val_json:
        raise SystemExit("--val_json is required.")
    if not args.test_json:
        raise SystemExit("--test_json is required.")

    eval_json = get_eval_json_path(args)

    print("[REPRAG] token id:", reprag_token_id)
    print(f"[Eval] eval_on={args.eval_on} | eval_json={eval_json}")
    print(f"[Retrieval] train_json={args.train_json}")

    process_batch_vqa_from_json(
        args,
        model,
        processor,
        device,
        eval_json=eval_json,
        train_json=args.train_json,
        images_root=args.images_root if args.images_root else None,
        out_dir=output_dir,
        batch_size=args.batch_size,
        split=args.split,
        reprag_token_id=reprag_token_id,
    )