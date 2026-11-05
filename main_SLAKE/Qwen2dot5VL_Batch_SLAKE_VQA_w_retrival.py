import os
import sys
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import re

import faiss
from qwen_vl_utils import process_vision_info
from transformers.tokenization_utils import AddedToken

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from VQA.utilise.model_loading import load_model_with_quantization, load_model_with_deepspeed, load_model_standard, load_gemma3_standard
from VQA.utilise.retrieval_architecture import PrefixProjector
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index, retrieve_neighbors_for_batch_by_embedding
from VQA.utilise.eval_utils import estimate_gflops, get_batch_token_count, record_efficiency_to_txt

from VQA.utilise.mlp_utilise.mlp_train_stage2_utilise import build_student_prompt_texts

from VQA.arg_SLAKE.Slake_arg import parse_args
from VQA.utilise.SLAKE.Slake_data_process import load_slake_items, build_tasks_from_slake, load_pil_or_none, load_captions_map
from VQA.utilise.SLAKE.Slake_post_process import normalize_answer_with_numbers, answers_match
from VQA.utilise.SLAKE.Slake_prompt_message import build_slake_vqa_message, build_slake_rag_incontext_message
from VQA.utilise.SLAKE.Slake_result_record import write_final_results_txt, make_stats_bucket, norm_cat, update_stats
from VQA.utilise.SLAKE.Slake_eval_utils import score_and_record_batch, strip_answer_prefix

from VQA.utilise.SLAKE.Slake_retrieval_utilise import build_prefix_last_layer_for_batch_from_retrieved


REPRAG_TOKEN = "<RepRAG>"


def _to_gemma_messages(messages):
    """Strip PIL images from content dicts (Gemma format) and return (gemma_msgs, flat_pil_list)."""
    from PIL.Image import Image as PILImage
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


# =========================================================
# Helper
# =========================================================
def _unwrap_base_model(model):
    return model.module if hasattr(model, "module") else model
    

def build_prefix_layer0_for_batch_from_retrieved_slake(batch_neighbors_qids: List[List[int]], train_task_map: Dict[str, Dict[str, Any]], model, processor, device, 
images_root: Optional[str] = None, max_tokens_per_example: int = 2048) -> torch.Tensor:
    """
    Build [B, Lmax, H] by concatenating token-embedding sequences (layer0) from retrieved examples.
    This is the SLAKE counterpart of build_prefix_layer0_for_batch_from_retrieved.
    """
    base_model = _unwrap_base_model(model)
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype
    embed_layer = base_model.get_input_embeddings()

    B = len(batch_neighbors_qids)

    # 1) Flatten retrieved messages
    flat_msgs = []
    flat_group_lens = []
    for qids in batch_neighbors_qids:
        group = []
        for qid in qids:
            tt = train_task_map.get(str(qid))
            if tt is None:
                continue
            img = load_pil_or_none(tt.get("image_path"))
            group.append(
                build_slake_vqa_message(
                    question=tt["question"],
                    image=img,
                    answer_type=tt.get("answer_type"),
                    base_type=tt.get("base_type"),
                    triple=tt.get("triple"),
                )
            )
        flat_group_lens.append(len(group))
        flat_msgs.extend(group)

    if len(flat_msgs) == 0:
        # no retrieved examples at all
        H = getattr(base_model.config, "hidden_size", None) or getattr(getattr(base_model.config, "text_config", None), "hidden_size", None)
        return torch.zeros((B, 0, H), dtype=torch.float32, device=model_device)

    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in flat_msgs]
    image_inputs, video_inputs = process_vision_info(flat_msgs)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]  # [Nflat, T]
    attn = inputs["attention_mask"]  # [Nflat, T]

    # 2) Layer0 embeddings (text token embeddings; vision placeholders are still ids here)
    with torch.no_grad():
        embs = embed_layer(input_ids).to(dtype=model_dtype)  # [Nflat, T, H]

    # 3) For each retrieved example, keep only non-pad tokens (and optionally truncate)
    ex_seqs = []
    Nflat, T, H = embs.shape
    for i in range(Nflat):
        tlen = int(attn[i].sum().item())
        tlen = min(tlen, max_tokens_per_example)
        ex_seqs.append(embs[i, :tlen, :])

    # 4) Pack per test sample by concatenation
    prefixes = []
    idx = 0
    maxL = 0
    for glen in flat_group_lens:
        if glen == 0:
            prefixes.append(None)
            continue
        seqs = ex_seqs[idx : idx + glen]
        idx += glen
        cat = torch.cat(seqs, dim=0)  # [Li, H]
        prefixes.append(cat)
        maxL = max(maxL, cat.size(0))

    # 5) Pad to [B, Lmax, H]
    out = torch.zeros((B, maxL, H), device=model_device, dtype=model_dtype)
    for b in range(B):
        if prefixes[b] is None:
            continue
        Lb = prefixes[b].size(0)
        out[b, :Lb, :] = prefixes[b]

    return out


# =========================================================
# Generation: batch function to generate answer for SLAKE QA
# =========================================================
def batch_vqa_answer_text(messages: List[Dict], model, processor, device, max_new_tokens: int = 16, min_new_tokens: int = 1, num_beams: int = 1, do_sample: bool = False,
temperature: Optional[float] = None, top_p: Optional[float] = None, is_gemma: bool = False) -> List[str]:
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
    return decoded


def batch_vqa_answer_text_with_prefix(messages: List[Dict], model, processor, device, prefix_embs: torch.Tensor, max_new_tokens: int = 16, min_new_tokens: int = 1, num_beams: int = 1, 
do_sample: bool = False, temperature: Optional[float] = None, top_p: Optional[float] = None) -> List[str]:
    """
    SLAKE version of ScienceQA batch_vqa_answer_with_prefix:
    inject prefix embeddings before the text tokens via inputs_embeds,
    while still passing input_ids for Qwen2.5-VL to correctly overwrite vision placeholder slots.
    """
    base_model = _unwrap_base_model(model)
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype

    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    B, T = input_ids.shape

    # Base token embeddings
    with torch.no_grad():
        embed_layer = base_model.get_input_embeddings()
        base_embs = embed_layer(input_ids).to(dtype=model_dtype)  # [B, T, H]

    # Prefix concat
    if prefix_embs is not None and prefix_embs.numel() > 0 and prefix_embs.size(1) > 0:
        assert prefix_embs.shape[0] == B, "prefix_embs batch mismatch"
        prefix_embs = prefix_embs.to(device=model_device, dtype=model_dtype)
        L = prefix_embs.shape[1]

        new_embs = torch.cat([prefix_embs, base_embs], dim=1)  # [B, L+T, H]
        prefix_mask = torch.ones((B, L), device=model_device, dtype=attn_mask.dtype)
        new_attn_mask = torch.cat([prefix_mask, attn_mask], dim=1)  # [B, L+T]

        # Dummy prefix ids (must NOT be vision placeholders or special tokens)
        tok = processor.tokenizer
        dummy_id = tok.encode(".", add_special_tokens=False)[0]
        special_ids = set(tok.all_special_ids)
        if dummy_id in special_ids:
            dummy_id = tok.encode("the", add_special_tokens=False)[0]
        if dummy_id in special_ids:
            for cand in range(tok.vocab_size):
                if cand not in special_ids:
                    dummy_id = cand
                    break
        dummy_prefix_ids = torch.full((B, L), fill_value=dummy_id, device=model_device, dtype=input_ids.dtype)
        new_input_ids = torch.cat([dummy_prefix_ids, input_ids], dim=1)  # [B, L+T]
    else:
        L = 0
        new_embs = base_embs
        new_attn_mask = attn_mask
        new_input_ids = input_ids

    prompt_len = new_embs.shape[1]
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

    generate_kwargs = {
        "inputs_embeds": new_embs,
        "input_ids": new_input_ids,          # CRITICAL for Qwen2.5-VL vision slot replacement
        "attention_mask": new_attn_mask,
    }
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        generate_kwargs[k] = v

    with torch.no_grad():
        out = model.generate(**generate_kwargs, **gen_args)

    # Trim using prompt_len
    trimmed = []
    for seq in out:
        trimmed.append(seq[prompt_len:] if seq.shape[0] > prompt_len else seq)

    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded = [d.strip().splitlines()[0].strip() for d in decoded]
    return decoded


def batch_vqa_answer_text_with_reprag_hook(messages: List[Dict], base_model, processor, device, model_dtype, ret_emb_batch: torch.Tensor, mlp_model, reprag_token_id: int, K: int, 
max_new_tokens: int = 16, min_new_tokens: int = 1, num_beams: int = 1, do_sample: bool = False, temperature: Optional[float] = None, top_p: Optional[float] = None) -> List[str]:
    """
    Inject retrieval embeddings into K <RepRAG> prefix tokens via input-embedding hook,
    then generate answers for a batch of VQA messages.

    Expected:
        ret_emb_batch: [B, K, H]
    """
    mlp_was_training = mlp_model.training
    base_model.eval()
    mlp_model.eval()

     # ---- 1) Check retrieval embeddings ----
    if ret_emb_batch.dim() != 3:
        raise ValueError(f"ret_emb_batch must be [B, K, H], got shape={tuple(ret_emb_batch.shape)}")

    B, K2, H = ret_emb_batch.shape
    hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(getattr(base_model.config, "text_config", None), "hidden_size", None)

    if K2 != K:
        raise ValueError(f"K mismatch: ret_emb_batch has K={K2}, expected K={K}")
    if H != hidden_size:
        raise ValueError(f"Hidden size mismatch: ret_emb_batch has H={H}, model hidden size={hidden_size}")

    # ---- 2) Project retrieval embeddings ----
    mlp_dtype = next(mlp_model.parameters()).dtype
    inj_raw = mlp_model(ret_emb_batch.reshape(B * K, H).to(mlp_dtype)).reshape(B, K, H)
    inj_flat = inj_raw.to(dtype=model_dtype).reshape(B * K, H)

    # ---- 3) Build prompts with K <RepRAG> tokens ----
    prompt_texts = build_student_prompt_texts(processor, messages, k_prefix=K)
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_inputs = processor(text=prompt_texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

    input_ids = prompt_inputs["input_ids"]
    attn_mask = prompt_inputs["attention_mask"]

     # ---- 4) Sanity check: exactly K <RepRAG> tokens per sample ----
    xmask = (input_ids == reprag_token_id)
    counts = xmask.sum(dim=1).tolist()
    if not all(c == K for c in counts):
        raise RuntimeError(
            f"Expected exactly K={K} <RepRAG> tokens per sample, got counts={counts}. "
            f"Likely <RepRAG> is not a single token or prompt template changed."
        )

    # ---- 5) Register embedding injection hook ----
    emb_layer = base_model.get_input_embeddings()
    fired = {"done": False}

    def _inject_hook(module, hook_inputs, output):
        if fired["done"]:
            return output
        ids = hook_inputs[0]  # [B, T]
        mask = (ids == reprag_token_id)
        if not mask.any():
            return output
        out = output.clone()
        out[mask] = inj_flat.to(out.dtype)
        fired["done"] = True
        return out

    handle = emb_layer.register_forward_hook(_inject_hook)

    try:
        # ---- 6) Prepare generation args ----
        prompt_len = input_ids.shape[1]
        min_length = prompt_len + int(min_new_tokens)
        gen_args = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            num_beams=int(num_beams),
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            min_length=min_length,
        )

        if do_sample:
            if temperature is not None:
                gen_args["temperature"] = float(temperature)
            if top_p is not None:
                gen_args["top_p"] = float(top_p)

        # ---- 7) Explicitly pass all processor outputs into generate ----
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
        }
        for k_in, v_in in prompt_inputs.items():
            if k_in in ("input_ids", "attention_mask"):
                continue
            generate_kwargs[k_in] = v_in

        with torch.no_grad():
            out = base_model.generate(**generate_kwargs, **gen_args)

        # ---- 8) Trim prompt and decode ----
        trimmed = [seq[prompt_len:] if seq.shape[0] > prompt_len else seq for seq in out]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        pred_texts = []
        for d in decoded:
            s = strip_answer_prefix(d).strip()
            pred_texts.append(s)

        return pred_texts

    finally:
        handle.remove()
        mlp_model.train(mlp_was_training)


# =========================================================
# SLAKE eval with optional retrieval
# =========================================================
def process_batch_vqa_from_slake_json_rag(args, model, processor, device, slake_json: str, images_root: Optional[str], out_dir: str,
batch_size: int, split: Optional[str], lang: str, reprag_token_id: int | None = None, is_gemma: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    # --- load eval tasks ---
    items = load_slake_items(slake_json)
    tasks = build_tasks_from_slake(items, images_root=images_root, split_name=split, lang_filter=lang)
    tasks.sort(key=lambda x: x["qid"])
    total = len(tasks)
    print(f"*** Loaded {total} SLAKE QA items (lang={lang})" + (f" for split={split}" if split else ""))

    # --- retrieval setup ---
    retrieval_mode = getattr(args, "retrieval_mode", "none")
    use_retrieval = retrieval_mode != "none"
    print(f"[RAG] retrieval_mode={retrieval_mode} use_retrieval={use_retrieval}")

    faiss_index = None
    train_emb = None
    train_qids = None
    train_task_map: Dict[str, Dict[str, Any]] = {}

    captions_map = {}

    if use_retrieval:
        if not args.train_json:
            raise ValueError("--train_json is required when retrieval_mode != none")
        if not args.train_embeddings_path:
            raise ValueError("--train_embeddings_path is required when retrieval_mode != none")

        ckpt = torch.load(args.train_embeddings_path, map_location="cpu")
        train_emb = ckpt["embeddings"]
        if isinstance(train_emb, torch.Tensor):
            train_emb = train_emb.cpu().to(torch.float32).numpy()
        train_emb = train_emb.astype("float32")
        train_qids = [int(q) for q in ckpt["qids"]]
        print(f"[RAG] Loaded train embeddings: {train_emb.shape} (qids={len(train_qids)})")

        train_emb_normed = l2_normalize(train_emb)
        faiss_index = build_faiss_ip_index(train_emb_normed)
        print(f"[RAG] FAISS built. ntotal={faiss_index.ntotal}")

        train_items = load_slake_items(args.train_json)
        train_tasks = build_tasks_from_slake(train_items, images_root=images_root, split_name="train", lang_filter=lang)
        for tt in train_tasks:
            train_task_map[str(tt["qid"])] = tt
        print(f"[RAG] train_task_map size={len(train_task_map)}")

        mlp_model = None

        if retrieval_mode == "incontext":
            if args.retrieved_caption_json:
                captions_map = load_captions_map(args.retrieved_caption_json)
                print(f"[RAG] captions_map loaded: {len(captions_map)} captions")

        if retrieval_mode == "embed_prefix_w_train":
            if not getattr(args, "retrieval_mlp_ckpt", None):
                raise ValueError("--retrieval_mlp_ckpt is required for retrieval_mode=embed_prefix_w_train")

            assert reprag_token_id is not None, "reprag_token_id must be provided for embed_prefix_w_train"

            # unwrap base HF model
            base_model = model.module if hasattr(model, "module") else model
            model_dtype = next(base_model.parameters()).dtype
            H = getattr(base_model.config, "hidden_size", None) or getattr(getattr(base_model.config, "text_config", None), "hidden_size", None)

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

    # --- metrics + breakdown ---
    rows = []
    state = {
        "correct_all": 0, "seen_all": 0,
        "correct_open": 0, "seen_open": 0,
        "correct_closed": 0, "seen_closed": 0,
    }
    stats = {
        "modality":     {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "location":     {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "content_type": {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
        "base_type":    {"all": make_stats_bucket(), "open": make_stats_bucket(), "closed": make_stats_bucket()},
    }

    # --- Efficiency Accumulators ---
    total_overhead_tokens = 0
    total_batch_gflops = 0
    total_queries = 0

    # ------------- Main evaluation loop -------------
    for i in range(0, total, batch_size):
        batch_tasks = tasks[i:i+batch_size]

        # build base messages
        base_messages = []
        base_imgs = []
        for t in batch_tasks:
            img = load_pil_or_none(t.get("image_path"))
            base_imgs.append(img)
            base_messages.append(build_slake_vqa_message(question=t["question"], image=img, answer_type=t.get("answer_type"), base_type=t.get("base_type"), 
            triple=t.get("triple")))

        # --- Efficiency: baseline token count for this batch ---
        base_tokens = get_batch_token_count(processor, base_messages)
        batch_overhead = 0

        # ----------------- Retrieval logic -----------------
        if (not use_retrieval) or retrieval_mode == "none":
            pred_texts = batch_vqa_answer_text(
                base_messages, model, processor, device,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                is_gemma=is_gemma,
            )
        else:
            # get neighbors
            batch_neighbors_qids, batch_neighbors_scores = retrieve_neighbors_for_batch_by_embedding(base_messages, model=model, processor=processor, device=device, 
            faiss_index=faiss_index, train_qids=train_qids, top_k=args.top_k)

            if retrieval_mode == "incontext":
                retrieved_visual = getattr(args, "retrieved_visual", "none")

                aug_messages = []
                for bi, t in enumerate(batch_tasks):
                    neighbor_qids = batch_neighbors_qids[bi]

                    retrieved_tasks = []
                    retrieved_imgs = []
                    retrieved_caps = []

                    for qid in neighbor_qids:
                        tt = train_task_map.get(str(qid))
                        if tt is None:
                            continue

                        retrieved_tasks.append(tt)

                        # load retrieved example image (only if needed)
                        if retrieved_visual in ("image", "image_n_caption"):
                            retrieved_imgs.append(load_pil_or_none(tt.get("image_path")))
                        else:
                            retrieved_imgs.append(None)

                        # load caption (only if needed)
                        if retrieved_visual in ("caption", "image_n_caption"):
                            cap = captions_map.get(str(tt["qid"]), "")
                            retrieved_caps.append(cap)
                        else:
                            retrieved_caps.append("")

                    if not retrieved_tasks:
                        aug_messages.append(base_messages[bi])
                        continue

                    include_example_images = (retrieved_visual in ("image", "image_n_caption"))
                    include_example_captions = (retrieved_visual in ("caption", "image_n_caption"))

                    aug_messages.append(
                        build_slake_rag_incontext_message(test_task=t, test_img=base_imgs[bi], retrieved_tasks=retrieved_tasks, 
                        retrieved_imgs=retrieved_imgs, retrieved_caps=retrieved_caps, include_example_images=include_example_images, 
                        include_example_captions=include_example_captions)
                        )

                aug_tokens = get_batch_token_count(processor, aug_messages)
                batch_overhead = aug_tokens - base_tokens

                pred_texts = batch_vqa_answer_text(aug_messages, model, processor, device, max_new_tokens=args.max_new_tokens,
                min_new_tokens=1, num_beams=args.num_beams, do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None, top_p=args.top_p if args.do_sample else None,
                is_gemma=is_gemma)

            elif retrieval_mode == "embed_prefix_no_train":
                # 1) Build layer0 prefix token embeddings by concatenating retrieved examples
                prefix_embs = build_prefix_layer0_for_batch_from_retrieved_slake(
                    batch_neighbors_qids=batch_neighbors_qids,
                    train_task_map=train_task_map,
                    model=model,
                    processor=processor,
                    device=device,
                    images_root=images_root,
                    max_tokens_per_example=getattr(args, "max_tokens_per_example", 2048),
                )  # [B, Lmax, H]

                batch_overhead = prefix_embs.shape[1] * len(batch_tasks)

                # 2) Answer base messages with prefix injected
                pred_texts = batch_vqa_answer_text_with_prefix(
                    base_messages,
                    model,
                    processor,
                    device,
                    prefix_embs=prefix_embs,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None,
                )

            elif retrieval_mode == "embed_prefix_w_train":
                if "mlp_model" not in locals() or mlp_model is None:
                    raise RuntimeError("mlp_model not initialized for embed_prefix_w_train. See MLP/RepRAG setup block.")

                if "reprag_token_id" not in locals() or reprag_token_id is None:
                    raise RuntimeError("reprag_token_id not initialized for embed_prefix_w_train. See MLP/RepRAG setup block.")

                # 1) Build last-layer pooled embeddings for each retrieved example: [B, K, H]
                ret_emb_batch = build_prefix_last_layer_for_batch_from_retrieved(
                    batch_neighbors_qids=batch_neighbors_qids,
                    batch_neighbors_scores=batch_neighbors_scores,
                    train_task_map=train_task_map,
                    model=model,
                    processor=processor,
                    device=device,
                    output_mode=args.embedding_output_mode,
                    k_concat=args.top_k
                )

                batch_overhead = args.top_k * len(batch_tasks)

                # 2) Inject via RepRAG hook and generate
                pred_texts = batch_vqa_answer_text_with_reprag_hook(
                    messages=base_messages,
                    base_model=base_model,
                    processor=processor,
                    device=device,
                    model_dtype=model_dtype,
                    ret_emb_batch=ret_emb_batch,
                    mlp_model=mlp_model,
                    reprag_token_id=reprag_token_id,
                    K=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=1,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None
                )
            else:
                pred_texts = batch_vqa_answer_text(base_messages, model, processor, device, is_gemma=is_gemma)

        # --- Efficiency: GFLOPs estimate ---
        est_gen_tokens = args.max_new_tokens * len(batch_tasks)
        total_tokens_in_pass = base_tokens + batch_overhead + est_gen_tokens
        current_gflops = estimate_gflops(model, total_tokens_in_pass)

        total_overhead_tokens += batch_overhead
        total_batch_gflops += current_gflops
        total_queries += len(batch_tasks)

        # scoring + record
        score_and_record_batch(batch_tasks=batch_tasks, pred_texts=pred_texts, rows=rows, stats=stats, state=state, retrieval_mode=retrieval_mode, verbose=False)

        done = min(i + batch_size, total)
        acc_all = 100.0 * state["correct_all"] / max(state["seen_all"], 1)
        acc_open = 100.0 * state["correct_open"] / max(state["seen_open"], 1)
        acc_closed = 100.0 * state["correct_closed"] / max(state["seen_closed"], 1)

        print(
            f"Processed {done}/{total} | "
            f"correct={state['correct_all']}/{state['seen_all']} | "
            f"acc={acc_all:.2f}% (open={state['correct_open']}/{state['seen_open']}={acc_open:.2f}%, "
            f"closed={state['correct_closed']}/{state['seen_closed']}={acc_closed:.2f}%)"
        )

        torch.cuda.empty_cache()

    final_metrics = {
        "avg_overhead": total_overhead_tokens / total_queries if total_queries > 0 else 0,
        "avg_gflops": total_batch_gflops / total_queries if total_queries > 0 else 0,
        "total_queries": total_queries,
    }
    record_efficiency_to_txt(out_dir, final_metrics, args)
    
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f"slake_preds_rag_{split or 'all'}_{lang}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    write_final_results_txt(out_dir=out_dir, args=args, split=split, lang=lang, total_questions=total, correct_all=state["correct_all"], seen_all=state["seen_all"], 
    correct_open=state["correct_open"], seen_open=state["seen_open"], correct_closed=state["correct_closed"], seen_closed=state["seen_closed"], out_csv_path=out_csv, stats=stats)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # save args
    arg_file = os.path.join(output_dir, "argument.txt")
    with open(arg_file, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
    print(f"Arguments saved to {arg_file}")

    model_map = {
        "qwen2_5-vl-instruct:2b": "Qwen/Qwen2.5-VL-2B-Instruct",
        "qwen2_5-vl-instruct:3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2_5-vl-instruct:7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2_5-vl-instruct:72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        "gemma3:4b": "google/gemma-3-4b-it",
        "gemma3:12b": "google/gemma-3-12b-it",
    }
    model_name = model_map[args.model]
    is_gemma = args.model.startswith("gemma3")

    if is_gemma:
        processor, qwen_model = load_gemma3_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reprag_token_id = None
    elif args.use_deepspeed:
        processor, qwen_model, reprag_token_id = load_model_with_deepspeed(model_name)
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    elif args.use_quantization:
        processor, qwen_model = load_model_with_quantization(model_name, args.use_quantization)
        device = "cuda"
        reprag_token_id = None
    else:
        processor, qwen_model = load_model_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qwen_model.to(device)
        reprag_token_id = None

    processor.tokenizer.padding_side = "left"

    # For embed_prefix_w_train the <RepRAG> token must exist in the vocabulary.
    # Add it here (no-op if already present) and resize embeddings to match.
    retrieval_mode = getattr(args, "retrieval_mode", "none")
    if retrieval_mode == "embed_prefix_w_train":
        base_model = qwen_model.module if hasattr(qwen_model, "module") else qwen_model
        num_added = processor.tokenizer.add_tokens(
            [AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)]
        )
        if num_added > 0:
            base_model.resize_token_embeddings(len(processor.tokenizer))
        reprag_token_id = processor.tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)
        assert reprag_token_id != processor.tokenizer.unk_token_id, \
            f"<RepRAG> mapped to unk — token was not added correctly"

    print("[REPRAG] token id:", reprag_token_id)

    process_batch_vqa_from_slake_json_rag(args=args, model=qwen_model, processor=processor, device=device, slake_json=args.problems_json,
    images_root=args.images_root if args.images_root else None, out_dir=output_dir, batch_size=args.batch_size, split=args.split,
    lang=args.lang, reprag_token_id=reprag_token_id, is_gemma=is_gemma)
