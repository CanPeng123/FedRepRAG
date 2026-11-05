"""
KILT-NQ VQA with in-context / embed-prefix retrieval (Qwen3) — v2.

Adds per-query efficiency tracking on top of the original script:
  - avg retrieval overhead tokens/query
  - avg GFLOPs/query
  - efficiency_report.txt written alongside final_results.txt

Usage (see bash/VQA/w_embedding_retrival/KILT_NQ/):
    python Qwen3_Batch_KILT_NQ_VQA_w_retrival_v2.py \
        --model qwen3:8b \
        --data_jsonl /.../datasets/KILT_NQ/validation.jsonl \
        --split validation \
        --retrieval_mode incontext \
        --train_jsonl /.../datasets/KILT_NQ/train.jsonl \
        --train_embeddings_path /.../output/KILT_NQ/train_embeddings/Qwen3_8b_train_q/kilt_nq_embeddings_train.pt \
        --top_k 3 \
        --use_deepspeed \
        --batch_size 1 \
        --max_new_tokens 32 \
        --out_dir /.../output/KILT_NQ/incontext_top3
"""

import os
import sys
import torch
import numpy as np
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AddedToken
from torch.utils.data import DataLoader

from arg_KILT_NQ.KILT_NQ_arg import parse_args
from utilise.KILT_NQ.KILT_NQ_data_process import load_kilt_nq, shard_records, KILTNQDataset, collate_records
from utilise.KILT_NQ.KILT_NQ_prompt_message import (
    build_kilt_nq_message,
    build_kilt_nq_embedding_message,
    build_kilt_nq_incontext_message,
)
from utilise.KILT_NQ.KILT_NQ_post_process import (
    extract_short_answer, exact_match, token_f1,
    write_predictions_csv, write_final_results, save_arguments,
)
from utilise.retrieval_architecture import PrefixProjector
from utilise.retrieval_helper import l2_normalize, build_faiss_ip_index
from utilise.KILT_NQ.KILT_NQ_mlp_utilise import (
    build_kilt_nq_vqa_message_text,
    build_student_prompt_texts_kilt_nq,
    compute_kilt_nq_batch_acc,
)
from utilise.KILT_NQ.KILT_NQ_mlp_train_stage1 import retrieve_train_for_batch, lookup_content_embeddings
from utilise.eval_utils import estimate_gflops, record_efficiency_to_txt

REPRAG_TOKEN = "<RepRAG>"


MODEL_MAP = {
    'qwen3:8b':  'Qwen/Qwen3-8B',
    'qwen3:14b': 'Qwen/Qwen3-14B',
    'qwen3:32b': 'Qwen/Qwen3-32B',
}


def resolve_model_name(key: str) -> str:
    return MODEL_MAP.get(key, key)


# -------------------------------------------------------
# Token counting
# -------------------------------------------------------

def get_token_count_batch(tokenizer, messages_list: list, max_length: int = 4096) -> int:
    """Count total attended tokens for a batch of message-lists (same format as generate_batch)."""
    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for msgs in messages_list
    ]
    inputs = tokenizer(
        texts, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    return int(inputs['attention_mask'].sum().item())


# -------------------------------------------------------
# Model loading
# -------------------------------------------------------

def load_model(args):
    model_name = resolve_model_name(args.model)
    print(f"[model] Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if args.use_deepspeed:
        import deepspeed
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True,
        )
        ds_config = {
            "tensor_parallel": {"tp_size": torch.cuda.device_count()},
            "dtype": "fp16",
            "replace_with_kernel_inject": True,
            "max_tokens": 8192,
        }
        model = deepspeed.init_inference(model, config=ds_config)
        device = torch.device('cuda')

    elif args.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map='auto', trust_remote_code=True,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map='auto', trust_remote_code=True,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    return tokenizer, model, device


# -------------------------------------------------------
# Embedding + FAISS
# -------------------------------------------------------

def embed_batch(messages, tokenizer, model, device, max_length=512):
    """L2-normalised mean-pool embeddings from the last hidden state."""
    texts = [
        tokenizer.apply_chat_template(
            [m], tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        for m in messages
    ]
    inputs = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)

    return emb.cpu().float().numpy()


def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_neighbours(query_messages, tokenizer, model, device, faiss_index, train_ids, top_k):
    q_emb = embed_batch(query_messages, tokenizer, model, device)
    scores, idxs = faiss_index.search(q_emb, top_k)

    batch_ids, batch_scores = [], []
    for b in range(len(query_messages)):
        ids, scs = [], []
        for j, s in zip(idxs[b].tolist(), scores[b].tolist()):
            if 0 <= j < len(train_ids) and np.isfinite(s):
                ids.append(train_ids[j])
                scs.append(float(s))
        batch_ids.append(ids)
        batch_scores.append(scs)

    return batch_ids, batch_scores


# -------------------------------------------------------
# Generation
# -------------------------------------------------------

def generate_batch(messages_list, tokenizer, model, device, args, max_input_length=4096):
    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for msgs in messages_list
    ]
    inputs = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True,
        max_length=max_input_length,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs['input_ids'].shape[1]
    return tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)


# -------------------------------------------------------
# RepRAG generation helper
# -------------------------------------------------------

def generate_batch_reprag(
    base_messages, tokenizer, model, device, model_dtype, args,
    inj_flat, reprag_token_id, K, max_input_length=1024,
):
    """Generate answers with RepRAG injection via embedding-layer forward hook."""
    student_texts = build_student_prompt_texts_kilt_nq(tokenizer, base_messages, k_prefix=K)
    inputs = tokenizer(
        student_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_input_length,
    ).to(device)

    emb_layer = model.get_input_embeddings()
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
        _dynamo_was = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature if args.do_sample else None,
                    top_p=args.top_p if args.do_sample else None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            torch._dynamo.config.disable = _dynamo_was
    finally:
        handle.remove()

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)


# -------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------

def run_evaluation(args, records, train_record_map, faiss_index, train_ids, tokenizer, model, device,
                   mlp=None, content_emb_map=None, reprag_token_id=None, model_dtype=None):
    loader = DataLoader(
        KILTNQDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    results = []
    total_em = total_f1 = 0.0

    # Efficiency accumulators
    total_overhead_tokens = 0
    total_batch_gflops = 0.0
    total_queries = 0

    for batch_idx, batch in enumerate(loader):
        # Base messages (no retrieval context) for overhead baseline
        base_messages_list = [[build_kilt_nq_message(r)] for r in batch]
        base_tokens = get_token_count_batch(tokenizer, base_messages_list)
        batch_overhead = 0

        if args.retrieval_mode == 'none' or faiss_index is None:
            messages_list = base_messages_list
            raw_preds = generate_batch(messages_list, tokenizer, model, device, args)

        elif args.retrieval_mode == 'incontext':
            query_msgs = [build_kilt_nq_embedding_message(r, include_gold=False) for r in batch]
            nb_ids, _ = retrieve_neighbours(
                query_msgs, tokenizer, model, device, faiss_index, train_ids, args.top_k,
            )
            messages_list = []
            for rec, ids in zip(batch, nb_ids):
                retrieved = [train_record_map[i] for i in ids if i in train_record_map]
                messages_list.append([build_kilt_nq_incontext_message(rec, retrieved)])

            aug_tokens = get_token_count_batch(tokenizer, messages_list)
            batch_overhead = aug_tokens - base_tokens
            raw_preds = generate_batch(messages_list, tokenizer, model, device, args)

        elif args.retrieval_mode == 'embed_prefix_w_train':
            K = args.top_k
            B = len(batch)
            query_msgs = [build_kilt_nq_embedding_message(r, include_gold=False) for r in batch]
            nb_ids, _ = retrieve_neighbours(
                query_msgs, tokenizer, model, device, faiss_index, train_ids, K,
            )
            ret_emb_batch = lookup_content_embeddings(nb_ids, content_emb_map, device, K)
            D_content = ret_emb_batch.shape[-1]
            base_mdl = model.module if hasattr(model, "module") else model
            H = base_mdl.config.hidden_size
            mlp_dtype = next(mlp.parameters()).dtype
            inj_raw = mlp(ret_emb_batch.reshape(B * K, D_content).to(mlp_dtype)).reshape(B, K, H)
            inj_flat = inj_raw.to(model_dtype).reshape(B * K, H)
            base_messages = [build_kilt_nq_vqa_message_text(r) for r in batch]
            # Overhead = actual student prompt tokens minus the base prompt tokens.
            # build_student_prompt_texts_kilt_nq prepends "[Retrieved Evidence]\n<RepRAG>*K\n\n"
            # so the overhead is more than just K tokens.
            student_texts = build_student_prompt_texts_kilt_nq(tokenizer, base_messages, k_prefix=K)
            student_inputs = tokenizer(
                student_texts, return_tensors='pt', padding=True,
                truncation=True, max_length=1024,
            )
            reprag_prompt_tokens = int(student_inputs['attention_mask'].sum().item())
            batch_overhead = reprag_prompt_tokens - base_tokens
            raw_preds = generate_batch_reprag(
                base_messages, tokenizer, model, device, model_dtype, args,
                inj_flat, reprag_token_id, K,
            )

        # GFLOPs estimate for this batch
        est_gen_tokens = args.max_new_tokens * len(batch)
        total_tokens_in_pass = base_tokens + batch_overhead + est_gen_tokens
        batch_gflops = estimate_gflops(model, total_tokens_in_pass)

        total_overhead_tokens += batch_overhead
        total_batch_gflops += batch_gflops
        total_queries += len(batch)

        for rec, raw in zip(batch, raw_preds):
            pred_clean = extract_short_answer(raw)
            em = exact_match(pred_clean, rec['gold_answers'])
            f1 = token_f1(pred_clean, rec['gold_answers'])
            total_em += float(em)
            total_f1 += f1
            results.append({
                'id': rec['id'],
                'split': args.split,
                'question': rec['question'],
                'gold_answers': rec['gold_answers'],
                'pred_raw': raw.strip()[:300],
                'pred_clean': pred_clean,
                'em': float(em),
                'f1': round(f1, 4),
            })

        done = min((batch_idx + 1) * args.batch_size, len(records))
        n = len(results)
        print(f"  [batch {batch_idx+1} | {done}/{len(records)}]  "
              f"EM={total_em/n:.4f}  F1={total_f1/n:.4f}  "
              f"overhead={batch_overhead} toks  gflops={batch_gflops:.2f}")
        for rec, raw in zip(batch, raw_preds):
            print(f"    Q: {rec['question'][:80]!r}")
            print(f"    pred: {extract_short_answer(raw)!r}  gold: {rec['gold_answers'][:2]!r}")

    efficiency_metrics = {
        "avg_overhead": total_overhead_tokens / total_queries if total_queries > 0 else 0.0,
        "avg_gflops":   total_batch_gflops   / total_queries if total_queries > 0 else 0.0,
        "total_queries": total_queries,
    }

    return results, efficiency_metrics


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------

def main():
    from datetime import datetime
    args = parse_args()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    args.out_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    save_arguments(args, args.out_dir)

    print(f"[data] Loading validation data from {args.data_jsonl}")
    records = load_kilt_nq(args.data_jsonl)
    records = shard_records(records, args.idx, args.num_parts)
    print(f"[data] Shard {args.idx}/{args.num_parts}: {len(records)} records")

    train_record_map = {}
    faiss_index = None
    train_ids = []
    content_emb_map = None
    mlp = None
    reprag_token_id = None
    model_dtype = None

    if args.retrieval_mode in ('incontext', 'embed_prefix_w_train'):
        if not args.train_jsonl:
            raise ValueError("--train_jsonl required for retrieval")
        if not args.train_embeddings_path:
            raise ValueError("--train_embeddings_path required for retrieval")

        print(f"[data] Loading train corpus from {args.train_jsonl}")
        train_records = load_kilt_nq(args.train_jsonl)
        train_record_map = {r['id']: r for r in train_records}
        print(f"[data] {len(train_records)} train records loaded")

        print(f"[data] Loading train embeddings from {args.train_embeddings_path}")
        ckpt = torch.load(args.train_embeddings_path, map_location='cpu')
        train_emb = ckpt['embeddings'].float().numpy()
        train_ids = [str(i) for i in ckpt['qids']]
        faiss_index = build_faiss_index(train_emb)
        print(f"[data] FAISS index built: {train_emb.shape}")

    tokenizer, model, device = load_model(args)
    model_dtype = next(model.parameters()).dtype

    if args.retrieval_mode == 'embed_prefix_w_train':
        if not args.content_embeddings_path:
            raise ValueError("--content_embeddings_path required for embed_prefix_w_train")
        if not args.retrieval_mlp_ckpt:
            raise ValueError("--retrieval_mlp_ckpt required for embed_prefix_w_train")

        print(f"[data] Loading content embeddings from {args.content_embeddings_path}")
        c_ckpt = torch.load(args.content_embeddings_path, map_location='cpu')
        content_emb_raw = c_ckpt['embeddings'].float().numpy()
        content_ids = [str(i) for i in c_ckpt['qids']]
        content_emb_map = {i: content_emb_raw[idx] for idx, i in enumerate(content_ids)}
        D_content = content_emb_raw.shape[1]
        print(f"[data] Content embeddings: {content_emb_raw.shape}")

        base_model = model.module if hasattr(model, "module") else model
        num_added = tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
        if num_added > 0:
            base_model.resize_token_embeddings(len(tokenizer))
        reprag_token_id = tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)
        assert reprag_token_id != tokenizer.unk_token_id, "<RepRAG> mapped to unk"
        print(f"[token] <RepRAG> id = {reprag_token_id}")

        H = base_model.config.hidden_size
        mlp = PrefixProjector(input_size=D_content, output_size=H,
                              projector_type="mlp2x_gelu", p_drop=0.0).to(device)
        ckpt_state = torch.load(args.retrieval_mlp_ckpt, map_location='cpu')
        mlp_state = ckpt_state.get("mlp_state") or ckpt_state.get("mlp_state_dict") or ckpt_state
        mlp.load_state_dict(mlp_state)
        mlp.eval()
        print(f"[MLP] Loaded PrefixProjector from {args.retrieval_mlp_ckpt}")

    results, efficiency_metrics = run_evaluation(
        args, records, train_record_map, faiss_index, train_ids,
        tokenizer, model, device,
        mlp=mlp, content_emb_map=content_emb_map,
        reprag_token_id=reprag_token_id, model_dtype=model_dtype,
    )

    csv_path = write_predictions_csv(results, args.out_dir)
    print(f"[saved] Predictions: {csv_path}")
    write_final_results(results, args.out_dir)
    print(f"[saved] Results: {args.out_dir}/final_results.txt")

    record_efficiency_to_txt(args.out_dir, efficiency_metrics, args)


if __name__ == '__main__':
    main()