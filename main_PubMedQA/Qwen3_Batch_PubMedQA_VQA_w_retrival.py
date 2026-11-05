import os
import sys
import torch
import numpy as np
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader

from arg_PubMedQA.PubMedQA_arg import parse_args
from utilise.PubMedQA.PubMedQA_data_process import (
    load_benchmark, load_train, load_pubmedqa_train, load_pubmed_corpus, load_combined_pool,
    build_train_task_map, shard_tasks, PubMedQADataset, collate_tasks,
)
from utilise.PubMedQA.PubMedQA_prompt_message import (
    build_mcq_vqa_message, build_mcq_incontext_message, build_mcq_embedding_message,
    build_pubmedqa_incontext_message, build_corpus_incontext_message,
    build_pubmedqa_embedding_message, build_combined_pool_embedding_message,
    build_combined_incontext_message,
)
from utilise.PubMedQA.PubMedQA_post_process import parse_answer_letter, is_correct, write_predictions_csv, write_final_results, save_arguments
from utilise.retrieval_architecture import PrefixProjector
from utilise.PubMedQA.PubMedQA_mlp_utilise import build_student_prompt_texts_pubmedqa, build_mcq_vqa_message_text
from utilise.eval_utils import estimate_gflops, record_efficiency_to_txt
from transformers import AddedToken


# -------------------------------------------------------
# Model name resolution
# -------------------------------------------------------
MODEL_MAP = {
    'qwen3:8b':  'Qwen/Qwen3-8B',
    'qwen3:14b': 'Qwen/Qwen3-14B',
    'qwen3:32b': 'Qwen/Qwen3-32B',
}


def resolve_model_name(key: str) -> str:
    return MODEL_MAP.get(key, key)


# -------------------------------------------------------
# Token counting (text-only, no processor/vision)
# -------------------------------------------------------
def get_token_count_text(tokenizer, messages: list, max_length: int = 8192) -> int:
    """Count total attended tokens for a batch of chat messages."""
    texts = [
        tokenizer.apply_chat_template(
            [m], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for m in messages
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
# Embedding utilities
# -------------------------------------------------------

def embed_batch(messages, tokenizer, model, device):
    """Embed a batch of chat messages → L2-normalised float32 numpy [B, D]."""
    texts = [
        tokenizer.apply_chat_template(
            [m], tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        for m in messages
    ]
    inputs = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad():
        outs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outs.hidden_states[-1]                              # [B, T, D]
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # [B, D]
        emb = emb / emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)

    return emb.cpu().float().numpy()


def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_neighbours(query_messages, tokenizer, model, device, faiss_index, train_qids, top_k):
    q_emb = embed_batch(query_messages, tokenizer, model, device)
    scores, idxs = faiss_index.search(q_emb, top_k)

    batch_qids, batch_scores = [], []
    for b in range(len(query_messages)):
        qids, scs = [], []
        for j, s in zip(idxs[b].tolist(), scores[b].tolist()):
            if 0 <= j < len(train_qids) and np.isfinite(s):
                qids.append(train_qids[j])
                scs.append(float(s))
        batch_qids.append(qids)
        batch_scores.append(scs)

    return batch_qids, batch_scores


# -------------------------------------------------------
# Embed-prefix (no training)
# -------------------------------------------------------
def build_prefix_embeds_no_train(neighbour_qids_batch, train_task_map, tokenizer, model, device, max_tokens_per_example=128):
    embed_layer = model.get_input_embeddings()
    B = len(neighbour_qids_batch)
    all_prefix = []

    for b in range(B):
        parts = []
        for qid in neighbour_qids_batch[b]:
            ex = train_task_map.get(qid)
            if ex is None:
                continue
            msg = build_mcq_embedding_message(ex, include_gold=True)
            text = tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            ids = tokenizer(text, return_tensors='pt', truncation=True,
                            max_length=max_tokens_per_example).input_ids.to(device)
            with torch.no_grad():
                tok_emb = embed_layer(ids)[0]
            parts.append(tok_emb)

        if parts:
            all_prefix.append(torch.cat(parts, dim=0))
        else:
            D = embed_layer.embedding_dim
            all_prefix.append(torch.zeros(1, D, device=device))

    max_len = max(p.shape[0] for p in all_prefix)
    D = all_prefix[0].shape[1]
    prefix_pad = torch.zeros(B, max_len, D, device=device)
    lengths = []
    for b, p in enumerate(all_prefix):
        prefix_pad[b, :p.shape[0]] = p
        lengths.append(p.shape[0])

    return prefix_pad, lengths


# -------------------------------------------------------
# Generation
# -------------------------------------------------------
def generate_batch(messages, tokenizer, model, device, args, max_input_length=8192):
    texts = [
        tokenizer.apply_chat_template(
            [m], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for m in messages
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


def generate_batch_with_prefix(messages, prefix_embeds, prefix_lengths, tokenizer, model, device, args):
    texts = [
        tokenizer.apply_chat_template(
            [m], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for m in messages
    ]
    inputs = tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True, max_length=512,
    ).to(device)

    embed_layer = model.get_input_embeddings()
    base_embeds = embed_layer(inputs['input_ids'])
    B, T_base, D = base_embeds.shape
    L_pre = prefix_embeds.shape[1]

    full_embeds = torch.cat([prefix_embeds, base_embeds], dim=1)

    pre_mask = torch.zeros(B, L_pre, dtype=inputs['attention_mask'].dtype, device=device)
    for b, l in enumerate(prefix_lengths):
        pre_mask[b, :l] = 1
    full_mask = torch.cat([pre_mask, inputs['attention_mask']], dim=1)

    with torch.no_grad():
        out_ids = model.generate(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = full_embeds.shape[1]
    return tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)


# -------------------------------------------------------
# MLP inference helpers
# -------------------------------------------------------

REPRAG_TOKEN = "<RepRAG>"


def load_mlp(ckpt_path, device, base_model):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["mlp_state"]
    first_w = state.get("projector.0.weight")
    H = base_model.config.hidden_size
    D_content = first_w.shape[1] if first_w is not None else H
    mlp = PrefixProjector(input_size=D_content, output_size=H, projector_type="mlp2x_gelu", p_drop=0.0)
    mlp.load_state_dict(state, strict=True)
    mlp.to(device).eval()
    print(f"[MLP] Loaded from {ckpt_path} (D_content={D_content}, H={H})")
    return mlp, D_content


def generate_batch_with_mlp(
    batch_tasks, neighbour_qids_batch, content_emb_map,
    tokenizer, base_model, mlp, device, args,
    reprag_token_id, D_content, model_dtype,
):
    K = args.top_k
    B = len(batch_tasks)
    H = base_model.config.hidden_size

    pad = torch.zeros(D_content, dtype=torch.float32)
    ret_embs = []
    for qids in neighbour_qids_batch:
        embs = []
        for qid in qids[:K]:
            vec = content_emb_map.get(str(qid))
            embs.append(torch.as_tensor(vec, dtype=torch.float32) if vec is not None else pad)
        while len(embs) < K:
            embs.append(pad)
        ret_embs.append(torch.stack(embs, dim=0))
    ret_emb_batch = torch.stack(ret_embs, dim=0).to(device)

    with torch.no_grad():
        mlp_dtype = next(mlp.parameters()).dtype
        inj_flat = mlp(ret_emb_batch.reshape(B * K, D_content).to(mlp_dtype)).reshape(B * K, H).to(model_dtype)

    base_messages = [build_mcq_vqa_message_text(t) for t in batch_tasks]
    student_prompt_texts = build_student_prompt_texts_pubmedqa(tokenizer, base_messages, k_prefix=K)
    enc = tokenizer(
        student_prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    prompt_input_ids = enc["input_ids"]
    prompt_attn_mask = enc["attention_mask"]

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
        with torch.no_grad():
            out_ids = base_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attn_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    prompt_len = prompt_input_ids.shape[1]
    return tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)


# -------------------------------------------------------
# Main evaluation loop
# -------------------------------------------------------

def run_evaluation(args, tasks, train_tasks, train_embeddings, train_qids, tokenizer, model, device,
                   mlp=None, content_emb_map=None, D_content=None,
                   reprag_token_id=None, model_dtype=None, base_model=None):
    task_map = build_train_task_map(train_tasks) if train_tasks else {}
    faiss_index = build_faiss_index(train_embeddings) if train_embeddings is not None else None
    train_task_map = task_map

    loader = DataLoader(
        PubMedQADataset(tasks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tasks,
    )

    results = []

    # --- Efficiency accumulators ---
    total_overhead_tokens = 0
    total_batch_gflops = 0.0
    total_queries = 0

    for batch_idx, batch in enumerate(loader):
        base_messages = [build_mcq_vqa_message(t) for t in batch]

        # Baseline token count for this batch (no retrieval overhead)
        base_tokens = get_token_count_text(tokenizer, base_messages)
        batch_overhead = 0

        # --- Retrieval and generation ---
        if args.retrieval_mode == 'none' or faiss_index is None:
            messages = base_messages
            raw_preds = generate_batch(messages, tokenizer, model, device, args)

        elif args.retrieval_mode == 'incontext':
            if args.retrieval_source == 'combined':
                query_msgs = [build_combined_pool_embedding_message(t) for t in batch]
            elif args.train_data_format == 'pubmedqa' and args.retrieval_source == 'train':
                query_msgs = [build_pubmedqa_embedding_message(t, include_gold=False) for t in batch]
            else:
                query_msgs = [build_mcq_embedding_message(t, include_gold=False) for t in batch]

            nb_qids, _ = retrieve_neighbours(
                query_msgs, tokenizer, model, device,
                faiss_index, train_qids, args.top_k,
            )
            messages = []
            for t, qids in zip(batch, nb_qids):
                retrieved = [task_map[q] for q in qids if q in task_map]
                if args.retrieval_source == 'combined':
                    messages.append(build_combined_incontext_message(t, retrieved))
                elif args.retrieval_source == 'corpus':
                    messages.append(build_corpus_incontext_message(t, retrieved))
                elif args.train_data_format == 'pubmedqa':
                    messages.append(build_pubmedqa_incontext_message(t, retrieved))
                else:
                    messages.append(build_mcq_incontext_message(t, retrieved))

            aug_tokens = get_token_count_text(tokenizer, messages)
            batch_overhead = aug_tokens - base_tokens
            raw_preds = generate_batch(messages, tokenizer, model, device, args, max_input_length=8192)

        elif args.retrieval_mode == 'embed_prefix_no_train':
            query_msgs = [build_mcq_embedding_message(t, include_gold=False) for t in batch]
            nb_qids, _ = retrieve_neighbours(
                query_msgs, tokenizer, model, device,
                faiss_index, train_qids, args.top_k,
            )
            prefix_embeds, prefix_lengths = build_prefix_embeds_no_train(
                nb_qids, train_task_map, tokenizer, model, device,
            )
            # Overhead = total prefix tokens across the batch
            batch_overhead = sum(prefix_lengths)
            raw_preds = generate_batch_with_prefix(
                base_messages, prefix_embeds, prefix_lengths,
                tokenizer, model, device, args,
            )

        elif args.retrieval_mode == 'embed_prefix_mlp':
            query_msgs = [build_pubmedqa_embedding_message(t, include_gold=False) for t in batch]
            nb_qids, _ = retrieve_neighbours(
                query_msgs, tokenizer, model, device,
                faiss_index, train_qids, args.top_k,
            )
            # Overhead = K RepRAG tokens per sample
            batch_overhead = args.top_k * len(batch)
            raw_preds = generate_batch_with_mlp(
                batch_tasks=batch,
                neighbour_qids_batch=nb_qids,
                content_emb_map=content_emb_map,
                tokenizer=tokenizer,
                base_model=base_model,
                mlp=mlp,
                device=device,
                args=args,
                reprag_token_id=reprag_token_id,
                D_content=D_content,
                model_dtype=model_dtype,
            )

        # --- GFLOPs estimate for this batch ---
        est_gen_tokens = args.max_new_tokens * len(batch)
        total_tokens_in_pass = base_tokens + batch_overhead + est_gen_tokens
        batch_gflops = estimate_gflops(model, total_tokens_in_pass)

        total_overhead_tokens += batch_overhead
        total_batch_gflops += batch_gflops
        total_queries += len(batch)

        # --- Record results ---
        for t, raw in zip(batch, raw_preds):
            pred_letter = parse_answer_letter(raw)
            correct = is_correct(raw, t['gold_answer'])
            results.append({
                'qid': t['qid'],
                'dataset': t['dataset'],
                'split': t['split'],
                'question': t['question'],
                'gold_answer': t['gold_answer'],
                'pred_raw': raw.strip()[:200],
                'pred_letter': pred_letter,
                'correct': correct,
                'retrieval_mode': args.retrieval_mode,
            })

        # Per-batch summary
        batch_correct = sum(is_correct(r['pred_raw'], r['gold_answer']) for r in results[-len(batch):])
        batch_acc = batch_correct / len(batch)
        overall_correct = sum(r['correct'] for r in results)
        overall_acc = overall_correct / len(results)
        done = (batch_idx + 1) * args.batch_size
        print(f"  [batch {batch_idx+1} | {min(done, len(tasks))}/{len(tasks)}] "
              f"batch_acc={batch_acc:.2f} ({batch_correct}/{len(batch)})  "
              f"overall_acc={overall_acc:.4f} ({overall_correct}/{len(results)})  "
              f"overhead={batch_overhead} toks  gflops={batch_gflops:.2f}")
        for t, raw in zip(batch, raw_preds):
            print(f"    pred={parse_answer_letter(raw)!r}  gold={t['gold_answer']!r}  raw={raw.strip()[:60]!r}")

        torch.cuda.empty_cache()

    # --- Final efficiency metrics ---
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

    # Load test data
    print(f"[data] Loading benchmark from {args.benchmark_json}")
    all_tasks = load_benchmark(args.benchmark_json, dataset_filter=args.dataset)
    tasks = shard_tasks(all_tasks, args.idx, args.num_parts)
    print(f"[data] Shard {args.idx}/{args.num_parts}: {len(tasks)} test tasks")

    # Load retrieval data
    train_tasks = []
    if args.retrieval_source == 'combined':
        if not getattr(args, 'combined_pool_jsonl', ''):
            raise ValueError("--combined_pool_jsonl required when retrieval_source=combined")
        print(f"[data] Loading combined pool from {args.combined_pool_jsonl}")
        train_tasks = load_combined_pool(args.combined_pool_jsonl)
        print(f"[data] {len(train_tasks)} combined pool items loaded")
    elif args.retrieval_source == 'corpus' and args.chunk_dir:
        print(f"[data] Loading PubMed corpus from {args.chunk_dir}")
        train_tasks = load_pubmed_corpus(args.chunk_dir)
        print(f"[data] {len(train_tasks)} corpus passages loaded")
    elif args.train_json:
        if args.train_data_format == 'pubmedqa':
            print(f"[data] Loading PubMedQA train from {args.train_json}")
            train_tasks = load_pubmedqa_train(args.train_json)
        else:
            print(f"[data] Loading MedQA train from {args.train_json}")
            train_tasks = load_train(args.train_json)
        print(f"[data] {len(train_tasks)} train tasks loaded")

    # Load pre-built embeddings
    train_embeddings, train_qids = None, []
    if args.retrieval_source == 'combined':
        if args.retrieval_mode == 'embed_prefix_mlp':
            emb_path = getattr(args, 'corpus_embeddings_path', '')
        else:
            emb_path = getattr(args, 'combined_embeddings_path', '')
    elif args.retrieval_source == 'corpus':
        emb_path = args.corpus_embeddings_path
    else:
        emb_path = args.train_embeddings_path
    if emb_path:
        print(f"[data] Loading embeddings from {emb_path}")
        ckpt = torch.load(emb_path, map_location='cpu')
        train_embeddings = ckpt['embeddings'].float().numpy()
        train_qids = ckpt['qids']
        print(f"[data] Embeddings shape: {train_embeddings.shape}")

    # Load model
    tokenizer, model, device = load_model(args)
    base_model = model.module if hasattr(model, 'module') else model
    model_dtype = next(base_model.parameters()).dtype

    # Setup MLP for embed_prefix_mlp
    mlp = content_emb_map = D_content = reprag_token_id = None
    if args.retrieval_mode == 'embed_prefix_mlp':
        if not args.mlp_ckpt:
            raise ValueError("--mlp_ckpt required for embed_prefix_mlp mode")
        if not args.content_embeddings_path:
            raise ValueError("--content_embeddings_path required for embed_prefix_mlp mode")

        num_added = tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
        if num_added > 0:
            base_model.resize_token_embeddings(len(tokenizer))
        reprag_token_id = tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)

        cont_ckpt = torch.load(args.content_embeddings_path, map_location='cpu')
        cont_emb = cont_ckpt['embeddings'].float().numpy()
        cont_qids = [str(q) for q in cont_ckpt['qids']]
        content_emb_map = {qid: cont_emb[i] for i, qid in enumerate(cont_qids)}
        print(f"[MLP] Loaded {len(content_emb_map)} content embeddings.")

        mlp, D_content = load_mlp(args.mlp_ckpt, device, base_model)

    # Run evaluation
    results, efficiency_metrics = run_evaluation(
        args, tasks, train_tasks, train_embeddings, train_qids,
        tokenizer, model, device,
        mlp=mlp, content_emb_map=content_emb_map, D_content=D_content,
        reprag_token_id=reprag_token_id, model_dtype=model_dtype, base_model=base_model,
    )

    # Save outputs
    csv_path = write_predictions_csv(results, args.out_dir)
    print(f"[saved] Predictions: {csv_path}")
    write_final_results(results, args.out_dir, retrieval_mode=args.retrieval_mode)
    print(f"[saved] Results: {args.out_dir}/final_results.txt")

    # Save efficiency report
    record_efficiency_to_txt(args.out_dir, efficiency_metrics, args)


if __name__ == '__main__':
    main()