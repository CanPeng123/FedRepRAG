"""
KILT-NQ VQA without retrieval (Qwen3) — v2.

Adds per-query efficiency tracking on top of the original script:
  - avg input tokens/query
  - avg GFLOPs/query
  - efficiency_report.txt written alongside final_results.txt

This serves as the retrieval-free baseline for comparison with v2 retrieval scripts.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader

from arg_KILT_NQ.KILT_NQ_arg import parse_args
from utilise.KILT_NQ.KILT_NQ_data_process import load_kilt_nq, shard_records, KILTNQDataset, collate_records
from utilise.KILT_NQ.KILT_NQ_prompt_message import build_kilt_nq_message
from utilise.KILT_NQ.KILT_NQ_post_process import (
    extract_short_answer, exact_match, token_f1,
    write_predictions_csv, write_final_results, save_arguments,
)
from utilise.eval_utils import estimate_gflops, record_efficiency_to_txt


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

def get_token_count_batch(tokenizer, messages_list: list, max_length: int = 2048) -> int:
    """Count total attended tokens for a batch of message-lists."""
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
# Generation
# -------------------------------------------------------

def generate_batch(messages_list, tokenizer, model, device, args, max_input_length=2048):
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
# Evaluation loop
# -------------------------------------------------------

def run_evaluation(args, records, tokenizer, model, device):
    loader = DataLoader(
        KILTNQDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    results = []
    total_em = total_f1 = 0.0

    # Efficiency accumulators
    total_batch_gflops = 0.0
    total_queries = 0

    for batch_idx, batch in enumerate(loader):
        messages_list = [[build_kilt_nq_message(r)] for r in batch]

        batch_tokens = get_token_count_batch(tokenizer, messages_list)
        est_gen_tokens = args.max_new_tokens * len(batch)
        total_tokens_in_pass = batch_tokens + est_gen_tokens
        batch_gflops = estimate_gflops(model, total_tokens_in_pass)

        total_batch_gflops += batch_gflops
        total_queries += len(batch)

        raw_preds = generate_batch(messages_list, tokenizer, model, device, args)

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
              f"tokens={batch_tokens}  gflops={batch_gflops:.2f}")
        for rec, raw in zip(batch, raw_preds):
            print(f"    Q: {rec['question'][:80]!r}")
            print(f"    pred: {extract_short_answer(raw)!r}  gold: {rec['gold_answers'][:2]!r}")

    efficiency_metrics = {
        "avg_overhead": 0.0,  # no retrieval context — overhead is always zero
        "avg_gflops":   total_batch_gflops / total_queries if total_queries > 0 else 0.0,
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

    print(f"[data] Loading KILT-NQ from {args.data_jsonl}")
    records = load_kilt_nq(args.data_jsonl)
    records = shard_records(records, args.idx, args.num_parts)
    print(f"[data] Shard {args.idx}/{args.num_parts}: {len(records)} records")

    tokenizer, model, device = load_model(args)

    results, efficiency_metrics = run_evaluation(args, records, tokenizer, model, device)

    csv_path = write_predictions_csv(results, args.out_dir)
    print(f"[saved] Predictions: {csv_path}")
    write_final_results(results, args.out_dir)
    print(f"[saved] Results: {args.out_dir}/final_results.txt")

    record_efficiency_to_txt(args.out_dir, efficiency_metrics, args)


if __name__ == '__main__':
    main()