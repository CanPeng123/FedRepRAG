"""
Extract FAISS-ready embeddings from KILT-NQ training data using Qwen3.

Usage (see bash/VQA/w_embedding_retrival/KILT_NQ/):
    python Qwen3_Batch_KILT_NQ_VQA_get_retrival_embedding.py \
        --model qwen3:8b \
        --data_jsonl /.../datasets/KILT_NQ/train.jsonl \
        --split train \
        --store_embeddings \
        --batch_size 8 \
        --out_dir /.../output/KILT_NQ/train_embeddings/Qwen3_8b_train
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader

from arg_KILT_NQ.KILT_NQ_retrieval_arg import parse_args
from utilise.KILT_NQ.KILT_NQ_data_process import load_kilt_nq, shard_records, KILTNQDataset, collate_records
from utilise.KILT_NQ.KILT_NQ_prompt_message import build_kilt_nq_embedding_message, build_kilt_nq_content_message
from utilise.KILT_NQ.KILT_NQ_post_process import save_arguments


MODEL_MAP = {
    'qwen3:8b':  'Qwen/Qwen3-8B',
    'qwen3:14b': 'Qwen/Qwen3-14B',
    'qwen3:32b': 'Qwen/Qwen3-32B',
}


def resolve_model_name(key: str) -> str:
    return MODEL_MAP.get(key, key)


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
# Embedding extraction
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
        hidden = outputs.hidden_states[-1]                               # [B, T, D]
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)      # [B, D]
        emb = emb / emb.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)

    return emb.cpu().float().numpy()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    save_arguments(args, args.out_dir)

    print(f"[data] Loading KILT-NQ from {args.data_jsonl}")
    all_records = load_kilt_nq(args.data_jsonl)
    records = shard_records(all_records, args.idx, args.num_parts)
    print(f"[data] Shard {args.idx}/{args.num_parts}: {len(records)} records")

    tokenizer, model, device = load_model(args)

    if not args.store_embeddings:
        print("[info] --store_embeddings not set; nothing to do. Exiting.")
        return

    loader = DataLoader(
        KILTNQDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    do_question = True
    do_content  = (args.embed_mode == 'both')

    q_embeddings, c_embeddings = [], []
    all_ids, all_meta = [], []

    for batch_idx, batch in enumerate(loader):
        # Question-only embeddings (FAISS retrieval index)
        q_msgs = [build_kilt_nq_embedding_message(r, include_gold=args.embeddings_w_gold) for r in batch]
        q_embeddings.append(embed_batch(q_msgs, tokenizer, model, device, max_length=512))

        # Content embeddings (MLP injection input)
        if do_content:
            c_msgs = [build_kilt_nq_content_message(r) for r in batch]
            c_embeddings.append(embed_batch(c_msgs, tokenizer, model, device, max_length=512))

        for r in batch:
            all_ids.append(r['id'])
            all_meta.append({
                'id':           r['id'],
                'split':        args.split,
                'question':     r['question'],
                'gold_answers': r['gold_answers'],
            })

        if (batch_idx + 1) % 50 == 0:
            done = min((batch_idx + 1) * args.batch_size, len(records))
            print(f"  embedded {done}/{len(records)}")

    suffix = f"_part{args.idx}" if args.num_parts > 1 else ""

    # Save question embeddings
    q_arr = np.concatenate(q_embeddings, axis=0)
    print(f"[embed] Question embeddings shape: {q_arr.shape}")
    q_path = os.path.join(args.out_dir, f"kilt_nq_embeddings_{args.split}{suffix}.pt")
    torch.save({'embeddings': torch.tensor(q_arr), 'qids': all_ids, 'meta': all_meta}, q_path)
    print(f"[saved] Question embeddings → {q_path}")

    # Save content embeddings
    if do_content:
        c_arr = np.concatenate(c_embeddings, axis=0)
        print(f"[embed] Content embeddings shape: {c_arr.shape}")
        c_path = os.path.join(args.out_dir, f"kilt_nq_content_embeddings_{args.split}{suffix}.pt")
        torch.save({'embeddings': torch.tensor(c_arr), 'qids': all_ids, 'meta': all_meta}, c_path)
        print(f"[saved] Content embeddings → {c_path}")


if __name__ == '__main__':
    main()