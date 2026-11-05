"""
Extract FAISS-ready embeddings from PubMedQA training data using Qwen3.

Usage (see bash/VQA/w_embedding_retrival/PubMedQA/):
    python Qwen3_Batch_PubMedQA_VQA_get_retrival_embedding.py \
        --model qwen3:7b \
        --train_json /.../datasets/PubMedQA/train.json \
        --store_embeddings --embeddings_w_gold \
        --batch_size 8 \
        --out_dir /.../output/PubMedQA/train_embeddings
"""

import os
import sys
import torch
import numpy as np

# Allow imports from src/VQA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader

from arg_PubMedQA.PubMedQA_retrieval_arg import parse_args
from utilise.PubMedQA.PubMedQA_data_process import (
    load_train, load_pubmedqa_train, load_pubmedqa_fl_train,
    load_pubmed_corpus, load_combined_pool,
    shard_tasks, PubMedQADataset, collate_tasks,
)
from utilise.PubMedQA.PubMedQA_prompt_message import (
    build_mcq_embedding_message,
    build_mcq_vqa_message,
    build_pubmedqa_embedding_message,
    build_pubmed_corpus_embedding_message,
    build_pubmed_corpus_content_message,
    build_combined_pool_embedding_message,
)
from utilise.PubMedQA.PubMedQA_post_process import save_arguments, parse_answer_letter


# -------------------------------------------------------
# Model name resolution
# -------------------------------------------------------

MODEL_MAP = {
    'qwen3:8b':  'Qwen/Qwen3-8B',
    'qwen3:14b': 'Qwen/Qwen3-14B',
    'qwen3:32b': 'Qwen/Qwen3-32B',
}


def resolve_model_name(model_key: str) -> str:
    """Resolve shorthand to HF ID, or pass through local/full HF paths unchanged."""
    return MODEL_MAP.get(model_key, model_key)


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
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
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
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    return tokenizer, model, device


# -------------------------------------------------------
# Embedding extraction
# -------------------------------------------------------

def embed_batch(messages, tokenizer, model, device, max_length=2048):
    """
    Encode a list of chat messages and return L2-normalised pooled embeddings.

    Args:
        messages:   list of {"role": "user", "content": str} dicts
        max_length: tokenizer truncation limit (default 2048 to cover full
                    PubMed abstracts; use 512 for question-only embeddings)
    Returns:
        np.ndarray of shape [B, D], float32, L2-normalised
    """
    texts = []
    for msg in messages:
        text = tokenizer.apply_chat_template(
            [msg],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]           # [B, T, D]
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)          # [B, D]
        emb = summed / mask.sum(dim=1).clamp(min=1e-6)
        norm = emb.norm(p=2, dim=1, keepdim=True)
        emb = emb / norm.clamp(min=1e-6)

    return emb.cpu().float().numpy()


# -------------------------------------------------------
# Optional debug generation
# -------------------------------------------------------

def generate_batch(messages, tokenizer, model, device, args):
    texts = []
    for msg in messages:
        text = tokenizer.apply_chat_template(
            [msg],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs['input_ids'].shape[1]
    decoded = tokenizer.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)
    return decoded


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    save_arguments(args, args.out_dir)

    # Load data
    if args.data_format == 'medqa':
        print(f"[data] Loading MedQA train data from {args.train_json}")
        all_tasks = load_train(args.train_json)
        tasks = shard_tasks(all_tasks, args.idx, args.num_parts)
    elif args.data_format == 'pubmedqa':
        print(f"[data] Loading PubMedQA labeled train from {args.train_json}")
        all_tasks = load_pubmedqa_train(args.train_json)
        tasks = shard_tasks(all_tasks, args.idx, args.num_parts)
    elif args.data_format == 'pqa_fl':
        print(f"[data] Loading PubMedQA FL client train from {args.train_json}")
        all_tasks = load_pubmedqa_fl_train(args.train_json)
        tasks = shard_tasks(all_tasks, args.idx, args.num_parts)
    elif args.data_format == 'pubmed_corpus':
        print(f"[data] Loading PubMed corpus chunks from {args.chunk_dir}")
        if args.pubmedqa_train_json:
            print(f"[data] Using PubMedQA questions from {args.pubmedqa_train_json} for embeddings (test PMIDs excluded)")
        tasks = load_pubmed_corpus(args.chunk_dir, args.idx, args.num_parts, args.pubmedqa_train_json)
    elif args.data_format == 'pqa_combined':
        if not args.combined_pool_jsonl:
            raise ValueError("--combined_pool_jsonl required for data_format=pqa_combined")
        print(f"[data] Loading combined pool from {args.combined_pool_jsonl}")
        tasks = load_combined_pool(args.combined_pool_jsonl, args.idx, args.num_parts)
    else:
        raise ValueError(f"Unknown data_format: {args.data_format}")
    print(f"[data] Shard {args.idx}/{args.num_parts}: {len(tasks)} tasks")

    # Load model
    tokenizer, model, device = load_model(args)

    if not args.store_embeddings:
        print("[info] --store_embeddings not set; nothing to do. Exiting.")
        return

    loader = DataLoader(
        PubMedQADataset(tasks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tasks,
    )

    embed_mode = getattr(args, 'embed_mode', 'question')
    # pubmed_corpus, pqa_combined, and pqa_fl support 'both' / 'content' / 'title' modes;
    # all other formats only support 'question'.
    if args.data_format not in ('pubmed_corpus', 'pqa_combined', 'pqa_fl'):
        embed_mode = 'question'

    do_question = embed_mode in ('question', 'both')
    do_content  = embed_mode in ('content',  'both')
    do_title    = embed_mode == 'title'

    q_embeddings, c_embeddings, t_embeddings = [], [], []
    all_qids, all_meta = [], []

    for batch_idx, batch in enumerate(loader):
        # ---- question / retrieval embeddings ----
        if do_question:
            if args.data_format == 'pubmedqa':
                q_msgs = [build_pubmedqa_embedding_message(t, include_gold=args.embeddings_w_gold) for t in batch]
            elif args.data_format == 'pubmed_corpus':
                q_msgs = [build_pubmed_corpus_embedding_message(t) for t in batch]
            elif args.data_format in ('pqa_combined', 'pqa_fl'):
                # question mode: question text only (for FAISS retrieval index)
                q_msgs = [build_combined_pool_embedding_message(t, include_content=False) for t in batch]
            else:
                q_msgs = [build_mcq_embedding_message(t, include_gold=args.embeddings_w_gold) for t in batch]
            q_embeddings.append(embed_batch(q_msgs, tokenizer, model, device, max_length=512))

        # ---- title-only embeddings ----
        if do_title:
            t_msgs = [
                {"role": "user", "content": t.get('title', '').strip() or t.get('question', '.')}
                for t in batch
            ]
            t_embeddings.append(embed_batch(t_msgs, tokenizer, model, device, max_length=512))

        # ---- content / MLP injection embeddings ----
        if do_content:
            if args.data_format in ('pqa_combined', 'pqa_fl'):
                # content mode: title + content only (corpus body, no question)
                c_msgs = [
                    {"role": "user", "content": (
                        f"{t.get('title', '').strip()}\n{t.get('content', '').strip()}".strip()
                        or t.get('question', '.')
                    )}
                    for t in batch
                ]
            else:
                c_msgs = [build_pubmed_corpus_content_message(t) for t in batch]
            c_embeddings.append(embed_batch(c_msgs, tokenizer, model, device, max_length=2048))

        for t in batch:
            all_qids.append(t['qid'])
            if args.data_format == 'pubmed_corpus':
                all_meta.append({
                    'qid':   t['qid'],
                    'dataset': t['dataset'],
                    'split': t['split'],
                    'title': t.get('title', ''),
                    'pmid':  t.get('pmid', ''),
                })
            elif args.data_format in ('pqa_combined', 'pqa_fl'):
                all_meta.append({
                    'qid':            t['qid'],
                    'pmid':           t.get('pmid', ''),
                    'dataset':        t['dataset'],
                    'split':          t['split'],
                    'source':         t.get('source', ''),
                    'final_decision': t.get('final_decision', ''),
                })
            else:
                all_meta.append({
                    'qid':        t['qid'],
                    'dataset':    t['dataset'],
                    'split':      t['split'],
                    'question':   t['question'],
                    'gold_answer': t.get('gold_answer', ''),
                })

        if (batch_idx + 1) % 50 == 0:
            done = (batch_idx + 1) * args.batch_size
            print(f"  embedded {min(done, len(tasks))}/{len(tasks)}")

        if args.debug_generate and batch_idx == 0:
            vqa_msgs = [build_mcq_vqa_message(t) for t in batch]
            preds = generate_batch(vqa_msgs, tokenizer, model, device, args)
            for t, p in zip(batch, preds):
                print(f"  [debug] qid={t['qid']} gold={t['gold_answer']} pred={parse_answer_letter(p)!r}")

    suffix = f"_part{args.idx}" if args.num_parts > 1 else ""

    if do_question:
        q_arr = np.concatenate(q_embeddings, axis=0)
        print(f"[embed] Question embeddings shape: {q_arr.shape}")
        q_path = args.question_output or os.path.join(args.out_dir, f"pubmedqa_embeddings_{args.split}{suffix}.pt")
        os.makedirs(os.path.dirname(os.path.abspath(q_path)), exist_ok=True)
        torch.save({'embeddings': torch.tensor(q_arr), 'qids': all_qids, 'meta': all_meta}, q_path)
        print(f"[saved] Question embeddings → {q_path}")

    if do_content:
        c_arr = np.concatenate(c_embeddings, axis=0)
        print(f"[embed] Content embeddings shape: {c_arr.shape}")
        c_path = args.content_output or os.path.join(args.out_dir, f"pubmedqa_content_embeddings_{args.split}{suffix}.pt")
        os.makedirs(os.path.dirname(os.path.abspath(c_path)), exist_ok=True)
        torch.save({'embeddings': torch.tensor(c_arr), 'qids': all_qids, 'meta': all_meta}, c_path)
        print(f"[saved] Content embeddings   → {c_path}")

    if do_title:
        t_arr = np.concatenate(t_embeddings, axis=0)
        print(f"[embed] Title embeddings shape: {t_arr.shape}")
        t_path = os.path.join(args.out_dir, f"pubmedqa_title_embeddings_{args.split}{suffix}.pt")
        torch.save({'embeddings': torch.tensor(t_arr), 'qids': all_qids, 'meta': all_meta}, t_path)
        print(f"[saved] Title embeddings     → {t_path}")


if __name__ == '__main__':
    main()