import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KILT-NQ: extract retrieval embeddings (Qwen3)")

    # -------- Model --------
    parser.add_argument('--model', type=str, required=True,
                        help="Qwen3 checkpoint: shorthand (qwen3:8b) or HF ID or local path.")
    parser.add_argument('--use_quantization', action='store_true')
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)

    # -------- Parallel sharding --------
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--num_parts', type=int, default=1)

    # -------- Dataset --------
    parser.add_argument('--data_jsonl', type=str, required=True,
                        help="Path to KILT-NQ JSONL (train / validation).")
    parser.add_argument('--split', type=str, default='train',
                        help="Split label stored in outputs.")
    parser.add_argument('--embeddings_w_gold', action='store_true',
                        help="Include gold answer in the embedding message.")

    # -------- Inference --------
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)

    # -------- Embedding storage --------
    parser.add_argument('--store_embeddings', action='store_true',
                        help="Save embeddings to --out_dir.")
    parser.add_argument('--embed_mode', type=str, default='question',
                        choices=['question', 'both'],
                        help="'question' = question-only embeddings (FAISS index); "
                             "'both' = also save answer/content embeddings (MLP input).")

    # -------- Output --------
    parser.add_argument('--out_dir', type=str, default='KILT_NQ_embeddings')

    args = parser.parse_args()

    if args.num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if not (0 <= args.idx < args.num_parts):
        raise ValueError("--idx must satisfy 0 <= idx < num_parts")

    return args