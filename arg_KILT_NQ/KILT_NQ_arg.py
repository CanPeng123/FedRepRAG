import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KILT-NQ Open-Domain QA (Qwen3)")

    # -------- Model --------
    parser.add_argument('--model', type=str, required=True,
                        help="Qwen3 checkpoint: shorthand (qwen3:8b) or HF ID or local path.")
    parser.add_argument('--use_quantization', action='store_true',
                        help="Use 4-bit quantization via bitsandbytes.")
    parser.add_argument('--use_deepspeed', action='store_true',
                        help="Use DeepSpeed for distributed inference.")
    parser.add_argument('--local_rank', type=int, default=-1)

    # -------- Parallel sharding --------
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--num_parts', type=int, default=1)

    # -------- Inference / Decoding --------
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=32,
                        help="Max new tokens per answer (NQ answers are typically short phrases).")
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=None)

    # -------- Dataset --------
    parser.add_argument('--data_jsonl', type=str, required=True,
                        help="Path to KILT-NQ JSONL file (train / validation / test).")
    parser.add_argument('--split', type=str, default='validation',
                        help="Split label stored in outputs (train / validation / test).")

    # -------- Retrieval --------
    parser.add_argument('--retrieval_mode', type=str, default='none',
                        choices=['none', 'incontext', 'embed_prefix_w_train'],
                        help="Retrieval strategy.")
    parser.add_argument('--train_jsonl', type=str, default='',
                        help="Path to KILT-NQ train JSONL (retrieval corpus).")
    parser.add_argument('--train_embeddings_path', type=str, default='',
                        help="PT file with {embeddings, qids} for the train split.")
    parser.add_argument('--content_embeddings_path', type=str, default='',
                        help="PT file with {embeddings, qids} for content (Q+A) embeddings (required for embed_prefix_w_train).")
    parser.add_argument('--retrieval_mlp_ckpt', "--checkpoint", type=str, default='',
                        help="Path to trained PrefixProjector checkpoint (required for embed_prefix_w_train).")
    parser.add_argument('--top_k', type=int, default=3,
                        help="Number of retrieved neighbours.")

    # -------- Output --------
    parser.add_argument('--out_dir', type=str, default='KILT_NQ_preds',
                        help="Output directory for predictions and metrics.")

    args = parser.parse_args()

    if args.num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if not (0 <= args.idx < args.num_parts):
        raise ValueError("--idx must satisfy 0 <= idx < num_parts")

    return args