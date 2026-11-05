import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SLAKE VQA with Retrieval (RAG)")

    # -------- Model --------
    parser.add_argument(
        '--model', type=str, required=True,
        choices=[
            'qwen2_5-vl-instruct:2b',
            'qwen2_5-vl-instruct:3b',
            'qwen2_5-vl-instruct:7b',
            'qwen2_5-vl-instruct:72b',
            'gemma3:4b',
            'gemma3:12b',
        ],
        help="Which backbone model to use."
    )
    parser.add_argument('--use_quantization', action='store_true', help="Use 4-bit quantization.")
    parser.add_argument('--use_deepspeed', action='store_true', help="Use DeepSpeed for distributed inference.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DeepSpeed internal use (auto-injected).")

    # -------- Parallel sharding --------
    parser.add_argument('--idx', type=int, default=0, help='Which shard to run (0-based).')
    parser.add_argument('--num_parts', type=int, default=1, help='Total number of shards.')

    # -------- Inference / Decoding --------
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for VLM inference.")
    parser.add_argument('--max_new_tokens', type=int, default=16, help="Max new tokens to generate per answer.")
    parser.add_argument('--num_beams', type=int, default=1, help="Beam search width (1 = greedy).")

    parser.add_argument('--do_sample', action='store_true', help="Use sampling during generation.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature (only if --do_sample).")
    parser.add_argument('--top_p', type=float, default=None, help="Nucleus sampling top_p (only if --do_sample).")

    # -------- Dataset --------
    parser.add_argument('--problems_json', type=str, required=True, help="Path to SLAKE json file (train/val/test).")
    parser.add_argument('--images_root', type=str, default="", help="Optional root to prepend to image paths in SLAKE json.")
    parser.add_argument('--split', type=str, default=None, help="Optional split name tag to store in outputs (e.g., train/val/test).")
    parser.add_argument('--lang', type=str, default="en", choices=["en", "zh"], help="Filter question language (en or zh).")
    parser.add_argument("--train_json", type=str, default="", help="SLAKE train json for retrieval (required if retrieval_mode != none).")

    # -------- retrieval --------
    parser.add_argument("--retrieval_mode", type=str, default="none",choices=["none", "incontext", "embed_prefix_no_train", "embed_prefix_w_train"])
    parser.add_argument("--train_embeddings_path", type=str, default="", help="PT file with {'embeddings': [N,D], 'qids': [...]} for TRAIN split.")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--retrieval_mlp_ckpt", "--checkpoint", type=str, default="", help="Optional MLP ckpt for embed_prefix_w_train.")
    parser.add_argument("--inj_mult", type=float, default=0.1)
    parser.add_argument("--embedding_output_mode", type=str, default=None, choices=["weighted_avg", "concat"], help="Retrieval embedding aggregate strategy: weighted_avg or concat.")

    # -------- retrieval visual for incontext --------
    parser.add_argument("--retrieved_visual", type=str, default="none", choices=["none", "image", "caption", "image_n_caption"])
    parser.add_argument("--retrieved_visual_w_image", action="store_true", help="(optional) alias for retrieved_visual=image; keep if you want.")
    parser.add_argument("--retrieved_caption_json", type=str, default="", help="Optional JSON file mapping qid->caption (or list of {qid, caption}). Used when retrieved_visual includes caption.")

    # -------- 
    parser.add_argument("--client_embeddings_dir", type=str, default=None, help="Directory containing per-client JSON files with qid lists, used to map qid -> client.")

    # -------- Output --------
    parser.add_argument('--out_dir', type=str, default="VQA_preds", help="Output directory for predictions CSV/logs.")

    args = parser.parse_args()

    # Basic sanity checks
    if args.num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if not (0 <= args.idx < args.num_parts):
        raise ValueError("--idx must satisfy 0 <= idx < num_parts")

    return args