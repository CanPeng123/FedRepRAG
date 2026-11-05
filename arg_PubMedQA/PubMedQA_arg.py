import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PubMedQA Medical QA with FAISS Retrieval (Qwen3)")

    # -------- Model --------
    parser.add_argument(
        '--model', type=str, required=True,
        help="Qwen3 checkpoint: shorthand (qwen3:7b) or HuggingFace ID or local path."
    )
    parser.add_argument('--use_quantization', action='store_true', help="Use 4-bit quantization via bitsandbytes.")
    parser.add_argument('--use_deepspeed', action='store_true', help="Use DeepSpeed for distributed inference.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DeepSpeed internal (auto-injected).")

    # -------- Parallel sharding --------
    parser.add_argument('--idx', type=int, default=0, help='Which shard to run (0-based).')
    parser.add_argument('--num_parts', type=int, default=1, help='Total number of shards.')

    # -------- Inference / Decoding --------
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference.")
    parser.add_argument('--max_new_tokens', type=int, default=16, help="Max new tokens to generate per answer.")
    parser.add_argument('--num_beams', type=int, default=1, help="Beam search width (1 = greedy).")
    parser.add_argument('--do_sample', action='store_true', help="Use sampling during generation.")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=None)

    # -------- Dataset --------
    parser.add_argument('--benchmark_json', type=str, required=True,
                        help="Path to benchmark.json containing the PubMedQA test set.")
    parser.add_argument('--train_json', type=str, default='',
                        help="Path to PubMedQA train.json (MedQA train, used as retrieval corpus).")
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu'],
                        help="Which sub-dataset(s) to evaluate.")
    parser.add_argument('--split', type=str, default='test', help="Split tag stored in outputs.")

    # -------- Retrieval --------
    parser.add_argument('--retrieval_mode', type=str, default='none',
                        choices=['none', 'incontext', 'embed_prefix_no_train', 'embed_prefix_mlp'],
                        help="Retrieval strategy.")
    parser.add_argument('--retrieval_source', type=str, default='train',
                        choices=['train', 'corpus', 'combined'],
                        help="Source of retrieved items: 'train' (QA pairs), 'corpus' (passages), "
                             "or 'combined' (pqa_labeled train + pqa_unlabeled).")
    parser.add_argument('--train_data_format', type=str, default='medqa',
                        choices=['medqa', 'pubmedqa'],
                        help="Format of --train_json: 'medqa' (MCQ list) or 'pubmedqa' (labeled_train_set.json).")
    parser.add_argument('--train_embeddings_path', type=str, default='',
                        help="PT file with {embeddings: [N,D], qids: [...]} for train split.")
    parser.add_argument('--corpus_embeddings_path', type=str, default='',
                        help="PT file with {embeddings: [N,D], qids: [...]} for PubMed corpus.")
    parser.add_argument('--chunk_dir', type=str, default='',
                        help="Path to PubMed corpus JSONL chunk dir (used when retrieval_source=corpus).")
    parser.add_argument('--combined_pool_jsonl', type=str, default='',
                        help="Path to pool_combined.jsonl (used when retrieval_source=combined).")
    parser.add_argument('--combined_embeddings_path', type=str, default='',
                        help="PT file with pre-built embeddings for the combined pool.")
    parser.add_argument('--top_k', type=int, default=3, help="Number of retrieved neighbours.")

    # -------- MLP inference (embed_prefix_mlp mode) --------
    parser.add_argument('--mlp_ckpt', "--checkpoint", type=str, default='',
                        help="Path to trained MLP checkpoint (.pt with mlp_state key).")
    parser.add_argument('--content_embeddings_path', type=str, default='',
                        help="PT file: pubmedqa_content_embeddings_pool.pt — content embeddings "
                             "for the corpus pool (MLP input).")
    parser.add_argument('--corpus_jsonl', type=str, default='',
                        help="pool_combined.jsonl — corpus text (not used for generation but "
                             "available for reference).")

    # -------- Output --------
    parser.add_argument('--out_dir', type=str, default='PubMedQA_preds',
                        help="Output directory for predictions CSV and logs.")

    args = parser.parse_args()

    if args.num_parts < 1:
        raise ValueError("--num_parts must be >= 1")
    if not (0 <= args.idx < args.num_parts):
        raise ValueError("--idx must satisfy 0 <= idx < num_parts")
    if args.retrieval_mode != 'none':
        if args.retrieval_source == 'train' and not args.train_embeddings_path:
            raise ValueError("--train_embeddings_path required when retrieval_source=train")
        if args.retrieval_source == 'corpus' and not args.corpus_embeddings_path:
            raise ValueError("--corpus_embeddings_path required when retrieval_source=corpus")
        if args.retrieval_source == 'combined' and not args.combined_embeddings_path:
            # embed_prefix_mlp uses --corpus_embeddings_path for FAISS instead
            if args.retrieval_mode != 'embed_prefix_mlp' or not args.corpus_embeddings_path:
                raise ValueError("--combined_embeddings_path required when retrieval_source=combined")

    return args