import argparse


def parse_args():
    parser = argparse.ArgumentParser("Extract PubMedQA train embeddings for FAISS retrieval (Qwen3).")

    # -------- Model --------
    parser.add_argument('--model', type=str, required=True,
                        help="Qwen3 checkpoint: shorthand (qwen3:7b) or HuggingFace ID or local path.")
    parser.add_argument('--use_quantization', action='store_true')
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)

    # -------- Dataset --------
    parser.add_argument('--data_format', type=str, default='medqa',
                        choices=['medqa', 'pubmedqa', 'pubmed_corpus', 'pqa_combined', 'pqa_fl'],
                        help="Input data format: medqa (MCQ train JSON), "
                             "pubmedqa (labeled_train_set.json), "
                             "pubmed_corpus (JSONL chunk dir), "
                             "pqa_combined (pool_combined.jsonl — labeled train + unlabeled), "
                             "pqa_fl (FL client train JSON from create_pubmedqa_FL_split.py).")
    parser.add_argument('--train_json', type=str, default=None,
                        help="Path to train JSON (medqa or pubmedqa format).")
    parser.add_argument('--chunk_dir', type=str, default=None,
                        help="Path to PubMed corpus chunk dir (pubmed_corpus format).")
    parser.add_argument('--combined_pool_jsonl', type=str, default=None,
                        help="Path to pool_combined.jsonl (pqa_combined format).")
    parser.add_argument('--pubmedqa_train_json', type=str, default=None,
                        help="Path to labeled_train_set.json. When set for pubmed_corpus, "
                             "each article is embedded using its PubMedQA question instead of "
                             "passage text, aligning index and query spaces. Test PMIDs are excluded.")
    parser.add_argument('--embed_mode', type=str, default='question', choices=['question', 'content', 'title', 'both'],
                         help="Embedding target. For pqa_combined and pubmed_corpus: "
                         "'question' embeds question text for retrieval, "
                         "'content' embeds title+abstract/content for MLP injection, "
                         "'title' embeds title only, "
                         "'both' generates both outputs in one run. "
                         "For pubmedqa and medqa, only question mode is used.")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--question_output', help='Optional destination file for question embeddings.')
    parser.add_argument('--content_output', help='Optional destination file for content embeddings.')
    parser.add_argument('--out_dir', type=str, default='PubMedQA_embeddings')

    # -------- Batching / sharding --------
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--num_parts', type=int, default=1)

    # -------- Embedding options --------
    parser.add_argument('--store_embeddings', action='store_true',
                        help="Compute and save embeddings to disk.")
    parser.add_argument('--embeddings_w_gold', action='store_true',
                        help="Append gold answer letter into prompt when encoding training examples.")

    # -------- Optional debug generation --------
    parser.add_argument('--debug_generate', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=16)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)

    return parser.parse_args()