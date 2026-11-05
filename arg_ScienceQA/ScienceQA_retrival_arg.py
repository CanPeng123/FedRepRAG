import argparse


def parse_args(*, embedding_only=False):
    parser = argparse.ArgumentParser(description="Generate visual descriptions for images using Qwen2.5-VL-Instruct.")
    parser.add_argument('--model', type=str, required=True, choices=[
        'qwen2_5-vl-instruct:3b',
        'qwen2_5-vl-instruct:7b',
        'qwen2_5-vl-instruct:72b',
        'gemma3:12b',
    ], help="Backbone model: Qwen2.5-VL (3b/7b/72b) or Gemma 3 (12b).")
    parser.add_argument('--use_quantization', action='store_true', help="Use 4-bit quantization.")
    parser.add_argument('--use_deepspeed', action='store_true', help="Use DeepSpeed for distributed training.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DeepSpeed internal use (auto-injected).")

    parser.add_argument('--idx', type=int, default=0, help='Which part to run (0-based)')
    parser.add_argument('--num_parts', type=int, default=1, help='Total number of parallel parts')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--do_sample', action='store_true', help="During VLM inference, do samle.")
    parser.add_argument('--temperature', type=float, default=0.7, help='During VLM inference, temperature value for control output')

    parser.add_argument('--problems_json', type=str, help="Path to ScienceQA problems.json")
    parser.add_argument('--images_root', type=str, default="", help="Optional root to prepend to relative image paths in problems.json")
    parser.add_argument('--split', type=str, default=None, help="Filter split: train/val/test")

    parser.add_argument("--train_json", type=str, required=not embedding_only)
    parser.add_argument("--val_json", type=str, required=not embedding_only)
    parser.add_argument("--test_json", type=str, required=not embedding_only)
    parser.add_argument("--eval_on", type=str, default="test", choices=["train", "val", "test"])

    parser.add_argument('--out_dir', type=str, default="VQA_preds")

    # --- get retrival embeddings from training data ---
    parser.add_argument("--store_embeddings", action="store_true", help="If set, extract and save embeddings for each ScienceQA example.")
    parser.add_argument("--embeddings_w_context", action="store_true", 
        help="If w context set, extract and save embeddings with context information for each ScienceQA example.")
    parser.add_argument("--embeddings_w_lecture", action="store_true", 
        help="If w lecture set, extract and save embeddings with lecture information for each ScienceQA example.")
    parser.add_argument("--embeddings_w_solution", action="store_true", 
        help="If w solution set, extract and save embeddings with solution information for each ScienceQA example.")

    parser.add_argument("--train_embeddings_path", type=str, default=None,
    help="Path to training embeddings .pt file (with keys 'embeddings' and 'qids'). Required for retrieval modes.")

    parser.add_argument("--top_k", type=int, default=5, help="Top-K neighbors to retrieve from training embeddings.")
    
    # --- retrival ---
    parser.add_argument("--retrieval_mode", type=str, default="none", choices=["none", "incontext", "embed_prefix_no_train", "embed_prefix_w_train"],
    help="Retrieval strategy: 'none' (baseline), incontext' (prompt with retrieved examples), 'embed_prefix_no_train' (build prefix embeddings from retrieved examples' token embeddings).")
    parser.add_argument("--retrieval_mlp_ckpt", "--checkpoint", type=str, default=None, help="Path to trained RetrievalPrefixMLP checkpoint for embed_prefix_w_train.")
    parser.add_argument("--embedding_output_mode", type=str, default=None, choices=["weighted_avg", "concat"],
    help="Retrieval embedding aggregate strategy: weighted_avg or concat.")

    parser.add_argument("--retrieved_visual", type=str, default="none", choices=["none", "image", "caption", "image_n_caption"],
    help="How to present retrieved examples: 'none' (text only), 'image' (attach example images), 'caption' (attach image captions).")
    parser.add_argument("--retrieved_visual_w_choice_img", action="store_true", help="If set, extract and save embeddings for each ScienceQA example.")

    parser.add_argument("--client_embeddings_dir", type=str, default=None)

    return parser.parse_args()