import argparse
import os
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized MLP projector training for PubMedQA/PubMedQA (Qwen3, text-only)."
    )

    # -------- Model --------
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID or local path for Qwen3.")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Load model in 4-bit via bitsandbytes.")

    # -------- Training data --------
    parser.add_argument("--train_json", type=str, required=True,
                        help="PubMedQA labeled train JSON (labeled_train_set.json).")
    parser.add_argument("--train_embeddings_path", type=str, required=True,
                        help="PT file: pre-computed question embeddings for training set "
                             "(used as queries against corpus FAISS).")

    # -------- Validation data --------
    parser.add_argument("--val_json", type=str, required=True,
                        help="PubMedQA labeled JSON for validation (test_set.json or a held-out split).")
    # Val queries are encoded on-the-fly at eval time using the frozen LM.

    # -------- Shared corpus (used for MLP training — must NOT contain test items) --------
    parser.add_argument("--corpus_embeddings_path", type=str, required=True,
                        help="PT file: pubmedqa_embeddings_pool.pt — question embeddings for the "
                             "combined corpus pool (used to build the retrieval FAISS index).")
    parser.add_argument("--content_embeddings_path", type=str, required=True,
                        help="PT file: pubmedqa_content_embeddings_pool.pt — content (title+abstract) "
                             "embeddings for the corpus pool (MLP input).")
    parser.add_argument("--corpus_jsonl", type=str, required=True,
                        help="JSONL file: pool_combined.jsonl — provides evidence text (title+content) "
                             "for teacher prompts.")

    # -------- Eval-only corpus (optional; includes test items for inference-time retrieval) --------
    # If not set, evaluation falls back to the training corpus above.
    parser.add_argument("--eval_corpus_embeddings_path", type=str, default=None,
                        help="PT file: question embeddings for the eval corpus pool "
                             "(may include test items; used only during evaluation).")
    parser.add_argument("--eval_content_embeddings_path", type=str, default=None,
                        help="PT file: content embeddings for the eval corpus pool "
                             "(used as MLP input during evaluation).")
    parser.add_argument("--eval_corpus_jsonl", type=str, default=None,
                        help="JSONL file: eval corpus pool — provides evidence text during evaluation.")

    # -------- MLP / training --------
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # -------- Stage 1 --------
    parser.add_argument("--stage1_pretrain", action="store_true",
                        help="Run stage-1 paraphrase pretraining.")
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=None,
                        help="Override lr for stage-1 (defaults to --lr).")
    parser.add_argument("--stage1_target_max_tokens", type=int, default=256,
                        help="Max tokens for the paraphrase target span.")
    parser.add_argument("--stage1_eval_every", type=int, default=200)

    # -------- Stage 2 --------
    parser.add_argument("--stage2_finetuning", action="store_true",
                        help="Run stage-2 answer fine-tuning.")
    parser.add_argument("--stage2_init_ckpt", type=str, default=None,
                        help="Path to stage-1 checkpoint to initialise stage-2.")
    parser.add_argument("--stage2_load_optim", action="store_true",
                        help="Also restore the optimizer state from stage2_init_ckpt.")
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage2_lr", type=float, default=None,
                        help="Override lr for stage-2 (defaults to --lr).")
    parser.add_argument("--stage2_finetune_topk", type=int, default=1,
                        help="Number of retrieved corpus items injected per question.")
    parser.add_argument("--stage2_eval_every", type=int, default=200)
    parser.add_argument("--stage2_wo_kl", action="store_true",
                        help="Disable KL distillation; use CE loss only.")
    parser.add_argument("--alpha_kl", type=float, default=1.0,
                        help="Weight for KL distillation loss.")
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--distill_topk", type=int, default=None,
                        help="Restrict KL to top-k teacher logits (None = full vocab).")
    parser.add_argument("--teacher_max_chars", type=int, default=5000,
                        help="Max chars of evidence text passed to teacher.")

    # -------- Generation / eval --------
    parser.add_argument("--eval_max_new_tokens", type=int, default=8)
    parser.add_argument("--eval_min_new_tokens", type=int, default=1)
    parser.add_argument("--eval_num_beams", type=int, default=1)
    parser.add_argument("--eval_do_sample", action="store_true")
    parser.add_argument("--eval_log_ce", action="store_true", default=True,
                        help="Compute CE loss during eval in addition to accuracy.")
    parser.add_argument("--acc_every", type=int, default=20,
                        help="Compute train accuracy every N steps.")
    parser.add_argument("--val_encode_batch_size", type=int, default=8,
                        help="Batch size for on-the-fly val query encoding.")

    # -------- Checkpointing / logging --------
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Directory to save per-epoch checkpoints.")
    parser.add_argument("--tb_dir", type=str, default="runs/pubmedqa_mlp",
                        help="TensorBoard log directory.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint to resume from.")

    args = parser.parse_args()

    # Derived defaults
    if args.stage1_lr is None:
        args.stage1_lr = args.lr
    if args.stage2_lr is None:
        args.stage2_lr = args.lr

    return args
