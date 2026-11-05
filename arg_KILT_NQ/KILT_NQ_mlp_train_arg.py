import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized MLP projector training for KILT-NQ (Qwen3, text-only)."
    )

    # -------- Model --------
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Load model in 4-bit via bitsandbytes.")

    # -------- Training data --------
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="KILT-NQ train JSONL (used as retrieval corpus and training source).")
    parser.add_argument("--train_embeddings_path", type=str, required=True,
                        help="PT file: question-only embeddings for train split (FAISS index).")
    parser.add_argument("--content_embeddings_path", type=str, required=True,
                        help="PT file: Q+A content embeddings for train split (MLP input).")

    # -------- Validation data --------
    parser.add_argument("--val_jsonl", type=str, required=True,
                        help="KILT-NQ validation JSONL (val queries encoded on-the-fly).")

    # -------- MLP / training --------
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # -------- Stage 1 --------
    parser.add_argument("--stage1_pretrain", action="store_true",
                        help="Run stage-1 paraphrase pretraining.")
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=None)
    parser.add_argument("--stage1_target_max_tokens", type=int, default=64,
                        help="Max tokens for the paraphrase target span (answers are short).")
    parser.add_argument("--stage1_eval_every", type=int, default=500)

    # -------- Stage 2 --------
    parser.add_argument("--stage2_finetuning", action="store_true",
                        help="Run stage-2 answer fine-tuning.")
    parser.add_argument("--stage2_init_ckpt", type=str, default=None,
                        help="Stage-1 checkpoint to initialise stage-2.")
    parser.add_argument("--stage2_load_optim", action="store_true")
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage2_lr", type=float, default=None)
    parser.add_argument("--stage2_finetune_topk", type=int, default=3,
                        help="Number of retrieved train items injected per question.")
    parser.add_argument("--stage2_eval_every", type=int, default=500)
    parser.add_argument("--stage2_answer_max_tokens", type=int, default=16,
                        help="Max tokens for gold answer tokenization in stage-2.")
    parser.add_argument("--stage2_wo_kl", action="store_true",
                        help="Disable KL distillation; use CE loss only.")
    parser.add_argument("--alpha_kl", type=float, default=1.0,
                        help="Weight for KL distillation loss.")
    parser.add_argument("--kl_temperature", type=float, default=1.0)

    # -------- Eval --------
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    parser.add_argument("--eval_num_beams", type=int, default=1)
    parser.add_argument("--val_encode_batch_size", type=int, default=8)
    parser.add_argument("--acc_every", type=int, default=50)

    # -------- Checkpointing / logging --------
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--tb_dir", type=str, default="runs/kilt_nq_mlp")

    args = parser.parse_args()

    if args.stage1_lr is None:
        args.stage1_lr = args.lr
    if args.stage2_lr is None:
        args.stage2_lr = args.lr

    return args