import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated MLP projector training for KILT-NQ (Qwen3, text-only)."
    )

    # -------- Model --------
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--use_quantization", action="store_true",
                        help="Load model in 4-bit via bitsandbytes.")

    # -------- Per-client corpus data --------
    parser.add_argument("--client_corpus_dir", type=str, required=True,
                        help="Directory containing client_N.jsonl files (FL split output).")

    # -------- Global pre-computed embeddings --------
    parser.add_argument("--global_question_embeddings_path", type=str, required=True,
                        help="PT file: global question-only embeddings for all train items.")
    parser.add_argument("--global_content_embeddings_path", type=str, required=True,
                        help="PT file: global Q+A content embeddings for all train items (MLP input).")

    # -------- Validation data --------
    parser.add_argument("--val_jsonl", type=str, required=True,
                        help="KILT-NQ validation JSONL (queries encoded on-the-fly).")

    # -------- Training --------
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # -------- Stage 1 --------
    parser.add_argument("--stage1_pretrain", action="store_true",
                        help="Run stage-1 paraphrase pretraining.")
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=None)
    parser.add_argument("--stage1_target_max_tokens", type=int, default=64)
    parser.add_argument("--stage1_eval_every", type=int, default=500)
    parser.add_argument("--stage1_clamp_inj", type=float, default=None,
                        help="Clamp Stage-1 injection vectors to [-value, value] to prevent overflow.")

    # -------- Stage 2 --------
    parser.add_argument("--stage2_finetuning", action="store_true",
                        help="Run stage-2 answer fine-tuning.")
    parser.add_argument("--stage2_init_ckpt", type=str, default=None,
                        help="Stage-1 checkpoint to initialise stage-2.")
    parser.add_argument("--stage2_epochs", type=int, default=5)
    parser.add_argument("--stage2_lr", type=float, default=None)
    parser.add_argument("--stage2_finetune_topk", type=int, default=3,
                        help="Number of retrieved train items injected per question.")
    parser.add_argument("--stage2_eval_every", type=int, default=500)
    parser.add_argument("--stage2_answer_max_tokens", type=int, default=16)
    parser.add_argument("--stage2_wo_kl", action="store_true",
                        help="Disable KL distillation; use CE loss only.")
    parser.add_argument("--alpha_kl", type=float, default=1.0)
    parser.add_argument("--kl_temperature", type=float, default=1.0)

    # -------- Eval --------
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    parser.add_argument("--eval_num_beams", type=int, default=1)
    parser.add_argument("--val_encode_batch_size", type=int, default=8)
    parser.add_argument("--acc_every", type=int, default=50)
    parser.add_argument("--eval_client_id", type=int, default=0,
                        help="Client retrieval bank used for validation. -1 = all clients.")

    # -------- FL --------
    parser.add_argument("--fl_method", type=str, default="fedRepRag",
                        choices=["fedRepRag"],
                        help="Federated optimization method.")
    parser.add_argument("--comm_rounds", type=int, default=10,
                        help="Number of federated communication rounds.")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="Local training epochs per communication round.")

    # -------- Checkpointing / logging --------
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--tb_dir", type=str, default="runs/kilt_nq_fl_mlp")

    args = parser.parse_args()

    if args.stage1_lr is None:
        args.stage1_lr = args.lr
    if args.stage2_lr is None:
        args.stage2_lr = args.lr

    return args