import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated MLP projector training for PubMedQA/PubMedQA (Qwen3, text-only)."
    )

    # -------- Model --------
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID or local path for Qwen3.",
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Load model in 4-bit via bitsandbytes.",
    )

    # -------- Client Stage-1 corpus data --------
    parser.add_argument(
        "--client_corpus_dir",
        type=str,
        required=True,
        help="Directory containing client_*.jsonl files for each client's local combined corpus pool.",
    )
    parser.add_argument(
        "--client_question_embeddings_dir",
        type=str,
        required=True,
        help="Directory containing client_*.pt files of local question embeddings used to build each client's FAISS index.",
    )
    parser.add_argument(
        "--client_content_embeddings_dir",
        type=str,
        required=True,
        help="Directory containing client_*.pt files of local corpus/title+content embeddings used as MLP input.",
    )

    # -------- Client Stage-2 labeled train data --------
    parser.add_argument(
        "--client_train_dir",
        type=str,
        required=True,
        help="Directory containing client_*.json files for each client's labeled PubMedQA train split.",
    )
    parser.add_argument(
        "--client_train_embeddings_dir",
        type=str,
        required=True,
        help="Directory containing client_*.pt files of question embeddings for each client's labeled train set.",
    )

    # -------- Eval-only corpus (optional; includes test items for inference-time retrieval) --------
    # If not set, evaluation falls back to client 0's training corpus.
    parser.add_argument(
        "--eval_corpus_embeddings_path",
        type=str,
        default=None,
        help="PT file: question embeddings for the global eval corpus pool "
             "(may include test items; used only during evaluation, not training).",
    )
    parser.add_argument(
        "--eval_content_embeddings_path",
        type=str,
        default=None,
        help="PT file: content embeddings for the global eval corpus pool.",
    )
    parser.add_argument(
        "--eval_corpus_jsonl",
        type=str,
        default=None,
        help="JSONL file: global eval corpus pool providing evidence text during evaluation.",
    )

    # -------- Validation / optional test --------
    parser.add_argument(
        "--val_json",
        type=str,
        required=True,
        help="PubMedQA labeled JSON for validation.",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default=None,
        help="Optional PubMedQA labeled JSON for testing.",
    )
    parser.add_argument(
        "--val_encode_batch_size",
        type=int,
        default=8,
        help="Batch size for on-the-fly validation query encoding.",
    )

    # -------- Training --------
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    # -------- Stage 1 --------
    parser.add_argument(
        "--stage1_pretrain",
        action="store_true",
        help="Run Stage-1 paraphrase pretraining.",
    )
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument(
        "--stage1_lr",
        type=float,
        default=None,
        help="Override learning rate for Stage 1 (defaults to --lr).",
    )
    parser.add_argument(
        "--stage1_target_max_tokens",
        type=int,
        default=256,
        help="Maximum target length for the Stage-1 paraphrase span.",
    )
    parser.add_argument(
        "--stage1_eval_every",
        type=int,
        default=200,
        help="Validation frequency during Stage 1.",
    )
    parser.add_argument(
        "--stage1_clamp_inj",
        type=float,
        default=None,
        help="If set, clamp Stage-1 injection vectors to [-value, value] before inserting into "
             "inputs_embeds. Prevents float16 overflow when local FL training pushes MLP outputs "
             "out of range (e.g. --stage1_clamp_inj 100). Off by default.",
    )

    # -------- Stage 2 --------
    parser.add_argument(
        "--stage2_finetuning",
        action="store_true",
        help="Run Stage-2 answer fine-tuning.",
    )
    parser.add_argument(
        "--stage2_init_ckpt",
        type=str,
        default=None,
        help="Path to a Stage-1 checkpoint used to initialize Stage 2.",
    )
    parser.add_argument(
        "--stage2_load_optim",
        action="store_true",
        help="Also restore optimizer state from --stage2_init_ckpt when supported by the training code.",
    )
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument(
        "--stage2_lr",
        type=float,
        default=None,
        help="Override learning rate for Stage 2 (defaults to --lr).",
    )
    parser.add_argument(
        "--stage2_finetune_topk",
        type=int,
        default=1,
        help="Number of retrieved corpus items injected per question in Stage 2.",
    )
    parser.add_argument(
        "--stage2_eval_every",
        type=int,
        default=200,
        help="Validation frequency during Stage 2.",
    )
    parser.add_argument(
        "--stage2_wo_kl",
        action="store_true",
        help="Disable KL distillation and use CE loss only.",
    )
    parser.add_argument(
        "--alpha_kl",
        type=float,
        default=1.0,
        help="Weight for KL distillation loss.",
    )
    parser.add_argument(
        "--kl_temperature",
        type=float,
        default=1.0,
        help="Temperature used for KL distillation.",
    )
    parser.add_argument(
        "--distill_topk",
        type=int,
        default=None,
        help="Restrict KL distillation to top-k teacher logits (None = full vocab).",
    )
    parser.add_argument(
        "--teacher_max_chars",
        type=int,
        default=5000,
        help="Maximum number of evidence-text characters passed to the teacher model.",
    )

    # -------- Generation / eval --------
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=8,
        help="Maximum number of tokens generated during evaluation.",
    )
    parser.add_argument(
        "--eval_min_new_tokens",
        type=int,
        default=1,
        help="Minimum number of tokens generated during evaluation.",
    )
    parser.add_argument(
        "--eval_num_beams",
        type=int,
        default=1,
        help="Beam size used during evaluation decoding.",
    )
    parser.add_argument(
        "--eval_do_sample",
        action="store_true",
        help="Use sampling instead of deterministic decoding during evaluation.",
    )
    parser.add_argument(
        "--no_eval_log_ce",
        action="store_true",
        help="Disable CE-loss computation during evaluation.",
    )
    parser.add_argument(
        "--acc_every",
        type=int,
        default=20,
        help="Compute train accuracy every N steps.",
    )
    parser.add_argument(
        "--eval_client_id",
        type=int,
        default=0,
        help="Client retrieval bank used for evaluation. Set to -1 to evaluate on all clients; otherwise use a single client index.",
    )

    # -------- FL --------
    parser.add_argument(
        "--fl_method",
        type=str,
        default="fedRepRag",
        choices=["fedRepRag"],
        help="Federated optimization method.",
    )
    parser.add_argument(
        "--comm_rounds",
        type=int,
        default=10,
        help="Number of federated communication rounds.",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local epochs per communication round.",
    )

    # -------- Eval-only --------
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, no training.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation-only mode.",
    )
    parser.add_argument(
        "--eval_stage",
        type=str,
        default="stage2",
        choices=["stage1", "stage2"],
        help="Checkpoint stage used for evaluation-only mode.",
    )
    parser.add_argument(
        "--eval_ckpt",
        type=str,
        default=None,
        help="Path to an MLP checkpoint (.pt) for evaluation-only mode.",
    )

    # -------- Logging / checkpointing --------
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default="runs/pubmedqa_fl_mlp",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )

    args = parser.parse_args()

    if args.stage1_lr is None:
        args.stage1_lr = args.lr
    if args.stage2_lr is None:
        args.stage2_lr = args.lr

    args.eval_log_ce = not args.no_eval_log_ce

    return args