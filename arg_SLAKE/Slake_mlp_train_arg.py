import os
import argparse
import sys
import json
from datetime import datetime

# --------------------------
# Argument
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="", help="SLAKE train split json for training stage2.")
    parser.add_argument("--val_json", type=str, default="", help="SLAKE val split json for validation during training.")
    parser.add_argument("--test_json", type=str, default="", help="Optional SLAKE test split json for final eval.")
    parser.add_argument("--train_embeddings_path", type=str, required=True)
    parser.add_argument('--images_root', type=str, default="", help="Optional root to prepend to relative image paths in problems.json")
    parser.add_argument('--lang', type=str, default="en", choices=["en", "zh"], help="Filter question language (en or zh).")
    
    parser.add_argument("--embedding_output_mode", type=str, default="concat", choices=["weighted_avg", "concat"], help="Retrieval embedding aggregate strategy: weighted_avg or concat.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save per-epoch checkpoints (default: dirname of --mlp_ckpt_out)")
    parser.add_argument("--tb_dir", type=str, default="runs/injection_mlp", help="TensorBoard log dir")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume from")

    parser.add_argument("--stage1_pretrain", action="store_true", help="Run RepRAG-style stage-1 pretraining")
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage1_evidence_topk", type=int, default=1)
    parser.add_argument("--stage1_target_max_tokens", type=int, default=256)
    parser.add_argument("--stage1_lr", type=float, default=None, help="Override lr for stage-1 if set")
    parser.add_argument("--stage1_eval_every", type=int, default=200, help="Evalation frequency")

    parser.add_argument("--stage2_finetuning", action="store_true", help="Run RepRAG-style stage-2 finetuning")
    parser.add_argument("--stage2_init_ckpt", type=str, default=None, help="ckpt to start stage 2 training")
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage2_finetune_topk", type=int, default=1)
    parser.add_argument("--stage2_eval_every", type=int, default=200, help="Evalation frequency")
    parser.add_argument("--stage2_lr", type=float, default=None, help="Override lr for stage-1 if set")
    parser.add_argument("--stage2_wo_kl", action="store_true", help="Remove REPRAG Knowledge Distillation loss")

    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, no training.")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval_stage", type=str, default="stage2", choices=["stage1", "stage2"])
    parser.add_argument("--eval_ckpt", type=str, default=None, help="Path to MLP checkpoint (.pt) to evaluate.")

    return parser.parse_args()