import os
import sys
import pathlib
from typing import List, Dict, Any, Tuple
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AddedToken
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]  # go up to repo root
sys.path.append(str(ROOT))

from qwen_vl_utils import process_vision_info

from VQA.arg_SLAKE.Slake_mlp_train_arg import parse_args

from VQA.utilise.model_loading import load_model_standard
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index, retrieve_neighbors_for_batch
from VQA.utilise.retrieval_architecture import PrefixProjector
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir

from VQA.utilise.SLAKE.Slake_data_process import TrainInjectionDataset, collate_tasks
from VQA.utilise.SLAKE.Slake_data_process import load_slake_items, build_tasks_from_slake, load_pil_or_none, load_captions_map
from VQA.utilise.SLAKE.Slake_prompt_message import build_slake_vqa_message, build_slake_rag_incontext_message

from VQA.utilise.SLAKE.Slake_mlp_train_stage1 import stage1_pretrain_step, stage1_eval_step
from VQA.utilise.SLAKE.Slake_mlp_train_stage2 import stage2_finetune_step, stage2_eval_step

REPRAG_TOKEN = "<RepRAG>"


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    
    # make sure you have log_dir available here
    log_dir = getattr(args, "ckpt_dir", None)
    if log_dir is None:
        # fallback if your code uses another name
        log_dir = getattr(args, "output_dir", "./logs")

    dump_args_to_logdir(args, log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    # 1) Load train embeddings & build FAISS
    # ----------------------------------------------
    ckpt = torch.load(args.train_embeddings_path, map_location="cpu")
    train_emb = ckpt["embeddings"].cpu().numpy().astype("float32")  # [N, D]
    train_qids = [str(q) for q in ckpt["qids"]]                      # [N]
    N, D_emb = train_emb.shape
    print(f"[Emb] Loaded {N} train embeddings (dim={D_emb}).")

    # Normalize and build FAISS index
    train_emb_normed = l2_normalize(train_emb)
    index = build_faiss_ip_index(train_emb_normed)
    print(f"[FAISS] Built IP index with {index.ntotal} vectors.")

    # map qid -> row idx
    qid_to_row = {int(qid): i for i, qid in enumerate(train_qids)}

    # ----------------------------------------------
    # 2) Load SLAKE tasks (train/val provided by dataset)
    # ----------------------------------------------
    if not getattr(args, "train_json", None):
        raise ValueError("--train_json is required for SLAKE training.")
    if not getattr(args, "val_json", None):
        raise ValueError("--val_json is required for SLAKE validation.")

    lang = getattr(args, "lang", "en")
    images_root = getattr(args, "images_root", None)

    # ---- train split ----
    train_items = load_slake_items(args.train_json)
    train_tasks_all = build_tasks_from_slake(train_items, images_root=images_root, split_name="train", lang_filter=lang)
    print(f"[SLAKE] Train tasks loaded: {len(train_tasks_all)} from {args.train_json}")

    # ---- val split ----
    val_items = load_slake_items(args.val_json)
    val_tasks_all = build_tasks_from_slake(val_items, images_root=images_root, split_name="val", lang_filter=lang)
    print(f"[SLAKE] Val tasks loaded: {len(val_tasks_all)} from {args.val_json}")

    # map qid -> task (IMPORTANT: retrieval examples should come from TRAIN split only)
    train_task_map = {str(t["qid"]): t for t in train_tasks_all}

    # loaders (dataset filters gold_answer inside TrainInjectionDataset)
    train_dataset = TrainInjectionDataset(train_tasks_all)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tasks)

    val_dataset = TrainInjectionDataset(val_tasks_all)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tasks)

    # ----------------------------------------------
    # 3) Load frozen VLM
    # ----------------------------------------------
    processor, qwen_model = load_model_standard(args.model_name)
    processor.tokenizer.padding_side = "left"

    qwen_model.to(device)
    base_model = qwen_model.module if hasattr(qwen_model, "module") else qwen_model

    # Add reprag_token_id token
    num_added = processor.tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
    if num_added > 0:
        # IMPORTANT: resize embeddings even if model is frozen
        base_model.resize_token_embeddings(len(processor.tokenizer))

    # ---- IMPORTANT: sync config.vocab_size to actual lm head ----
    out_emb = base_model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight"):
        base_model.config.vocab_size = int(out_emb.weight.shape[0])
    else:
        # fallback (rare)
        base_model.config.vocab_size = int(base_model.get_input_embeddings().weight.shape[0])

    # (Optional but helpful) sanity check
    assert base_model.config.vocab_size == base_model.get_output_embeddings().weight.shape[0], \
        f"vocab mismatch: config={base_model.config.vocab_size}, head={base_model.get_output_embeddings().weight.shape[0]}"

    reprag_token_id = processor.tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)
    assert reprag_token_id != processor.tokenizer.unk_token_id

    qwen_model.requires_grad_(False)
    qwen_model.eval()
    
    base_model.requires_grad_(False)
    base_model.eval()
    
    model_dtype = next(base_model.parameters()).dtype

    # ----------------------------------------------
    # 4) MLP projector: retrieval embedding [H] -> injection vector [H]
    # ----------------------------------------------
    H = getattr(base_model.config, "hidden_size", None) or getattr(getattr(base_model.config, "text_config", None), "hidden_size", None)  # 3584
    mlp = PrefixProjector(input_size=H, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1).to(device)
    
    mlp.requires_grad_(True)
    
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.stage1_lr)

    # If PrefixProjector isn't guaranteed to output H, assert early:
    with torch.no_grad():
        dummy = torch.zeros(2, H, device=device)
        out = mlp(dummy)
        assert out.shape[-1] == H, f"MLP output dim must be H={H}, got {out.shape}"

    # ----------------------------------------------
    # 5) TensorBoard
    # ----------------------------------------------
    # Make a unique run folder so multiple runs don't overwrite each other
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(args.tb_dir, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TB] Logging to {tb_log_dir}")

    # ----------------------------------------------
    # 6) Training - Stage 1 - pretraining
    # ----------------------------------------------
    if args.stage1_pretrain:
        print(f"[Stage1] Starting stage-1 pretraining for {args.stage1_epochs} epochs")

        for g in optimizer.param_groups:
            g["lr"] = args.stage1_lr
        print(f"[Stage1] Using lr={args.stage1_lr}")

        mlp.train()
        global_step = 0

        ckpt_dir = args.ckpt_dir or "."
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(args.stage1_epochs):
            print(f"\n[Stage1] Epoch {epoch+1}/{args.stage1_epochs}")
            epoch_loss = 0.0

            for step_idx, batch_tasks in enumerate(dataloader):
                if not batch_tasks:
                    continue

                loss_val, global_step = stage1_pretrain_step(args=args, base_model=base_model, qwen_model=qwen_model, processor=processor, mlp=mlp, 
                optimizer=optimizer, device=device, model_dtype=model_dtype, batch_tasks=batch_tasks, train_emb_normed=train_emb_normed, 
                train_qids=train_qids, qid_to_row=qid_to_row, index=index, train_task_map=train_task_map, writer=writer, global_step=global_step,
                reprag_token_id=reprag_token_id)

                epoch_loss += loss_val

                if (step_idx + 1) % 50 == 0:  # Print every 50 update steps
                    print(f"[Stage1] epoch={epoch+1} step={step_idx+1}/{len(dataloader)} loss={loss_val:.4f}")

                if global_step % args.stage1_eval_every == 0:
                    val_loss_list = []
                    for i, val_batch in enumerate(val_loader):
                        vloss = stage1_eval_step(args=args, base_model=base_model, qwen_model=qwen_model, processor=processor, mlp=mlp, 
                        device=device, model_dtype=model_dtype, batch_tasks=val_batch, train_emb_normed=train_emb_normed, train_qids=train_qids, 
                        qid_to_row=qid_to_row, index=index, train_task_map=train_task_map, writer=writer, global_step=global_step, 
                        reprag_token_id=reprag_token_id, eval_prefix="stage1/eval")
                        if vloss is not None:
                            val_loss_list.append(vloss)
                    
                    val_mean = sum(val_loss_list) / max(1, len(val_loss_list))
                    writer.add_scalar("stage1/eval_avg_loss", val_mean, global_step)
                    print(f"[Stage1] eval_avg_loss={val_mean:.4f}")

            avg_loss = epoch_loss / max(len(dataloader), 1)
            print(f"[Stage1] Epoch {epoch+1} finished. Avg loss = {avg_loss:.4f}")
            writer.add_scalar("stage1/train_loss_epoch", avg_loss, epoch+1)

            # ---- Save per-epoch checkpoint ----
            ckpt_path_epoch = os.path.join(ckpt_dir, f"mlp_stage1_epoch{epoch+1:03d}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "avg_loss": float(avg_loss),
                    "mlp_state": mlp.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                ckpt_path_epoch,
            )
            print(f"[Stage1] Saved epoch checkpoint to {ckpt_path_epoch}")

    # ----------------------------------------------
    # 7) Training - Stage 2 - finetuning
    # ----------------------------------------------
    if args.stage2_finetuning:
        print(f"[Stage2] Starting stage-2 finetuning for {args.stage2_epochs} epochs")

        # ---- (A) Optionally load from a stage-1 checkpoint ----
        if getattr(args, "stage2_init_ckpt", None):
            ckpt2 = torch.load(args.stage2_init_ckpt, map_location="cpu")
            if "mlp_state" in ckpt2:
                mlp.load_state_dict(ckpt2["mlp_state"], strict=True)
                print(f"[Stage2] Loaded MLP from {args.stage2_init_ckpt}")
            if getattr(args, "stage2_load_optim", False) and ("optim_state" in ckpt2):
                optimizer.load_state_dict(ckpt2["optim_state"])
                print(f"[Stage2] Loaded optimizer state from {args.stage2_init_ckpt}")

            if "global_step" in ckpt2:
                global_step = int(ckpt2["global_step"])
                print(f"[Stage2] Resume global_step = {global_step}")
            else:
                pass
        else:
            # If stage1 ran in this same run, reuse its global_step; otherwise start from 0.
            if "global_step" not in locals():
                global_step = 0

        # ---- (B) Stage-2 LR override ----
        for g in optimizer.param_groups:
            g["lr"] = args.stage2_lr
        print(f"[Stage2] Using lr={args.stage2_lr}")

        mlp.train()

        ckpt_dir = args.ckpt_dir or "."
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(args.stage2_epochs):
            # epoch-level (weighted) stats
            total_loss_sum = 0.0
            total_acc_sum = 0.0
            total_samples = 0
            total_acc_samples = 0

            print(f"\n[Stage2] Epoch {epoch+1}/{args.stage2_epochs}")
            for step_idx, batch_tasks in enumerate(dataloader):
                bs = len(batch_tasks) if batch_tasks is not None else 0
                if bs == 0:
                    continue

                loss_val, acc_val, global_step = stage2_finetune_step(args=args, epoch=epoch, step=step_idx, global_step=global_step, 
                batch_tasks=batch_tasks, device=device, processor=processor, qwen_model=qwen_model, base_model=base_model, mlp=mlp, 
                optimizer=optimizer, writer=writer, model_dtype=model_dtype, train_emb_normed=train_emb_normed, train_qids=train_qids, 
                qid_to_row=qid_to_row, index=index, train_task_map=train_task_map, reprag_token_id=reprag_token_id)

                if loss_val is None:
                    continue

                # always accumulate loss
                total_loss_sum += float(loss_val) * bs
                total_samples += bs

                # accumulate acc only when it exists
                if acc_val is not None:
                    total_acc_sum += float(acc_val) * bs
                    total_acc_samples += bs

                # running global (within-epoch) metrics
                train_loss_global = total_loss_sum / max(1, total_samples)
                train_acc_global = total_acc_sum / max(1, total_acc_samples) if total_acc_samples > 0 else 0.0

                if (step_idx + 1) % 50 == 0:
                    msg = (
                        f"[Stage2][Train] epoch={epoch+1} step={step_idx+1}/{len(dataloader)} "
                        f"batch_loss={float(loss_val):.4f}"
                    )
                    if acc_val is not None:
                        msg += f" batch_acc={float(acc_val):.4f}"
                    msg += f" | global_loss={train_loss_global:.4f}"
                    if total_acc_samples > 0:
                        msg += f" global_acc={train_acc_global:.4f}"
                    print(msg)

                # ---- validation ----
                if global_step % args.stage2_eval_every == 0:
                    val_loss_sum = 0.0
                    val_loss_n = 0
                    val_acc_sum = 0.0
                    val_acc_n = 0

                    for i, val_batch in enumerate(val_loader):
                        v_bs = len(val_batch) if val_batch is not None else 0
                        if v_bs == 0:
                            continue

                        vloss, vacc = stage2_eval_step(args=args, epoch=epoch, step=i, global_step=global_step, batch_tasks=val_batch, device=device, 
                        processor=processor, qwen_model=qwen_model, base_model=base_model, mlp=mlp, writer=writer, model_dtype=model_dtype, 
                        train_emb_normed=train_emb_normed, train_qids=train_qids, qid_to_row=qid_to_row, index=index, train_task_map=train_task_map, 
                        reprag_token_id=reprag_token_id, eval_prefix="stage2/eval")
                        
                        if vacc is not None:
                            val_acc_sum += float(vacc) * v_bs
                            val_acc_n += v_bs

                        if vloss is not None:
                            val_loss_sum += float(vloss) * v_bs
                            val_loss_n += v_bs

                    eval_avg_acc = val_acc_sum / max(1, val_acc_n)
                    eval_avg_loss = (val_loss_sum / max(1, val_loss_n)) if val_loss_n > 0 else None

                    # TB logging
                    writer.add_scalar("stage2/eval_avg_acc", float(eval_avg_acc), global_step)
                    if eval_avg_loss is not None:
                        writer.add_scalar("stage2/eval_avg_loss", float(eval_avg_loss), global_step)

                    # Print global train + global val at this checkpoint
                    msg = (
                        f"[Stage2][Eval] global_step={global_step} | "
                        f"train_global_loss={train_loss_global:.4f}"
                    )
                    if total_acc_samples > 0:
                        msg += f" train_global_acc={train_acc_global:.4f}"
                    msg += f" | eval_avg_acc={eval_avg_acc:.4f}"
                    if eval_avg_loss is not None:
                        msg += f" eval_avg_loss={eval_avg_loss:.4f}"
                    print(msg)

            # ---- end epoch summary ----
            avg_loss = total_loss_sum / max(1, total_samples)
            avg_acc = total_acc_sum / max(1, total_acc_samples) if total_acc_samples > 0 else 0.0

            print(
                f"==== [Stage2] Epoch {epoch+1}/{args.stage2_epochs} | "
                f"Train loss: {avg_loss:.4f} | Train acc: {avg_acc:.4f} ===="
            )
            writer.add_scalar("stage2/train_loss_epoch", float(avg_loss), epoch + 1)
            if total_acc_samples > 0:
                writer.add_scalar("stage2/train_acc_epoch", float(avg_acc), epoch + 1)

            # ---- Save per-epoch checkpoint ----
            ckpt_path = os.path.join(ckpt_dir, f"mlp_stage2_epoch_{epoch+1:03d}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": int(global_step),
                    "avg_loss": float(avg_loss),
                    "mlp_state": mlp.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                ckpt_path
            )
            print(f"[Stage2] Saved epoch checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
