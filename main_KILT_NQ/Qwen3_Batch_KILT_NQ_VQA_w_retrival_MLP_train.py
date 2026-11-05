"""
Centralized MLP projector training for KILT-NQ (Qwen3, text-only).

Pipeline:
  1. Load train FAISS (question embeddings) for retrieval.
  2. Load pre-computed content embeddings (Q+A) as MLP input.
  3. Load frozen Qwen3, add <RepRAG> token.
  4. Create PrefixProjector(D_content -> H).
  5. Stage-1: paraphrase pretraining (reconstruct answer spans).
  6. Stage-2: answer CE fine-tuning.

Usage (see src/bash/VQA/w_embedding_retrival/KILT_NQ/):
    python Qwen3_Batch_KILT_NQ_VQA_w_retrival_MLP_train.py \\
        --model_name Qwen/Qwen3-8B \\
        --train_jsonl /.../datasets/KILT_NQ/train.jsonl \\
        --train_embeddings_path /.../kilt_nq_embeddings_train.pt \\
        --content_embeddings_path /.../kilt_nq_content_embeddings_train.pt \\
        --val_jsonl /.../datasets/KILT_NQ/validation.jsonl \\
        --stage1_pretrain --stage1_epochs 2 \\
        --stage2_finetuning --stage2_epochs 5 \\
        --batch_size 4 --lr 1e-4 \\
        --ckpt_dir /.../ckpts --tb_dir /.../tb_logs
"""

import os
import sys
import pathlib
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AddedToken,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]  # src/VQA
sys.path.append(str(ROOT))

from VQA.arg_KILT_NQ.KILT_NQ_mlp_train_arg import parse_args
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index
from VQA.utilise.retrieval_architecture import PrefixProjector
from VQA.utilise.KILT_NQ.KILT_NQ_data_process import load_kilt_nq, KILTNQDataset, collate_records
from VQA.utilise.KILT_NQ.KILT_NQ_prompt_message import build_kilt_nq_embedding_message
from VQA.utilise.KILT_NQ.KILT_NQ_mlp_train_stage1 import stage1_pretrain_step, stage1_eval_step
from VQA.utilise.KILT_NQ.KILT_NQ_mlp_train_stage2 import stage2_finetune_step, stage2_eval_step

REPRAG_TOKEN = "<RepRAG>"


# -------------------------------------------------------
# Model loading
# -------------------------------------------------------

def load_model(args):
    model_name = args.model_name
    print(f"[model] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.use_quantization:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    model.eval()
    return tokenizer, model


# -------------------------------------------------------
# On-the-fly val query encoding
# -------------------------------------------------------

def encode_records_online(records, tokenizer, base_model, device, batch_size=8, max_length=512):
    base_model.eval()
    all_embs, all_ids = [], []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        msgs = [build_kilt_nq_embedding_message(r, include_gold=False) for r in chunk]
        texts = [
            tokenizer.apply_chat_template(
                [m], tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            for m in msgs
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = base_model(**enc, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        all_embs.append(emb.cpu().float().numpy())
        all_ids.extend([r["id"] for r in chunk])

    emb_arr = np.concatenate(all_embs, axis=0)
    emb_arr = l2_normalize(emb_arr)
    id_to_row = {str(i): idx for idx, i in enumerate(all_ids)}
    return emb_arr, id_to_row


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir or ".", exist_ok=True)

    # --------------------------------------------------
    # 1) Load train data
    # --------------------------------------------------
    print(f"[data] Loading train data from {args.train_jsonl}")
    train_records = load_kilt_nq(args.train_jsonl)
    train_record_map = {r["id"]: r for r in train_records}
    print(f"[data] {len(train_records)} train records")

    print(f"[data] Loading val data from {args.val_jsonl}")
    val_records = load_kilt_nq(args.val_jsonl)
    print(f"[data] {len(val_records)} val records")

    # --------------------------------------------------
    # 2) Load pre-computed embeddings
    # --------------------------------------------------
    print(f"[data] Loading train question embeddings from {args.train_embeddings_path}")
    q_ckpt = torch.load(args.train_embeddings_path, map_location="cpu")
    train_emb_raw = q_ckpt["embeddings"].float().numpy()
    train_ids = [str(i) for i in q_ckpt["qids"]]
    train_emb_normed = l2_normalize(train_emb_raw)
    train_id_to_row = {i: idx for idx, i in enumerate(train_ids)}
    train_faiss_index = build_faiss_ip_index(train_emb_normed)
    print(f"[data] Train FAISS index: {train_emb_normed.shape}")

    print(f"[data] Loading content embeddings from {args.content_embeddings_path}")
    c_ckpt = torch.load(args.content_embeddings_path, map_location="cpu")
    content_emb_raw = c_ckpt["embeddings"].float().numpy()
    content_ids = [str(i) for i in c_ckpt["qids"]]
    content_emb_map = {i: content_emb_raw[idx] for idx, i in enumerate(content_ids)}
    D_content = content_emb_raw.shape[1]
    print(f"[data] Content embeddings: {content_emb_raw.shape}")

    # --------------------------------------------------
    # 3) Load model
    # --------------------------------------------------
    tokenizer, base_model = load_model(args)
    device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype

    # Add <RepRAG> token
    num_added = tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
    if num_added > 0:
        base_model.resize_token_embeddings(len(tokenizer))
    reprag_token_id = tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)
    print(f"[token] <RepRAG> id = {reprag_token_id}")

    # --------------------------------------------------
    # 4) Build MLP
    # --------------------------------------------------
    H = base_model.config.hidden_size
    print(f"[MLP] PrefixProjector: D_content={D_content} -> H={H}")
    mlp = PrefixProjector(
        input_size=D_content, output_size=H,
        projector_type="mlp2x_gelu", p_drop=0.1,
    ).to(device)
    mlp.train()

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.stage1_lr)

    # --------------------------------------------------
    # 5) DataLoaders
    # --------------------------------------------------
    stage1_loader = DataLoader(
        KILTNQDataset(train_records), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_records, drop_last=True,
    )
    stage2_loader = DataLoader(
        KILTNQDataset(train_records), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_records, drop_last=True,
    )
    val_loader = DataLoader(
        KILTNQDataset(val_records), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_records,
    )

    # --------------------------------------------------
    # 6) TensorBoard
    # --------------------------------------------------
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(args.tb_dir, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TB] Logging to {tb_log_dir}")

    if args.ckpt_dir:
        dump_args_to_logdir(args, args.ckpt_dir)

    # Val query encoder (on-the-fly)
    def get_val_query_emb():
        print("[Eval] Encoding val queries on-the-fly ...")
        emb, q2r = encode_records_online(
            val_records, tokenizer, base_model, device,
            batch_size=getattr(args, "val_encode_batch_size", 8),
        )
        print(f"[Eval] Val query embeddings: {emb.shape}")
        return emb, q2r

    ckpt_dir = args.ckpt_dir or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    # --------------------------------------------------
    # Stage 1 — paraphrase pretraining
    # --------------------------------------------------
    if args.stage1_pretrain:
        print(f"\n[Stage1] Starting for {args.stage1_epochs} epoch(s), lr={args.stage1_lr}")
        for g in optimizer.param_groups:
            g["lr"] = args.stage1_lr
        mlp.train()
        global_step = 0

        for epoch in range(args.stage1_epochs):
            print(f"\n[Stage1] Epoch {epoch+1}/{args.stage1_epochs}")
            epoch_loss = 0.0
            pbar = tqdm(stage1_loader, desc=f"Stage1 E{epoch+1}", dynamic_ncols=True)
            for batch in pbar:
                if not batch:
                    continue
                loss_val, global_step = stage1_pretrain_step(
                    args=args, base_model=base_model, tokenizer=tokenizer,
                    mlp=mlp, optimizer=optimizer, device=device, model_dtype=model_dtype,
                    batch_records=batch,
                    query_emb_normed=train_emb_normed, query_id_to_row=train_id_to_row,
                    train_faiss_index=train_faiss_index, train_ids=train_ids,
                    train_record_map=train_record_map, content_emb_map=content_emb_map,
                    writer=writer, global_step=global_step,
                    reprag_token_id=reprag_token_id, pbar=pbar,
                )
                epoch_loss += loss_val if np.isfinite(loss_val) else 0.0

                if global_step % args.stage1_eval_every == 0:
                    val_emb, val_q2r = get_val_query_emb()
                    v_losses = []
                    for val_batch in val_loader:
                        if not val_batch:
                            continue
                        vl = stage1_eval_step(
                            args=args, base_model=base_model, tokenizer=tokenizer,
                            mlp=mlp, device=device, model_dtype=model_dtype,
                            batch_records=val_batch,
                            query_emb_normed=val_emb, query_id_to_row=val_q2r,
                            train_faiss_index=train_faiss_index, train_ids=train_ids,
                            train_record_map=train_record_map, content_emb_map=content_emb_map,
                            writer=writer, global_step=global_step,
                            reprag_token_id=reprag_token_id,
                        )
                        if vl is not None:
                            v_losses.append(vl)
                    mlp.train()
                    if v_losses:
                        vmean = sum(v_losses) / len(v_losses)
                        writer.add_scalar("stage1/eval_avg_loss", vmean, global_step)
                        pbar.write(f"[Stage1] eval_avg_loss={vmean:.4f} @ step {global_step}")

            pbar.close()
            avg = epoch_loss / max(len(stage1_loader), 1)
            print(f"[Stage1] Epoch {epoch+1} done. avg_loss={avg:.4f}")
            writer.add_scalar("stage1/train_loss_epoch", avg, epoch + 1)

            ckpt_path = os.path.join(ckpt_dir, f"mlp_stage1_epoch{epoch+1:03d}.pt")
            torch.save({
                "epoch": epoch + 1, "global_step": global_step,
                "avg_loss": float(avg), "mlp_state": mlp.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[Stage1] Checkpoint saved → {ckpt_path}")

    # --------------------------------------------------
    # Stage 2 — answer fine-tuning
    # --------------------------------------------------
    if args.stage2_finetuning:
        print(f"\n[Stage2] Starting for {args.stage2_epochs} epoch(s), lr={args.stage2_lr}")

        if getattr(args, "stage2_init_ckpt", None):
            ckpt2 = torch.load(args.stage2_init_ckpt, map_location="cpu")
            mlp.load_state_dict(ckpt2["mlp_state"], strict=True)
            print(f"[Stage2] MLP loaded from {args.stage2_init_ckpt}")
            if getattr(args, "stage2_load_optim", False) and "optim_state" in ckpt2:
                optimizer.load_state_dict(ckpt2["optim_state"])
            global_step = int(ckpt2.get("global_step", 0))
        else:
            global_step = 0

        for g in optimizer.param_groups:
            g["lr"] = args.stage2_lr
        mlp.train()

        for epoch in range(args.stage2_epochs):
            total_loss = total_acc = total_n = acc_n = 0.0
            total_n_int = 0
            print(f"\n[Stage2] Epoch {epoch+1}/{args.stage2_epochs}")
            pbar = tqdm(stage2_loader, desc=f"Stage2 E{epoch+1}", dynamic_ncols=True)

            for batch in pbar:
                if not batch:
                    continue
                loss_val, acc_val, global_step = stage2_finetune_step(
                    args=args, epoch=epoch, step=0, global_step=global_step,
                    batch_records=batch, device=device, tokenizer=tokenizer,
                    base_model=base_model, mlp=mlp, optimizer=optimizer,
                    writer=writer, model_dtype=model_dtype,
                    query_emb_normed=train_emb_normed, query_id_to_row=train_id_to_row,
                    train_faiss_index=train_faiss_index, train_ids=train_ids,
                    train_record_map=train_record_map, content_emb_map=content_emb_map,
                    reprag_token_id=reprag_token_id, pbar=pbar,
                )
                if loss_val is None:
                    continue
                bs = len(batch)
                total_loss += float(loss_val) * bs
                total_n_int += bs
                if acc_val is not None:
                    total_acc += float(acc_val) * bs
                    acc_n += bs

                if global_step % args.stage2_eval_every == 0 and global_step > 0:
                    val_emb, val_q2r = get_val_query_emb()
                    vl_sum = va_sum = vl_n = va_n = 0.0
                    for val_batch in val_loader:
                        if not val_batch:
                            continue
                        vl, va = stage2_eval_step(
                            args=args, epoch=epoch, step=0, global_step=global_step,
                            batch_records=val_batch, device=device, tokenizer=tokenizer,
                            base_model=base_model, mlp=mlp, writer=writer,
                            model_dtype=model_dtype,
                            query_emb_normed=val_emb, query_id_to_row=val_q2r,
                            train_faiss_index=train_faiss_index, train_ids=train_ids,
                            train_record_map=train_record_map, content_emb_map=content_emb_map,
                            reprag_token_id=reprag_token_id,
                        )
                        vbs = len(val_batch)
                        if vl is not None: vl_sum += vl * vbs; vl_n += vbs
                        if va is not None: va_sum += va * vbs; va_n += vbs
                    mlp.train()
                    eval_acc = va_sum / max(va_n, 1)
                    eval_loss = vl_sum / max(vl_n, 1) if vl_n > 0 else None
                    writer.add_scalar("stage2/eval_avg_acc", eval_acc, global_step)
                    if eval_loss:
                        writer.add_scalar("stage2/eval_avg_loss", eval_loss, global_step)
                    train_loss_now = total_loss / max(total_n_int, 1)
                    pbar.write(f"[Stage2] step={global_step} train_loss={train_loss_now:.4f} "
                               f"eval_acc={eval_acc:.4f}"
                               + (f" eval_loss={eval_loss:.4f}" if eval_loss else ""))

            pbar.close()
            avg_loss = total_loss / max(total_n_int, 1)
            avg_acc = total_acc / max(acc_n, 1) if acc_n > 0 else 0.0
            print(f"==== [Stage2] Epoch {epoch+1} | loss={avg_loss:.4f} | acc={avg_acc:.4f} ====")
            writer.add_scalar("stage2/train_loss_epoch", avg_loss, epoch + 1)

            ckpt_path = os.path.join(ckpt_dir, f"mlp_stage2_epoch{epoch+1:03d}.pt")
            torch.save({
                "epoch": epoch + 1, "global_step": int(global_step),
                "avg_loss": float(avg_loss), "mlp_state": mlp.state_dict(),
                "optim_state": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[Stage2] Checkpoint saved → {ckpt_path}")

    writer.close()
    print("[Done]")


if __name__ == "__main__":
    main()