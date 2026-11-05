"""
Centralized MLP projector training for PubMedQA/PubMedQA (Qwen3, text-only).

Pipeline:
  1. Load corpus pool FAISS (question embeddings) for retrieval.
  2. Load pre-computed content embeddings (title+abstract) as MLP input.
  3. Load PubMedQA train + val tasks.
  4. Load frozen Qwen3, add <RepRAG> token.
  5. Create PrefixProjector(D_content → H).
  6. Stage-1: paraphrase pretraining (reconstruct passage spans).
  7. Stage-2: answer CE + optional KL distillation (yes/no/maybe → A/B/C).

Val queries are encoded on-the-fly with the frozen LM at each eval cycle.

Usage (see src/bash/VQA/w_embedding_retrival/PubMedQA/):
    python Qwen3_Batch_PubMedQA_VQA_w_retrival_MLP_train.py \\
        --model_name Qwen/Qwen3-8B \\
        --train_json /.../labeled_train_set.json \\
        --train_embeddings_path /.../pubmedqa_embeddings_train.pt \\
        --val_json /.../test_set.json \\
        --corpus_embeddings_path /.../pubmedqa_embeddings_pool.pt \\
        --content_embeddings_path /.../pubmedqa_content_embeddings_pool.pt \\
        --corpus_jsonl /.../pool_combined.jsonl \\
        --stage1_pretrain --stage1_epochs 2 \\
        --stage2_finetuning --stage2_epochs 3 \\
        --batch_size 4 --lr 1e-4 \\
        --ckpt_dir /.../ckpts --tb_dir /.../tb_logs
"""

import os
import sys
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AddedToken,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]  # src/VQA
sys.path.append(str(ROOT))

from VQA.arg_PubMedQA.PubMedQA_mlp_train_arg import parse_args
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index
from VQA.utilise.retrieval_architecture import PrefixProjector

from VQA.utilise.PubMedQA.PubMedQA_data_process import load_pubmedqa_train, load_combined_pool, PubMedQADataset, collate_tasks
from VQA.utilise.PubMedQA.PubMedQA_prompt_message import build_pubmedqa_embedding_message
from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage1 import stage1_pretrain_step, stage1_eval_step
from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage2 import stage2_finetune_step, stage2_eval_step

REPRAG_TOKEN = "<RepRAG>"


# -------------------------------------------------------
# Model loading (text-only Qwen3)
# -------------------------------------------------------

def load_model(args):
    model_name = args.model_name
    print(f"[model] Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


# -------------------------------------------------------
# On-the-fly val query encoding
# -------------------------------------------------------

def encode_tasks_online(
    tasks: List[Dict[str, Any]],
    tokenizer,
    base_model,
    device,
    batch_size: int = 8,
    max_length: int = 512,
) -> tuple:
    """
    Encode val/test query tasks on-the-fly with the frozen LM.

    Uses build_pubmedqa_embedding_message (question only) to build messages,
    then mean-pools the last hidden states and L2-normalises.

    Returns:
        emb_normed  : np.ndarray [N, D], float32, L2-normalised
        qid_to_row  : dict[str, int]  mapping qid -> row in emb_normed
    """
    was_training = base_model.training
    base_model.eval()

    all_embeddings = []
    qids_order = []

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        msgs = [build_pubmedqa_embedding_message(t, include_gold=False) for t in chunk]

        texts = [
            tokenizer.apply_chat_template(
                [m], tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            for m in msgs
        ]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = base_model(**enc, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1].float()               # [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)                   # [B, D]
            emb = summed / mask.sum(dim=1).clamp(min=1e-6)       # [B, D]
            norm = emb.norm(p=2, dim=1, keepdim=True)
            emb = emb / norm.clamp(min=1e-6)

        all_embeddings.append(emb.cpu().numpy())
        for t in chunk:
            qids_order.append(str(t["qid"]))

    base_model.train(was_training)

    emb_normed = np.concatenate(all_embeddings, axis=0).astype("float32")
    qid_to_row = {qid: i for i, qid in enumerate(qids_order)}
    return emb_normed, qid_to_row


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    args = parse_args()

    log_dir = args.ckpt_dir or "./logs"
    os.makedirs(log_dir, exist_ok=True)
    dump_args_to_logdir(args, log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1) Load corpus pool embeddings → build FAISS
    # --------------------------------------------------
    print(f"[Corpus] Loading corpus embeddings from {args.corpus_embeddings_path}")
    corp_ckpt = torch.load(args.corpus_embeddings_path, map_location="cpu")
    corpus_emb = corp_ckpt["embeddings"].cpu().numpy().astype("float32")   # [N_c, D]
    corpus_qids = [str(q) for q in corp_ckpt["qids"]]                      # [N_c]
    N_c, D_q = corpus_emb.shape
    print(f"[Corpus] {N_c} corpus embeddings (dim={D_q}).")

    corpus_emb_normed = l2_normalize(corpus_emb)
    corpus_faiss_index = build_faiss_ip_index(corpus_emb_normed)
    # Reverse map: combined pool qid -> row in corpus_emb_normed.
    # Used as query lookup for Stage 1 (all 62,249 combined pool items).
    corpus_qid_to_row = {qid: i for i, qid in enumerate(corpus_qids)}
    print(f"[FAISS]  Corpus index built: {corpus_faiss_index.ntotal} vectors.")

    # --------------------------------------------------
    # 2) Load labeled-train query embeddings (Stage 2 only)
    # --------------------------------------------------
    # Stage 2 trains on labeled items whose qids (plain PMIDs from
    # load_pubmedqa_train) differ from the combined-pool prefix format
    # (pqa_labeled_<pmid>), so a separate embedding file is needed.
    print(f"[Train]  Loading labeled-train query embeddings from {args.train_embeddings_path}")
    train_emb_ckpt = torch.load(args.train_embeddings_path, map_location="cpu")
    train_emb = train_emb_ckpt["embeddings"].cpu().numpy().astype("float32")  # [N_t, D]
    train_emb_qids = [str(q) for q in train_emb_ckpt["qids"]]
    N_t, _ = train_emb.shape
    print(f"[Train]  {N_t} labeled-train query embeddings.")

    train_emb_normed = l2_normalize(train_emb)
    train_qid_to_row = {qid: i for i, qid in enumerate(train_emb_qids)}

    # --------------------------------------------------
    # 3) Load content embeddings (title+abstract) → content_emb_map
    #    MLP input: content embeddings of retrieved corpus items.
    # --------------------------------------------------
    print(f"[Content] Loading content embeddings from {args.content_embeddings_path}")
    cont_ckpt = torch.load(args.content_embeddings_path, map_location="cpu")
    cont_emb = cont_ckpt["embeddings"].cpu().numpy().astype("float32")  # [N_c, D_content]
    cont_qids = [str(q) for q in cont_ckpt["qids"]]
    content_emb_map: Dict[str, np.ndarray] = {
        qid: cont_emb[i] for i, qid in enumerate(cont_qids)
    }
    D_content = cont_emb.shape[1]
    print(f"[Content] {len(content_emb_map)} content embeddings (dim={D_content}).")

    # --------------------------------------------------
    # 4) Load corpus text → corpus_task_map
    # --------------------------------------------------
    print(f"[Corpus] Loading corpus text from {args.corpus_jsonl}")
    corpus_tasks = load_combined_pool(args.corpus_jsonl, idx=0, num_parts=1)
    corpus_task_map: Dict[str, Dict[str, Any]] = {
        str(t["qid"]): t for t in corpus_tasks
    }
    print(f"[Corpus] {len(corpus_task_map)} corpus entries loaded.")

    # --------------------------------------------------
    # 4b) Load eval-only corpus (includes test items for inference-time retrieval).
    #     Falls back to training corpus if --eval_corpus_* args are not provided.
    # --------------------------------------------------
    if args.eval_corpus_embeddings_path:
        print(f"[EvalCorpus] Loading eval corpus embeddings from {args.eval_corpus_embeddings_path}")
        eval_corp_ckpt = torch.load(args.eval_corpus_embeddings_path, map_location="cpu")
        eval_corpus_emb = eval_corp_ckpt["embeddings"].cpu().numpy().astype("float32")
        eval_corpus_qids = [str(q) for q in eval_corp_ckpt["qids"]]
        eval_corpus_emb_normed = l2_normalize(eval_corpus_emb)
        eval_corpus_faiss_index = build_faiss_ip_index(eval_corpus_emb_normed)
        print(f"[EvalCorpus] {len(eval_corpus_qids)} eval corpus embeddings; FAISS index built.")

        print(f"[EvalCorpus] Loading eval content embeddings from {args.eval_content_embeddings_path}")
        eval_cont_ckpt = torch.load(args.eval_content_embeddings_path, map_location="cpu")
        eval_cont_emb = eval_cont_ckpt["embeddings"].cpu().numpy().astype("float32")
        eval_cont_qids = [str(q) for q in eval_cont_ckpt["qids"]]
        eval_content_emb_map: Dict[str, np.ndarray] = {
            qid: eval_cont_emb[i] for i, qid in enumerate(eval_cont_qids)
        }

        print(f"[EvalCorpus] Loading eval corpus text from {args.eval_corpus_jsonl}")
        eval_corpus_tasks = load_combined_pool(args.eval_corpus_jsonl, idx=0, num_parts=1)
        eval_corpus_task_map: Dict[str, Dict[str, Any]] = {
            str(t["qid"]): t for t in eval_corpus_tasks
        }
        print(f"[EvalCorpus] {len(eval_corpus_task_map)} eval corpus entries loaded.")
    else:
        print("[EvalCorpus] No separate eval corpus provided; falling back to training corpus.")
        eval_corpus_faiss_index = corpus_faiss_index
        eval_corpus_qids        = corpus_qids
        eval_content_emb_map    = content_emb_map
        eval_corpus_task_map    = corpus_task_map

    # --------------------------------------------------
    # 5) Load tasks
    #    Stage 1: full combined pool (62,249 items, no gold answer needed)
    #    Stage 2: labeled train only  (500 items, gold answer required)
    # --------------------------------------------------
    # Stage 1 training data: all combined pool items
    stage1_train_tasks = corpus_tasks   # already loaded above
    print(f"[Data] Stage-1 train tasks (combined pool): {len(stage1_train_tasks)}")

    # Stage 2 training data: labeled train
    print(f"[Data] Loading labeled-train tasks from {args.train_json}")
    stage2_train_tasks = load_pubmedqa_train(args.train_json)
    print(f"[Data] Stage-2 train tasks (labeled): {len(stage2_train_tasks)}")

    # Validation data
    print(f"[Data] Loading val tasks from {args.val_json}")
    val_tasks = load_pubmedqa_train(args.val_json)
    for t in val_tasks:
        t["split"] = "val"
    print(f"[Data] Val tasks: {len(val_tasks)}")

    # DataLoaders
    stage1_loader = DataLoader(
        PubMedQADataset(stage1_train_tasks),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tasks,
    )
    stage2_loader = DataLoader(
        PubMedQADataset(stage2_train_tasks),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_tasks,
    )
    val_loader = DataLoader(
        PubMedQADataset(val_tasks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tasks,
    )

    # --------------------------------------------------
    # 6) Load frozen Qwen3 (text-only)
    # --------------------------------------------------
    tokenizer, model = load_model(args)
    model.to(device)
    base_model = model.module if hasattr(model, "module") else model

    # Add <RepRAG> special token
    num_added = tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
    if num_added > 0:
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"[Model] Added {REPRAG_TOKEN} token; vocab resized to {len(tokenizer)}.")

    # Sync config.vocab_size
    out_emb = base_model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight"):
        base_model.config.vocab_size = int(out_emb.weight.shape[0])
    else:
        base_model.config.vocab_size = int(base_model.get_input_embeddings().weight.shape[0])

    assert base_model.config.vocab_size == base_model.get_output_embeddings().weight.shape[0], (
        f"vocab mismatch: config={base_model.config.vocab_size}, "
        f"head={base_model.get_output_embeddings().weight.shape[0]}"
    )

    reprag_token_id = tokenizer.convert_tokens_to_ids(REPRAG_TOKEN)
    assert reprag_token_id != tokenizer.unk_token_id, "<RepRAG> was not added correctly."

    model.requires_grad_(False)
    model.eval()
    base_model.requires_grad_(False)
    base_model.eval()
    model_dtype = next(base_model.parameters()).dtype
    print(f"[Model] Frozen Qwen3 loaded; hidden_size={base_model.config.hidden_size}, dtype={model_dtype}.")

    # --------------------------------------------------
    # 7) MLP projector: content embedding [D_content] -> injection vector [H]
    # --------------------------------------------------
    H = base_model.config.hidden_size
    mlp = PrefixProjector(
        input_size=D_content, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1
    ).to(device)
    mlp.requires_grad_(True)

    with torch.no_grad():
        dummy = torch.zeros(2, D_content, device=device)
        out_dummy = mlp(dummy)
        assert out_dummy.shape[-1] == H, (
            f"MLP output dim must be H={H}, got {out_dummy.shape}"
        )
    print(f"[MLP] PrefixProjector: D_content={D_content} -> H={H}.")

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.stage1_lr)

    # --------------------------------------------------
    # 8) TensorBoard
    # --------------------------------------------------
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(args.tb_dir, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TB] Logging to {tb_log_dir}")

    # --------------------------------------------------
    # Helper: encode val queries on-the-fly
    # --------------------------------------------------
    def get_val_query_emb():
        print("[Eval] Encoding val queries on-the-fly ...")
        emb, q2r = encode_tasks_online(
            val_tasks,
            tokenizer=tokenizer,
            base_model=base_model,
            device=device,
            batch_size=getattr(args, "val_encode_batch_size", 8),
        )
        print(f"[Eval] Val query embeddings: {emb.shape}")
        return emb, q2r

    # --------------------------------------------------
    # 9) Stage 1 — paraphrase pretraining
    # --------------------------------------------------
    if args.stage1_pretrain:
        print(f"\n[Stage1] Starting pretraining for {args.stage1_epochs} epoch(s).")
        for g in optimizer.param_groups:
            g["lr"] = args.stage1_lr
        print(f"[Stage1] lr={args.stage1_lr}")

        mlp.train()
        global_step = 0
        ckpt_dir = args.ckpt_dir or "."
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(args.stage1_epochs):
            print(f"\n[Stage1] Epoch {epoch+1}/{args.stage1_epochs}")
            epoch_loss = 0.0

            pbar = tqdm(stage1_loader, desc=f"Stage1 E{epoch+1}", dynamic_ncols=True)
            for step_idx, batch_tasks in enumerate(pbar):
                if not batch_tasks:
                    continue

                loss_val, global_step = stage1_pretrain_step(
                    args=args,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    mlp=mlp,
                    optimizer=optimizer,
                    device=device,
                    model_dtype=model_dtype,
                    batch_tasks=batch_tasks,
                    # Use corpus question embeddings as queries — all 62,249
                    # combined pool items already have their embeddings here.
                    query_emb_normed=corpus_emb_normed,
                    query_qid_to_row=corpus_qid_to_row,
                    corpus_faiss_index=corpus_faiss_index,
                    corpus_qids=corpus_qids,
                    content_emb_map=content_emb_map,
                    corpus_task_map=corpus_task_map,
                    writer=writer,
                    global_step=global_step,
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                )
                epoch_loss += loss_val

                if global_step % args.stage1_eval_every == 0:
                    val_emb_normed, val_qid_to_row = get_val_query_emb()
                    val_loss_list = []
                    for val_batch in val_loader:
                        if not val_batch:
                            continue
                        vloss = stage1_eval_step(
                            args=args,
                            base_model=base_model,
                            tokenizer=tokenizer,
                            mlp=mlp,
                            device=device,
                            model_dtype=model_dtype,
                            batch_tasks=val_batch,
                            query_emb_normed=val_emb_normed,
                            query_qid_to_row=val_qid_to_row,
                            corpus_faiss_index=eval_corpus_faiss_index,
                            corpus_qids=eval_corpus_qids,
                            content_emb_map=eval_content_emb_map,
                            corpus_task_map=eval_corpus_task_map,
                            writer=writer,
                            global_step=global_step,
                            reprag_token_id=reprag_token_id,
                            eval_prefix="stage1/eval",
                        )
                        if vloss is not None:
                            val_loss_list.append(vloss)
                    mlp.train()
                    if val_loss_list:
                        val_mean = sum(val_loss_list) / len(val_loss_list)
                        writer.add_scalar("stage1/eval_avg_loss", val_mean, global_step)
                        pbar.write(f"[Stage1] eval_avg_loss={val_mean:.4f} @ step {global_step}")

            pbar.close()
            avg_loss = epoch_loss / max(len(stage1_loader), 1)
            print(f"[Stage1] Epoch {epoch+1} done. avg_loss={avg_loss:.4f}")
            writer.add_scalar("stage1/train_loss_epoch", avg_loss, epoch + 1)

            ckpt_path = os.path.join(ckpt_dir, f"mlp_stage1_epoch{epoch+1:03d}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "avg_loss": float(avg_loss),
                    "mlp_state": mlp.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[Stage1] Checkpoint saved → {ckpt_path}")

    # --------------------------------------------------
    # 10) Stage 2 — answer fine-tuning
    # --------------------------------------------------
    if args.stage2_finetuning:
        print(f"\n[Stage2] Starting fine-tuning for {args.stage2_epochs} epoch(s).")

        # Optionally load from stage-1 checkpoint
        if getattr(args, "stage2_init_ckpt", None):
            ckpt2 = torch.load(args.stage2_init_ckpt, map_location="cpu")
            if "mlp_state" in ckpt2:
                mlp.load_state_dict(ckpt2["mlp_state"], strict=True)
                print(f"[Stage2] MLP loaded from {args.stage2_init_ckpt}")
            if getattr(args, "stage2_load_optim", False) and "optim_state" in ckpt2:
                optimizer.load_state_dict(ckpt2["optim_state"])
                print(f"[Stage2] Optimizer state loaded from {args.stage2_init_ckpt}")
            global_step = int(ckpt2.get("global_step", 0))
            print(f"[Stage2] Resuming from global_step={global_step}")
        else:
            # Always reset to 0 for Stage 2 so eval/acc intervals are computed
            # relative to Stage 2 steps, not accumulated from Stage 1.
            global_step = 0

        for g in optimizer.param_groups:
            g["lr"] = args.stage2_lr
        print(f"[Stage2] lr={args.stage2_lr}")

        mlp.train()
        ckpt_dir = args.ckpt_dir or "."
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(args.stage2_epochs):
            total_loss_sum = 0.0
            total_acc_sum = 0.0
            total_samples = 0
            total_acc_samples = 0
            print(f"\n[Stage2] Epoch {epoch+1}/{args.stage2_epochs}")

            pbar = tqdm(stage2_loader, desc=f"Stage2 E{epoch+1}", dynamic_ncols=True)
            for step_idx, batch_tasks in enumerate(pbar):
                bs = len(batch_tasks) if batch_tasks else 0
                if bs == 0:
                    continue

                loss_val, acc_val, global_step = stage2_finetune_step(
                    args=args,
                    epoch=epoch,
                    step=step_idx,
                    global_step=global_step,
                    batch_tasks=batch_tasks,
                    device=device,
                    tokenizer=tokenizer,
                    base_model=base_model,
                    mlp=mlp,
                    optimizer=optimizer,
                    writer=writer,
                    model_dtype=model_dtype,
                    query_emb_normed=train_emb_normed,
                    query_qid_to_row=train_qid_to_row,
                    corpus_faiss_index=corpus_faiss_index,
                    corpus_qids=corpus_qids,
                    content_emb_map=content_emb_map,
                    corpus_task_map=corpus_task_map,
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                )

                if loss_val is None:
                    continue

                total_loss_sum += float(loss_val) * bs
                total_samples += bs
                if acc_val is not None:
                    total_acc_sum += float(acc_val) * bs
                    total_acc_samples += bs

                train_loss_global = total_loss_sum / max(1, total_samples)
                train_acc_global = (
                    total_acc_sum / total_acc_samples if total_acc_samples > 0 else 0.0
                )

                if global_step % args.stage2_eval_every == 0 and global_step > 0:
                    val_emb_normed, val_qid_to_row = get_val_query_emb()
                    val_loss_sum = 0.0
                    val_loss_n = 0
                    val_acc_sum = 0.0
                    val_acc_n = 0

                    for i, val_batch in enumerate(val_loader):
                        v_bs = len(val_batch) if val_batch else 0
                        if v_bs == 0:
                            continue

                        vloss, vacc = stage2_eval_step(
                            args=args,
                            epoch=epoch,
                            step=i,
                            global_step=global_step,
                            batch_tasks=val_batch,
                            device=device,
                            tokenizer=tokenizer,
                            base_model=base_model,
                            mlp=mlp,
                            writer=writer,
                            model_dtype=model_dtype,
                            query_emb_normed=val_emb_normed,
                            query_qid_to_row=val_qid_to_row,
                            corpus_faiss_index=eval_corpus_faiss_index,
                            corpus_qids=eval_corpus_qids,
                            content_emb_map=eval_content_emb_map,
                            corpus_task_map=eval_corpus_task_map,
                            reprag_token_id=reprag_token_id,
                            eval_prefix="stage2/eval",
                        )

                        if vacc is not None:
                            val_acc_sum += float(vacc) * v_bs
                            val_acc_n += v_bs
                        if vloss is not None:
                            val_loss_sum += float(vloss) * v_bs
                            val_loss_n += v_bs

                    mlp.train()
                    eval_avg_acc = val_acc_sum / max(1, val_acc_n)
                    eval_avg_loss = (
                        val_loss_sum / max(1, val_loss_n) if val_loss_n > 0 else None
                    )

                    writer.add_scalar("stage2/eval_avg_acc", float(eval_avg_acc), global_step)
                    if eval_avg_loss is not None:
                        writer.add_scalar("stage2/eval_avg_loss", float(eval_avg_loss), global_step)

                    msg = (
                        f"[Stage2][Eval] step={global_step} | "
                        f"train_loss={train_loss_global:.4f}"
                    )
                    if total_acc_samples > 0:
                        msg += f" train_acc={train_acc_global:.4f}"
                    msg += f" | eval_acc={eval_avg_acc:.4f}"
                    if eval_avg_loss is not None:
                        msg += f" eval_loss={eval_avg_loss:.4f}"
                    pbar.write(msg)

            pbar.close()
            avg_loss = total_loss_sum / max(1, total_samples)
            avg_acc = (
                total_acc_sum / total_acc_samples if total_acc_samples > 0 else 0.0
            )
            print(
                f"==== [Stage2] Epoch {epoch+1}/{args.stage2_epochs} | "
                f"Train loss: {avg_loss:.4f} | Train acc: {avg_acc:.4f} ===="
            )
            writer.add_scalar("stage2/train_loss_epoch", float(avg_loss), epoch + 1)
            if total_acc_samples > 0:
                writer.add_scalar("stage2/train_acc_epoch", float(avg_acc), epoch + 1)

            ckpt_path = os.path.join(ckpt_dir, f"mlp_stage2_epoch{epoch+1:03d}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": int(global_step),
                    "avg_loss": float(avg_loss),
                    "mlp_state": mlp.state_dict(),
                    "optim_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"[Stage2] Checkpoint saved → {ckpt_path}")

    writer.close()
    print("[Done]")


if __name__ == "__main__":
    main()
