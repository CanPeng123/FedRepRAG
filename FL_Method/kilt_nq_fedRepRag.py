"""
FedRepRag implementation for KILT-NQ MLP training.

KILT-NQ specifics vs PubMedQA:
- Same client JSONL serves both Stage 1 (corpus retrieval) and Stage 2 (CE training)
- No separate labeled/unlabeled split; all items are Q+A pairs
- Self-retrieval exclusion handled inside KILT-NQ step functions
"""
import math
import os
import torch
from collections import OrderedDict
from typing import List, Dict, Any

from tqdm.auto import tqdm

from VQA.FL_utilise.fl_helper import clone_model_state_dict, load_model_state_dict
from VQA.utilise.retrieval_architecture import PrefixProjector

from VQA.utilise.KILT_NQ.KILT_NQ_mlp_train_stage1 import (
    stage1_pretrain_step as kilt_nq_stage1_pretrain_step,
    stage1_eval_step     as kilt_nq_stage1_eval_step,
)
from VQA.utilise.KILT_NQ.KILT_NQ_mlp_train_stage2 import (
    stage2_finetune_step as kilt_nq_stage2_finetune_step,
    stage2_eval_step     as kilt_nq_stage2_eval_step,
)


# -------------------------------------------------------
# Local epoch runners
# -------------------------------------------------------

def _run_kilt_nq_stage1_local_epoch(
    args, client_id, global_mlp_state, base_model, tokenizer,
    device, model_dtype, client_pool, client_loader, writer, round_idx, reprag_token_id,
):
    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    d_content = int(client_pool["content_dim"])

    local_mlp = PrefixProjector(
        input_size=d_content, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1
    ).to(device)
    load_model_state_dict(local_mlp, global_mlp_state, device)
    local_mlp.train()
    local_mlp.requires_grad_(True)

    optimizer = torch.optim.AdamW(local_mlp.parameters(), lr=args.stage1_lr)

    local_loss_sum = 0.0
    local_steps = 0
    global_step_dummy = 0

    total_steps = args.local_epochs * len(client_loader)
    pbar_desc = f"[Stage1][Round {round_idx+1}] Client {client_id+1}"

    with tqdm(total=total_steps, desc=pbar_desc, leave=False) as pbar:
        for _ in range(args.local_epochs):
            for batch_records in client_loader:
                if not batch_records:
                    pbar.update(1)
                    continue

                loss_val, global_step_dummy = kilt_nq_stage1_pretrain_step(
                    args=args,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    mlp=local_mlp,
                    optimizer=optimizer,
                    device=device,
                    model_dtype=model_dtype,
                    batch_records=batch_records,
                    query_emb_normed=client_pool["query_emb_normed"],
                    query_id_to_row=client_pool["query_id_to_row"],
                    train_faiss_index=client_pool["train_faiss_index"],
                    train_ids=client_pool["train_ids"],
                    train_record_map=client_pool["train_record_map"],
                    content_emb_map=client_pool["content_emb_map"],
                    writer=writer,
                    global_step=global_step_dummy,
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                )

                if not math.isfinite(loss_val):
                    param_ok = all(torch.isfinite(p).all() for p in local_mlp.parameters())
                    if not param_ok:
                        print(f"[KiltNQ FedRepRag][Stage1][Round {round_idx+1}] Client {client_id+1}: "
                              f"NaN in params — restoring from global state.")
                        load_model_state_dict(local_mlp, global_mlp_state, device)
                        optimizer = torch.optim.AdamW(local_mlp.parameters(), lr=args.stage1_lr)
                    continue

                local_loss_sum += float(loss_val)
                local_steps += 1

    avg_local_loss = local_loss_sum / max(1, local_steps)
    print(f"[KiltNQ FedRepRag][Stage1][Round {round_idx+1}] Client {client_id+1} "
          f"local_loss={avg_local_loss:.4f}")
    return clone_model_state_dict(local_mlp), len(client_loader.dataset), avg_local_loss


def _run_kilt_nq_stage2_local_epoch(
    args, client_id, global_mlp_state, base_model, tokenizer,
    device, model_dtype, client_pool, client_loader, writer, round_idx, reprag_token_id,
):
    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    d_content = int(client_pool["content_dim"])

    local_mlp = PrefixProjector(
        input_size=d_content, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1
    ).to(device)
    load_model_state_dict(local_mlp, global_mlp_state, device)
    local_mlp.train()
    local_mlp.requires_grad_(True)

    optimizer = torch.optim.AdamW(local_mlp.parameters(), lr=args.stage2_lr)

    loss_sum = 0.0
    acc_sum = 0.0
    n_loss = 0
    n_acc = 0
    global_step_dummy = 0

    total_steps = args.local_epochs * len(client_loader)
    pbar_desc = f"[Stage2][Round {round_idx+1}] Client {client_id+1}"

    with tqdm(total=total_steps, desc=pbar_desc, leave=False) as pbar:
        for local_epoch in range(args.local_epochs):
            for step_idx, batch_records in enumerate(client_loader):
                bs = len(batch_records) if batch_records else 0
                if bs == 0:
                    pbar.update(1)
                    continue

                loss_val, acc_val, global_step_dummy = kilt_nq_stage2_finetune_step(
                    args=args,
                    epoch=local_epoch,
                    step=step_idx,
                    global_step=global_step_dummy,
                    batch_records=batch_records,
                    device=device,
                    tokenizer=tokenizer,
                    base_model=base_model,
                    mlp=local_mlp,
                    optimizer=optimizer,
                    writer=writer,
                    model_dtype=model_dtype,
                    query_emb_normed=client_pool["query_emb_normed"],
                    query_id_to_row=client_pool["query_id_to_row"],
                    train_faiss_index=client_pool["train_faiss_index"],
                    train_ids=client_pool["train_ids"],
                    train_record_map=client_pool["train_record_map"],
                    content_emb_map=client_pool["content_emb_map"],
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                )

                if loss_val is None:
                    continue
                loss_sum += float(loss_val) * bs
                n_loss += bs

                if acc_val is not None:
                    acc_sum += float(acc_val) * bs
                    n_acc += bs

    avg_loss = loss_sum / max(1, n_loss)
    avg_acc = acc_sum / max(1, n_acc)
    print(f"[KiltNQ FedRepRag][Stage2][Round {round_idx+1}] Client {client_id+1} "
          f"local_loss={avg_loss:.4f} local_acc={avg_acc:.4f}")
    return clone_model_state_dict(local_mlp), len(client_loader.dataset), avg_loss, avg_acc


# -------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------

def _evaluate_kilt_nq_stage1(
    args, global_mlp, val_loader, base_model, tokenizer,
    device, model_dtype, eval_pool, writer, global_step, reprag_token_id,
    eval_client_id, get_val_query_emb_fn,
):
    global_mlp.eval()
    val_loss_list = []
    val_emb_normed, val_qid_to_row = get_val_query_emb_fn()

    with tqdm(total=len(val_loader), desc=f"[Stage1][Eval][Client {eval_client_id}]", leave=False) as pbar:
        for val_batch in val_loader:
            vloss = kilt_nq_stage1_eval_step(
                args=args,
                base_model=base_model,
                tokenizer=tokenizer,
                mlp=global_mlp,
                device=device,
                model_dtype=model_dtype,
                batch_records=val_batch,
                query_emb_normed=val_emb_normed,
                query_id_to_row=val_qid_to_row,
                train_faiss_index=eval_pool["train_faiss_index"],
                train_ids=eval_pool["train_ids"],
                train_record_map=eval_pool["train_record_map"],
                content_emb_map=eval_pool["content_emb_map"],
                writer=writer,
                global_step=global_step,
                reprag_token_id=reprag_token_id,
                eval_prefix=f"stage1/eval/client_{eval_client_id}",
                pbar=pbar,
            )
            if vloss is not None:
                val_loss_list.append(float(vloss))

    global_mlp.train()
    return sum(val_loss_list) / max(1, len(val_loss_list))


def _evaluate_kilt_nq_stage2(
    args, global_mlp, val_loader, base_model, tokenizer,
    device, model_dtype, eval_pool, writer, global_step, reprag_token_id,
    eval_client_id, get_val_query_emb_fn,
):
    global_mlp.eval()
    val_loss_list = []
    val_acc_list = []
    val_emb_normed, val_qid_to_row = get_val_query_emb_fn()

    with tqdm(total=len(val_loader), desc=f"[Stage2][Eval][Client {eval_client_id}]", leave=False) as pbar:
        for i, val_batch in enumerate(val_loader):
            v_bs = len(val_batch) if val_batch else 0
            if v_bs == 0:
                pbar.update(1)
                continue

            vloss, vacc = kilt_nq_stage2_eval_step(
                args=args,
                epoch=0,
                step=i,
                global_step=global_step,
                batch_records=val_batch,
                device=device,
                tokenizer=tokenizer,
                base_model=base_model,
                mlp=global_mlp,
                writer=writer,
                model_dtype=model_dtype,
                query_emb_normed=val_emb_normed,
                query_id_to_row=val_qid_to_row,
                train_faiss_index=eval_pool["train_faiss_index"],
                train_ids=eval_pool["train_ids"],
                train_record_map=eval_pool["train_record_map"],
                content_emb_map=eval_pool["content_emb_map"],
                reprag_token_id=reprag_token_id,
                eval_prefix=f"stage2/eval/client_{eval_client_id}",
                pbar=pbar,
            )
            if vacc is not None:
                val_acc_list.append(float(vacc) * v_bs)
            if vloss is not None:
                val_loss_list.append(float(vloss) * v_bs)

    global_mlp.train()
    n = max(1, len(val_loader.dataset))
    eval_loss = sum(val_loss_list) / n if val_loss_list else None
    eval_acc  = sum(val_acc_list)  / n if val_acc_list  else 0.0
    return eval_loss, eval_acc


# -------------------------------------------------------
# FedRepRag aggregation
# -------------------------------------------------------

def fedRepRag_aggregate(local_state_dicts: List[Dict[str, torch.Tensor]], client_weights: List[int]):
    total_weight = float(sum(client_weights))
    global_state = OrderedDict()
    for k in local_state_dicts[0].keys():
        agg = None
        for local_sd, w in zip(local_state_dicts, client_weights):
            tensor = local_sd[k].float()
            if agg is None:
                agg = tensor * (w / total_weight)
            else:
                agg += tensor * (w / total_weight)
        global_state[k] = agg
    return global_state


# -------------------------------------------------------
# Public training entry points
# -------------------------------------------------------

def train_kilt_nq_fedRepRag_stage1(
    args, global_mlp, client_loaders, val_loader, base_model, tokenizer,
    device, model_dtype, client_retrieval_pools, writer, reprag_token_id,
    get_val_query_emb_fn,
):
    ckpt_dir = args.ckpt_dir or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    num_clients = len(client_loaders)

    stage1_rounds = getattr(args, "stage1_epochs", args.comm_rounds)

    for round_idx in range(stage1_rounds):
        global_step = round_idx + 1
        print(f"\n[KiltNQ FedRepRag][Stage1] Communication round {round_idx+1}/{stage1_rounds}")

        global_state = clone_model_state_dict(global_mlp)
        local_states, client_weights, local_losses = [], [], []

        for client_id, (client_loader, client_pool) in enumerate(
            zip(client_loaders, client_retrieval_pools)
        ):
            print(f"[KiltNQ FedRepRag][Stage1] Round {round_idx+1} | "
                  f"Client {client_id+1}/{num_clients} | "
                  f"n={len(client_loader.dataset)}")

            local_sd, client_n, local_loss = _run_kilt_nq_stage1_local_epoch(
                args=args,
                client_id=client_id,
                global_mlp_state=global_state,
                base_model=base_model,
                tokenizer=tokenizer,
                device=device,
                model_dtype=model_dtype,
                client_pool=client_pool,
                client_loader=client_loader,
                writer=writer,
                round_idx=round_idx,
                reprag_token_id=reprag_token_id,
            )

            local_states.append(local_sd)
            client_weights.append(client_n)
            local_losses.append(local_loss)

        print(f"[KiltNQ FedRepRag][Stage1] Round {round_idx+1} | Aggregating {num_clients} clients")
        new_global_state = fedRepRag_aggregate(local_states, client_weights)
        load_model_state_dict(global_mlp, new_global_state, device)

        train_loss_round = sum(l * w for l, w in zip(local_losses, client_weights)) / max(1, sum(client_weights))

        eval_client_id = int(getattr(args, "eval_client_id", 0))
        if eval_client_id == -1:
            val_loss_list = []
            for cid, client_pool in enumerate(client_retrieval_pools):
                vloss = _evaluate_kilt_nq_stage1(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, tokenizer=tokenizer, device=device,
                    model_dtype=model_dtype, eval_pool=client_pool, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id=cid, get_val_query_emb_fn=get_val_query_emb_fn,
                )
                val_loss_list.append(vloss)
            val_loss = sum(val_loss_list) / max(1, len(val_loss_list))
        else:
            if not (0 <= eval_client_id < num_clients):
                raise ValueError(f"eval_client_id={eval_client_id} out of range for {num_clients} clients")
            val_loss = _evaluate_kilt_nq_stage1(
                args=args, global_mlp=global_mlp, val_loader=val_loader,
                base_model=base_model, tokenizer=tokenizer, device=device,
                model_dtype=model_dtype, eval_pool=client_retrieval_pools[eval_client_id],
                writer=writer, global_step=global_step, reprag_token_id=reprag_token_id,
                eval_client_id=eval_client_id, get_val_query_emb_fn=get_val_query_emb_fn,
            )

        writer.add_scalar("fedRepRag_stage1/train_loss_round", float(train_loss_round), round_idx + 1)
        writer.add_scalar("fedRepRag_stage1/val_loss_round", float(val_loss), round_idx + 1)
        print(f"[KiltNQ FedRepRag][Stage1] Round {round_idx+1} | "
              f"train_loss={train_loss_round:.4f} | val_loss={val_loss:.4f}")

        ckpt_path = os.path.join(ckpt_dir, f"fedRepRag_stage1_round_{round_idx+1:03d}.pt")
        torch.save({
            "round": round_idx + 1,
            "global_step": global_step,
            "mlp_state": global_mlp.state_dict(),
            "train_loss_round": float(train_loss_round),
            "val_loss_round": float(val_loss),
            "eval_client_id": eval_client_id,
        }, ckpt_path)
        print(f"[KiltNQ FedRepRag][Stage1] Saved checkpoint → {ckpt_path}")


def train_kilt_nq_fedRepRag_stage2(
    args, global_mlp, client_loaders, val_loader, base_model, tokenizer,
    device, model_dtype, client_retrieval_pools, writer, reprag_token_id,
    get_val_query_emb_fn,
):
    ckpt_dir = args.ckpt_dir or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    num_clients = len(client_loaders)

    stage2_rounds = getattr(args, "stage2_epochs", args.comm_rounds)

    for round_idx in range(stage2_rounds):
        global_step = round_idx + 1
        print(f"\n[KiltNQ FedRepRag][Stage2] Communication round {round_idx+1}/{stage2_rounds}")

        global_state = clone_model_state_dict(global_mlp)
        local_states, client_weights, local_losses, local_accs = [], [], [], []

        for client_id, (client_loader, client_pool) in enumerate(
            zip(client_loaders, client_retrieval_pools)
        ):
            print(f"[KiltNQ FedRepRag][Stage2] Round {round_idx+1} | "
                  f"Client {client_id+1}/{num_clients} | "
                  f"n={len(client_loader.dataset)}")

            local_sd, client_n, local_loss, local_acc = _run_kilt_nq_stage2_local_epoch(
                args=args,
                client_id=client_id,
                global_mlp_state=global_state,
                base_model=base_model,
                tokenizer=tokenizer,
                device=device,
                model_dtype=model_dtype,
                client_pool=client_pool,
                client_loader=client_loader,
                writer=writer,
                round_idx=round_idx,
                reprag_token_id=reprag_token_id,
            )

            local_states.append(local_sd)
            client_weights.append(client_n)
            local_losses.append(local_loss)
            local_accs.append(local_acc)

        print(f"[KiltNQ FedRepRag][Stage2] Round {round_idx+1} | Aggregating {num_clients} clients")
        new_global_state = fedRepRag_aggregate(local_states, client_weights)
        load_model_state_dict(global_mlp, new_global_state, device)

        train_loss_round = sum(l * w for l, w in zip(local_losses, client_weights)) / max(1, sum(client_weights))
        train_acc_round  = sum(a * w for a, w in zip(local_accs,   client_weights)) / max(1, sum(client_weights))

        eval_client_id = int(getattr(args, "eval_client_id", 0))
        if eval_client_id == -1:
            eval_loss_list, eval_acc_list = [], []
            for cid, client_pool in enumerate(client_retrieval_pools):
                el, ea = _evaluate_kilt_nq_stage2(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, tokenizer=tokenizer, device=device,
                    model_dtype=model_dtype, eval_pool=client_pool, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id=cid, get_val_query_emb_fn=get_val_query_emb_fn,
                )
                if el is not None:
                    eval_loss_list.append(el)
                eval_acc_list.append(ea)
            eval_loss = sum(eval_loss_list) / max(1, len(eval_loss_list)) if eval_loss_list else None
            eval_acc  = sum(eval_acc_list)  / max(1, len(eval_acc_list))
        else:
            if not (0 <= eval_client_id < num_clients):
                raise ValueError(f"eval_client_id={eval_client_id} out of range for {num_clients} clients")
            eval_loss, eval_acc = _evaluate_kilt_nq_stage2(
                args=args, global_mlp=global_mlp, val_loader=val_loader,
                base_model=base_model, tokenizer=tokenizer, device=device,
                model_dtype=model_dtype, eval_pool=client_retrieval_pools[eval_client_id],
                writer=writer, global_step=global_step, reprag_token_id=reprag_token_id,
                eval_client_id=eval_client_id, get_val_query_emb_fn=get_val_query_emb_fn,
            )

        writer.add_scalar("fedRepRag_stage2/train_loss_round", float(train_loss_round), round_idx + 1)
        writer.add_scalar("fedRepRag_stage2/train_acc_round",  float(train_acc_round),  round_idx + 1)
        if eval_loss is not None:
            writer.add_scalar("fedRepRag_stage2/eval_loss_round", float(eval_loss), round_idx + 1)
        writer.add_scalar("fedRepRag_stage2/eval_acc_round", float(eval_acc), round_idx + 1)

        msg = (f"[KiltNQ FedRepRag][Stage2] Round {round_idx+1} | "
               f"train_loss={train_loss_round:.4f} | train_acc={train_acc_round:.4f} | "
               f"eval_acc={eval_acc:.4f}")
        if eval_loss is not None:
            msg += f" | eval_loss={eval_loss:.4f}"
        print(msg)

        ckpt_path = os.path.join(ckpt_dir, f"fedRepRag_stage2_round_{round_idx+1:03d}.pt")
        torch.save({
            "round": round_idx + 1,
            "global_step": global_step,
            "mlp_state": global_mlp.state_dict(),
            "train_loss_round": float(train_loss_round),
            "train_acc_round":  float(train_acc_round),
            "eval_acc_round":   float(eval_acc),
            "eval_loss_round":  float(eval_loss) if eval_loss is not None else None,
            "eval_client_id":   eval_client_id,
        }, ckpt_path)
        print(f"[KiltNQ FedRepRag][Stage2] Saved checkpoint → {ckpt_path}")