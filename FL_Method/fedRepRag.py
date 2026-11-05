import math
import os
import torch
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

from VQA.FL_utilise.fl_helper import clone_model_state_dict, load_model_state_dict
from VQA.FL_utilise.fl_helper import run_stage1_local_epoch, run_stage2_local_epoch
from VQA.FL_utilise.fl_helper import evaluate_stage1_global, evaluate_stage2_global

# PubMedQA-specific step functions (text-only, tokenizer-based)
from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage1 import (
    stage1_pretrain_step as pubmedqa_stage1_pretrain_step,
    stage1_eval_step     as pubmedqa_stage1_eval_step,
)
from VQA.utilise.PubMedQA.PubMedQA_mlp_train_stage2 import (
    stage2_finetune_step as pubmedqa_stage2_finetune_step,
    stage2_eval_step     as pubmedqa_stage2_eval_step,
)
from VQA.utilise.retrieval_architecture import PrefixProjector
from tqdm.auto import tqdm


# -------------------------------------------------------
# PubMedQA local-epoch helpers (bypass fl_helper)
# -------------------------------------------------------

def _run_pubmedqa_stage1_local_epoch(args, client_id, global_mlp_state, base_model, tokenizer,
                                    device, model_dtype, client_pool, client_loader, client_task_map,
                                    writer, round_idx, reprag_token_id,
                                    c_global_state=None, c_local_state=None):
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
            for batch_tasks in client_loader:
                if not batch_tasks:
                    pbar.update(1)
                    continue

                loss_val, global_step_dummy = pubmedqa_stage1_pretrain_step(
                    args=args,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    mlp=local_mlp,
                    optimizer=optimizer,
                    device=device,
                    model_dtype=model_dtype,
                    batch_tasks=batch_tasks,
                    query_emb_normed=client_pool["corpus_emb_normed"],
                    query_qid_to_row=client_pool["corpus_qid_to_row"],
                    corpus_faiss_index=client_pool["corpus_faiss_index"],
                    corpus_qids=client_pool["corpus_qids"],
                    content_emb_map=client_pool["content_emb_map"],
                    corpus_task_map=client_pool["corpus_task_map"],
                    writer=writer,
                    global_step=global_step_dummy,
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                    c_global_state=c_global_state,
                    c_local_state=c_local_state,
                )

                if not math.isfinite(loss_val):
                    print(
                        f"[FedRepRag][Stage1][Round {round_idx+1}] Client {client_id+1}: "
                        f"non-finite loss skipped in accumulation."
                    )
                    # Restore MLP to last good state if params are corrupted
                    param_ok = all(
                        torch.isfinite(p).all()
                        for p in local_mlp.parameters()
                    )
                    if not param_ok:
                        print(
                            f"[FedRepRag][Stage1][Round {round_idx+1}] Client {client_id+1}: "
                            f"NaN detected in MLP params — restoring from global state."
                        )
                        load_model_state_dict(local_mlp, global_mlp_state, device)
                        optimizer = torch.optim.AdamW(local_mlp.parameters(), lr=args.stage1_lr)
                    continue

                local_loss_sum += float(loss_val)
                local_steps += 1

    avg_local_loss = local_loss_sum / max(1, local_steps)
    print(f"[FedRepRag][Stage1][Round {round_idx+1}] Client {client_id+1} local_loss={avg_local_loss:.4f}")
    return clone_model_state_dict(local_mlp), len(client_loader.dataset), avg_local_loss


def _run_pubmedqa_stage2_local_epoch(args, client_id, global_mlp_state, base_model, tokenizer,
                                    device, model_dtype, client_pool, client_loader, client_task_map,
                                    writer, round_idx, reprag_token_id,
                                    c_global_state=None, c_local_state=None):
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
    loss_n = 0
    acc_sum = 0.0
    acc_n = 0
    global_step_dummy = 0

    total_steps = args.local_epochs * len(client_loader)
    pbar_desc = f"[Stage2][Round {round_idx+1}] Client {client_id+1}"

    with tqdm(total=total_steps, desc=pbar_desc, leave=False) as pbar:
        for local_epoch in range(args.local_epochs):
            for step_idx, batch_tasks in enumerate(client_loader):
                bs = len(batch_tasks) if batch_tasks is not None else 0
                if bs == 0:
                    pbar.update(1)
                    continue

                loss_val, acc_val, global_step_dummy = pubmedqa_stage2_finetune_step(
                    args=args,
                    epoch=local_epoch,
                    step=step_idx,
                    global_step=global_step_dummy,
                    batch_tasks=batch_tasks,
                    device=device,
                    tokenizer=tokenizer,
                    base_model=base_model,
                    mlp=local_mlp,
                    optimizer=optimizer,
                    writer=writer,
                    model_dtype=model_dtype,
                    query_emb_normed=client_pool["train_query_emb_normed"],
                    query_qid_to_row=client_pool["train_query_qid_to_row"],
                    corpus_faiss_index=client_pool["corpus_faiss_index"],
                    corpus_qids=client_pool["corpus_qids"],
                    content_emb_map=client_pool["content_emb_map"],
                    corpus_task_map=client_pool["corpus_task_map"],
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                    c_global_state=c_global_state,
                    c_local_state=c_local_state,
                )

                if loss_val is not None:
                    loss_sum += float(loss_val) * bs
                    loss_n += bs
                if acc_val is not None:
                    acc_sum += float(acc_val) * bs
                    acc_n += bs

    avg_loss = loss_sum / max(1, loss_n)
    avg_acc = acc_sum / max(1, acc_n)
    print(
        f"[FedRepRag][Stage2][Round {round_idx+1}] Client {client_id+1} "
        f"local_loss={avg_loss:.4f} local_acc={avg_acc:.4f}"
    )
    return clone_model_state_dict(local_mlp), len(client_loader.dataset), avg_loss, avg_acc


def _evaluate_pubmedqa_stage1(args, global_mlp, val_loader, base_model, tokenizer,
                              device, model_dtype, eval_pool, writer, global_step,
                              reprag_token_id, eval_client_id, get_val_query_emb_fn=None):
    global_mlp.eval()
    val_loss_list = []

    val_emb_normed, val_qid_to_row = get_val_query_emb_fn()

    with tqdm(total=len(val_loader), desc=f"[Stage1][Eval][Client {eval_client_id}]", leave=False) as pbar:
        for val_batch in val_loader:
            vloss = pubmedqa_stage1_eval_step(
                args=args,
                base_model=base_model,
                tokenizer=tokenizer,
                mlp=global_mlp,
                device=device,
                model_dtype=model_dtype,
                batch_tasks=val_batch,
                query_emb_normed=val_emb_normed,
                query_qid_to_row=val_qid_to_row,
                corpus_faiss_index=eval_pool["corpus_faiss_index"],
                corpus_qids=eval_pool["corpus_qids"],
                content_emb_map=eval_pool["content_emb_map"],
                corpus_task_map=eval_pool["corpus_task_map"],
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


def _evaluate_pubmedqa_stage2(args, global_mlp, val_loader, base_model, tokenizer,
                              device, model_dtype, eval_pool, writer, global_step,
                              reprag_token_id, eval_client_id, get_val_query_emb_fn=None):
    global_mlp.eval()
    val_loss_list = []
    val_acc_list = []

    val_emb_normed, val_qid_to_row = get_val_query_emb_fn()

    with tqdm(total=len(val_loader), desc=f"[Stage2][Eval][Client {eval_client_id}]", leave=False) as pbar:
        for i, val_batch in enumerate(val_loader):
            v_bs = len(val_batch) if val_batch is not None else 0
            if v_bs == 0:
                pbar.update(1)
                continue

            vloss, vacc = pubmedqa_stage2_eval_step(
                args=args,
                epoch=0,
                step=i,
                global_step=global_step,
                batch_tasks=val_batch,
                device=device,
                tokenizer=tokenizer,
                base_model=base_model,
                mlp=global_mlp,
                writer=writer,
                model_dtype=model_dtype,
                query_emb_normed=val_emb_normed,
                query_qid_to_row=val_qid_to_row,
                corpus_faiss_index=eval_pool["corpus_faiss_index"],
                corpus_qids=eval_pool["corpus_qids"],
                content_emb_map=eval_pool["content_emb_map"],
                corpus_task_map=eval_pool["corpus_task_map"],
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
# Public FedRepRag entry points
# -------------------------------------------------------

def train_fedRepRag_stage1(args, global_mlp, client_loaders, client_task_maps, val_loader, base_model,
                         device, model_dtype, client_retrieval_pools, writer, reprag_token_id,
                         qwen_model=None, processor=None, tokenizer=None, get_val_query_emb_fn=None,
                         eval_retrieval_pool=None):

    is_pubmedqa = tokenizer is not None

    ckpt_dir = args.ckpt_dir or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    num_clients = len(client_loaders)

    if len(client_retrieval_pools) != num_clients:
        raise ValueError(
            f"Mismatch: {len(client_retrieval_pools)} retrieval pools vs {num_clients} clients"
        )

    stage1_rounds = getattr(args, "stage1_epochs", args.comm_rounds)
    start_round = int(getattr(args, "stage1_start_round", 0))
    if start_round > 0:
        print(f"[FedRepRag][Stage1] Resuming from round {start_round + 1}/{stage1_rounds}")
    for round_idx in range(start_round, stage1_rounds):
        global_step = round_idx + 1
        print(f"\n[FedRepRag][Stage1] Communication round {round_idx+1}/{stage1_rounds}")

        global_state = clone_model_state_dict(global_mlp)
        local_states = []
        client_weights = []
        local_losses = []

        for client_id, (client_loader, client_task_map, client_pool) in enumerate(
            zip(client_loaders, client_task_maps, client_retrieval_pools)
        ):
            print(
                f"[FedRepRag][Stage1] Round {round_idx+1}/{stage1_rounds} | "
                f"Starting client {client_id+1}/{num_clients} | "
                f"name={client_pool.get('client_name', f'client_{client_id}')} | "
                f"num_samples={len(client_loader.dataset)}"
            )

            if is_pubmedqa:
                local_sd, client_n, local_loss = _run_pubmedqa_stage1_local_epoch(
                    args=args,
                    client_id=client_id,
                    global_mlp_state=global_state,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    device=device,
                    model_dtype=model_dtype,
                    client_pool=client_pool,
                    client_loader=client_loader,
                    client_task_map=client_task_map,
                    writer=writer,
                    round_idx=round_idx,
                    reprag_token_id=reprag_token_id,
                )
            else:
                local_sd, client_n, local_loss = run_stage1_local_epoch(
                    args=args,
                    client_id=client_id,
                    global_mlp_state=global_state,
                    base_model=base_model,
                    qwen_model=qwen_model,
                    processor=processor,
                    device=device,
                    model_dtype=model_dtype,
                    train_emb_normed=client_pool["train_emb_normed"],
                    train_qids=client_pool["train_qids"],
                    qid_to_row=client_pool["qid_to_row"],
                    index=client_pool["index"],
                    client_loader=client_loader,
                    client_task_map=client_task_map,
                    writer=writer,
                    round_idx=round_idx,
                    reprag_token_id=reprag_token_id,
                )

            print(
                f"[FedRepRag][Stage1] Round {round_idx+1}/{stage1_rounds} | "
                f"Finished client {client_id+1}/{num_clients} | "
                f"local_loss={local_loss:.4f} | client_n={client_n}"
            )

            local_states.append(local_sd)
            client_weights.append(client_n)
            local_losses.append(local_loss)

        print(f"[FedRepRag][Stage1] Round {round_idx+1} | Aggregating {num_clients} clients")
        new_global_state = fedRepRag_aggregate(local_states, client_weights)
        load_model_state_dict(global_mlp, new_global_state, device)

        train_loss_round = sum(l * w for l, w in zip(local_losses, client_weights)) / max(1, sum(client_weights))

        eval_client_id = int(getattr(args, "eval_client_id", 0))

        # Use the dedicated eval pool if provided; otherwise fall back to a client pool.
        if eval_retrieval_pool is not None:
            effective_eval_pool = eval_retrieval_pool
            effective_eval_client_id = -2  # sentinel: global eval pool
        else:
            effective_eval_client_id = eval_client_id

        if eval_retrieval_pool is not None:
            if is_pubmedqa:
                val_loss = _evaluate_pubmedqa_stage1(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, tokenizer=tokenizer, device=device,
                    model_dtype=model_dtype, eval_pool=effective_eval_pool, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id="global", get_val_query_emb_fn=get_val_query_emb_fn,
                )
            else:
                val_loss = 0.0  # non-PubMedQA path unchanged

        elif eval_client_id == -1:
            print(f"[FedRepRag][Stage1] Round {round_idx+1} | Evaluating on all {num_clients} client retrieval banks")
            val_loss_list = []

            for client_id, (client_pool, client_task_map) in enumerate(zip(client_retrieval_pools, client_task_maps)):
                print(
                    f"[FedRepRag][Stage1] Round {round_idx+1} | "
                    f"Evaluating client {client_id+1}/{num_clients} "
                    f"(client_id={client_id}, name={client_pool.get('client_name', f'client_{client_id}')})"
                )

                if is_pubmedqa:
                    vloss = _evaluate_pubmedqa_stage1(
                        args=args, global_mlp=global_mlp, val_loader=val_loader,
                        base_model=base_model, tokenizer=tokenizer, device=device,
                        model_dtype=model_dtype, eval_pool=client_pool, writer=writer,
                        global_step=global_step, reprag_token_id=reprag_token_id,
                        eval_client_id=client_id, get_val_query_emb_fn=get_val_query_emb_fn,
                    )
                else:
                    vloss = evaluate_stage1_global(
                        args=args, global_mlp=global_mlp, val_loader=val_loader,
                        base_model=base_model, qwen_model=qwen_model, processor=processor,
                        device=device, model_dtype=model_dtype,
                        train_emb_normed=client_pool["train_emb_normed"],
                        train_qids=client_pool["train_qids"],
                        qid_to_row=client_pool["qid_to_row"], index=client_pool["index"],
                        train_task_map=client_task_map, writer=writer,
                        global_step=global_step, reprag_token_id=reprag_token_id,
                        eval_client_id=client_id,
                    )
                val_loss_list.append(vloss)

            val_loss = sum(val_loss_list) / max(1, len(val_loss_list))

        else:
            if not (0 <= eval_client_id < num_clients):
                raise ValueError(f"eval_client_id={eval_client_id} is out of range for {num_clients} clients")

            eval_pool = client_retrieval_pools[eval_client_id]
            eval_task_map = client_task_maps[eval_client_id]

            print(
                f"[FedRepRag][Stage1] Round {round_idx+1} | "
                f"Evaluating on client {eval_client_id+1}/{num_clients} "
                f"(client_id={eval_client_id}, name={eval_pool.get('client_name', f'client_{eval_client_id}')})"
            )

            if is_pubmedqa:
                val_loss = _evaluate_pubmedqa_stage1(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, tokenizer=tokenizer, device=device,
                    model_dtype=model_dtype, eval_pool=eval_pool, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id=eval_client_id, get_val_query_emb_fn=get_val_query_emb_fn,
                )
            else:
                val_loss = evaluate_stage1_global(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, qwen_model=qwen_model, processor=processor,
                    device=device, model_dtype=model_dtype,
                    train_emb_normed=eval_pool["train_emb_normed"],
                    train_qids=eval_pool["train_qids"],
                    qid_to_row=eval_pool["qid_to_row"], index=eval_pool["index"],
                    train_task_map=eval_task_map, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id=eval_client_id,
                )

        writer.add_scalar("fedRepRag_stage1/train_loss_round", float(train_loss_round), round_idx + 1)
        writer.add_scalar("fedRepRag_stage1/val_loss_round", float(val_loss), round_idx + 1)

        print(
            f"[FedRepRag][Stage1] Round {round_idx+1} | "
            f"train_loss={train_loss_round:.4f} | val_loss={val_loss:.4f}"
        )

        ckpt_path = os.path.join(ckpt_dir, f"fedRepRag_stage1_round_{round_idx+1:03d}.pt")
        torch.save(
            {
                "round": round_idx + 1,
                "global_step": global_step,
                "mlp_state": global_mlp.state_dict(),
                "train_loss_round": float(train_loss_round),
                "val_loss_round": float(val_loss),
                "eval_client_id": eval_client_id,
            },
            ckpt_path
        )
        print(f"[FedRepRag][Stage1] Saved checkpoint to {ckpt_path}")


def train_fedRepRag_stage2(args, global_mlp, client_loaders, client_task_maps, val_loader, base_model,
                         device, model_dtype, client_retrieval_pools, writer, reprag_token_id,
                         qwen_model=None, processor=None, tokenizer=None, get_val_query_emb_fn=None,
                         eval_retrieval_pool=None):

    is_pubmedqa = tokenizer is not None

    ckpt_dir = args.ckpt_dir or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    num_clients = len(client_loaders)

    if len(client_retrieval_pools) != num_clients:
        raise ValueError(
            f"Mismatch: {len(client_retrieval_pools)} retrieval pools vs {num_clients} clients"
        )

    stage2_rounds = getattr(args, "stage2_epochs", args.comm_rounds)
    for round_idx in range(stage2_rounds):
        global_step = round_idx + 1
        print(f"\n[FedRepRag][Stage2] Communication round {round_idx+1}/{stage2_rounds}")

        global_state = clone_model_state_dict(global_mlp)
        local_states = []
        client_weights = []
        local_losses = []
        local_accs = []

        for client_id, (client_loader, client_task_map, client_pool) in enumerate(
            zip(client_loaders, client_task_maps, client_retrieval_pools)
        ):
            print(
                f"[FedRepRag][Stage2] Round {round_idx+1}/{stage2_rounds} | "
                f"Starting client {client_id+1}/{num_clients} | "
                f"name={client_pool.get('client_name', f'client_{client_id}')} | "
                f"num_samples={len(client_loader.dataset)}"
            )

            if is_pubmedqa:
                local_sd, client_n, local_loss, local_acc = _run_pubmedqa_stage2_local_epoch(
                    args=args,
                    client_id=client_id,
                    global_mlp_state=global_state,
                    base_model=base_model,
                    tokenizer=tokenizer,
                    device=device,
                    model_dtype=model_dtype,
                    client_pool=client_pool,
                    client_loader=client_loader,
                    client_task_map=client_task_map,
                    writer=writer,
                    round_idx=round_idx,
                    reprag_token_id=reprag_token_id,
                )
            else:
                local_sd, client_n, local_loss, local_acc = run_stage2_local_epoch(
                    args=args,
                    client_id=client_id,
                    global_mlp_state=global_state,
                    base_model=base_model,
                    qwen_model=qwen_model,
                    processor=processor,
                    device=device,
                    model_dtype=model_dtype,
                    train_emb_normed=client_pool["train_emb_normed"],
                    train_qids=client_pool["train_qids"],
                    qid_to_row=client_pool["qid_to_row"],
                    index=client_pool["index"],
                    client_loader=client_loader,
                    client_task_map=client_task_map,
                    writer=writer,
                    round_idx=round_idx,
                    reprag_token_id=reprag_token_id,
                )

            print(
                f"[FedRepRag][Stage2] Round {round_idx+1}/{stage2_rounds} | "
                f"Finished client {client_id+1}/{num_clients} | "
                f"local_loss={local_loss:.4f} | local_acc={local_acc:.4f} | client_n={client_n}"
            )

            local_states.append(local_sd)
            client_weights.append(client_n)
            local_losses.append(local_loss)
            local_accs.append(local_acc)

        print(f"[FedRepRag][Stage2] Round {round_idx+1} | Aggregating {num_clients} clients")
        new_global_state = fedRepRag_aggregate(local_states, client_weights)
        load_model_state_dict(global_mlp, new_global_state, device)

        train_loss_round = sum(l * w for l, w in zip(local_losses, client_weights)) / max(1, sum(client_weights))
        train_acc_round  = sum(a * w for a, w in zip(local_accs,   client_weights)) / max(1, sum(client_weights))

        eval_client_id = int(getattr(args, "eval_client_id", 0))

        # Use the dedicated global eval pool if provided (test items available for retrieval,
        # but were excluded from all training corpora).
        if eval_retrieval_pool is not None:
            if is_pubmedqa:
                eval_loss, eval_acc = _evaluate_pubmedqa_stage2(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, tokenizer=tokenizer, device=device,
                    model_dtype=model_dtype, eval_pool=eval_retrieval_pool, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id="global", get_val_query_emb_fn=get_val_query_emb_fn,
                )
            else:
                eval_loss, eval_acc = None, 0.0  # non-PubMedQA path unchanged

        elif eval_client_id == -1:
            print(f"[FedRepRag][Stage2] Round {round_idx+1} | Evaluating on all {num_clients} client retrieval banks")
            eval_loss_list = []
            eval_acc_list = []

            for client_id, (client_pool, client_task_map) in enumerate(zip(client_retrieval_pools, client_task_maps)):
                print(
                    f"[FedRepRag][Stage2] Round {round_idx+1} | "
                    f"Evaluating client {client_id+1}/{num_clients} "
                    f"(client_id={client_id}, name={client_pool.get('client_name', f'client_{client_id}')})"
                )

                if is_pubmedqa:
                    eval_loss_i, eval_acc_i = _evaluate_pubmedqa_stage2(
                        args=args, global_mlp=global_mlp, val_loader=val_loader,
                        base_model=base_model, tokenizer=tokenizer, device=device,
                        model_dtype=model_dtype, eval_pool=client_pool, writer=writer,
                        global_step=global_step, reprag_token_id=reprag_token_id,
                        eval_client_id=client_id, get_val_query_emb_fn=get_val_query_emb_fn,
                    )
                else:
                    eval_loss_i, eval_acc_i = evaluate_stage2_global(
                        args=args, global_mlp=global_mlp, val_loader=val_loader,
                        base_model=base_model, qwen_model=qwen_model, processor=processor,
                        device=device, model_dtype=model_dtype,
                        train_emb_normed=client_pool["train_emb_normed"],
                        train_qids=client_pool["train_qids"],
                        qid_to_row=client_pool["qid_to_row"], index=client_pool["index"],
                        train_task_map=client_task_map, writer=writer,
                        global_step=global_step, reprag_token_id=reprag_token_id,
                        eval_client_id=client_id,
                    )

                if eval_loss_i is not None:
                    eval_loss_list.append(eval_loss_i)
                eval_acc_list.append(eval_acc_i)

            eval_loss = sum(eval_loss_list) / max(1, len(eval_loss_list)) if eval_loss_list else None
            eval_acc  = sum(eval_acc_list)  / max(1, len(eval_acc_list))

        else:
            if not (0 <= eval_client_id < num_clients):
                raise ValueError(f"eval_client_id={eval_client_id} is out of range for {num_clients} clients")

            eval_pool = client_retrieval_pools[eval_client_id]
            eval_task_map = client_task_maps[eval_client_id]

            print(
                f"[FedRepRag][Stage2] Round {round_idx+1} | "
                f"Evaluating on client {eval_client_id+1}/{num_clients} "
                f"(client_id={eval_client_id}, name={eval_pool.get('client_name', f'client_{eval_client_id}')})"
            )

            if is_pubmedqa:
                eval_loss, eval_acc = _evaluate_pubmedqa_stage2(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, tokenizer=tokenizer, device=device,
                    model_dtype=model_dtype, eval_pool=eval_pool, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id=eval_client_id, get_val_query_emb_fn=get_val_query_emb_fn,
                )
            else:
                eval_loss, eval_acc = evaluate_stage2_global(
                    args=args, global_mlp=global_mlp, val_loader=val_loader,
                    base_model=base_model, qwen_model=qwen_model, processor=processor,
                    device=device, model_dtype=model_dtype,
                    train_emb_normed=eval_pool["train_emb_normed"],
                    train_qids=eval_pool["train_qids"],
                    qid_to_row=eval_pool["qid_to_row"], index=eval_pool["index"],
                    train_task_map=eval_task_map, writer=writer,
                    global_step=global_step, reprag_token_id=reprag_token_id,
                    eval_client_id=eval_client_id,
                )

        writer.add_scalar("fedRepRag_stage2/train_loss_round", float(train_loss_round), round_idx + 1)
        writer.add_scalar("fedRepRag_stage2/train_acc_round",  float(train_acc_round),  round_idx + 1)
        if eval_loss is not None:
            writer.add_scalar("fedRepRag_stage2/eval_loss_round", float(eval_loss), round_idx + 1)
        writer.add_scalar("fedRepRag_stage2/eval_acc_round", float(eval_acc), round_idx + 1)

        msg = (
            f"[FedRepRag][Stage2] Round {round_idx+1} | "
            f"train_loss={train_loss_round:.4f} | train_acc={train_acc_round:.4f} | "
            f"eval_acc={eval_acc:.4f}"
        )
        if eval_loss is not None:
            msg += f" | eval_loss={eval_loss:.4f}"
        print(msg)

        ckpt_path = os.path.join(ckpt_dir, f"fedRepRag_stage2_round_{round_idx+1:03d}.pt")
        torch.save(
            {
                "round": round_idx + 1,
                "global_step": global_step,
                "mlp_state": global_mlp.state_dict(),
                "train_loss_round": float(train_loss_round),
                "train_acc_round":  float(train_acc_round),
                "eval_acc_round":   float(eval_acc),
                "eval_loss_round":  float(eval_loss) if eval_loss is not None else None,
                "eval_client_id":   eval_client_id,
            },
            ckpt_path
        )
        print(f"[FedRepRag][Stage2] Saved checkpoint to {ckpt_path}")


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