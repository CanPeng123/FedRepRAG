import torch
import torch.nn as nn
from typing import Dict
from tqdm.auto import tqdm

from VQA.utilise.retrieval_architecture import PrefixProjector

# =========================
# Dataset-specific imports
# =========================
from VQA.utilise.ScienceQA.ScienceQA_mlp_train_stage1 import (
    stage1_pretrain_step as scienceqa_stage1_pretrain_step,
    stage1_eval_step as scienceqa_stage1_eval_step,
)
from VQA.utilise.ScienceQA.ScienceQA_mlp_train_stage2 import (
    stage2_finetune_step as scienceqa_stage2_finetune_step,
    stage2_eval_step as scienceqa_stage2_eval_step,
)

from VQA.utilise.SLAKE.Slake_mlp_train_stage1 import (
    stage1_pretrain_step as slake_stage1_pretrain_step,
    stage1_eval_step as slake_stage1_eval_step,
)
from VQA.utilise.SLAKE.Slake_mlp_train_stage2 import (
    stage2_finetune_step as slake_stage2_finetune_step,
    stage2_eval_step as slake_stage2_eval_step,
)


def get_dataset_step_functions(dataset_name: str):
    """
    Return dataset-specific train/eval step functions according to args.dataset.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "scienceqa":
        return {
            "stage1_pretrain_step": scienceqa_stage1_pretrain_step,
            "stage1_eval_step": scienceqa_stage1_eval_step,
            "stage2_finetune_step": scienceqa_stage2_finetune_step,
            "stage2_eval_step": scienceqa_stage2_eval_step,
        }
    elif dataset_name == "slake":
        return {
            "stage1_pretrain_step": slake_stage1_pretrain_step,
            "stage1_eval_step": slake_stage1_eval_step,
            "stage2_finetune_step": slake_stage2_finetune_step,
            "stage2_eval_step": slake_stage2_eval_step,
        }
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets are: ['ScienceQA', 'SLAKE']"
        )


def clone_model_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], device: torch.device):
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()}, strict=True)


def run_stage1_local_epoch(args, client_id, global_mlp_state, base_model, qwen_model, processor, device, model_dtype, train_emb_normed, 
train_qids, qid_to_row, index, client_loader, client_task_map, writer, round_idx, reprag_token_id):
    step_fns = get_dataset_step_functions(args.dataset)
    stage1_pretrain_step = step_fns["stage1_pretrain_step"]

    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    local_mlp = PrefixProjector(
        input_size=H,
        output_size=H,
        projector_type="mlp2x_gelu",
        p_drop=0.1
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
        for local_epoch in range(args.local_epochs):
            for step_idx, batch_tasks in enumerate(client_loader):
                if not batch_tasks:
                    pbar.update(1)
                    continue

                loss_val, global_step_dummy = stage1_pretrain_step(
                    args=args,
                    base_model=base_model,
                    qwen_model=qwen_model,
                    processor=processor,
                    mlp=local_mlp,
                    optimizer=optimizer,
                    device=device,
                    model_dtype=model_dtype,
                    batch_tasks=batch_tasks,
                    train_emb_normed=train_emb_normed,
                    train_qids=train_qids,
                    qid_to_row=qid_to_row,
                    index=index,
                    train_task_map=client_task_map,
                    writer=writer,
                    global_step=global_step_dummy,
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                )

                local_loss_sum += float(loss_val)
                local_steps += 1

    avg_local_loss = local_loss_sum / max(1, local_steps)
    print(f"[FedRepRag][Stage1][Round {round_idx+1}] Client {client_id+1} local_loss={avg_local_loss:.4f}")

    return clone_model_state_dict(local_mlp), len(client_loader.dataset), avg_local_loss


def run_stage2_local_epoch(args, client_id, global_mlp_state, base_model, qwen_model, processor, device, model_dtype, train_emb_normed, 
train_qids, qid_to_row, index, client_loader, client_task_map, writer, round_idx, reprag_token_id):
    step_fns = get_dataset_step_functions(args.dataset)
    stage2_finetune_step = step_fns["stage2_finetune_step"]

    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    local_mlp = PrefixProjector(
        input_size=H,
        output_size=H,
        projector_type="mlp2x_gelu",
        p_drop=0.1
    ).to(device)

    load_model_state_dict(local_mlp, global_mlp_state, device)
    local_mlp.train()
    local_mlp.requires_grad_(True)

    optimizer = torch.optim.AdamW(local_mlp.parameters(), lr=args.stage2_lr)

    loss_sum = 0.0
    acc_sum = 0.0
    n_sum = 0
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

                loss_val, acc_val, global_step_dummy = stage2_finetune_step(
                    args=args,
                    epoch=local_epoch,
                    step=step_idx,
                    global_step=global_step_dummy,
                    batch_tasks=batch_tasks,
                    device=device,
                    processor=processor,
                    qwen_model=qwen_model,
                    base_model=base_model,
                    mlp=local_mlp,
                    optimizer=optimizer,
                    writer=writer,
                    model_dtype=model_dtype,
                    train_emb_normed=train_emb_normed,
                    train_qids=train_qids,
                    qid_to_row=qid_to_row,
                    index=index,
                    train_task_map=client_task_map,
                    reprag_token_id=reprag_token_id,
                    pbar=pbar,
                )

                if loss_val is None or acc_val is None:
                    continue

                loss_sum += float(loss_val) * bs
                acc_sum += float(acc_val) * bs
                n_sum += bs

    avg_loss = loss_sum / max(1, n_sum)
    avg_acc = acc_sum / max(1, n_sum)

    print(
        f"[FedRepRag][Stage2][Round {round_idx+1}] Client {client_id+1} "
        f"local_loss={avg_loss:.4f} local_acc={avg_acc:.4f}"
    )

    return clone_model_state_dict(local_mlp), len(client_loader.dataset), avg_loss, avg_acc


def evaluate_stage1_global(args, global_mlp, val_loader, base_model, qwen_model, processor, device, model_dtype, train_emb_normed, 
train_qids, qid_to_row, index, train_task_map, writer, global_step, reprag_token_id, eval_client_id=0):
    step_fns = get_dataset_step_functions(args.dataset)
    stage1_eval_step = step_fns["stage1_eval_step"]

    global_mlp.eval()
    val_loss_list = []

    with tqdm(total=len(val_loader), desc=f"[Stage1][Eval][Client {eval_client_id}]", leave=False) as pbar:
        for i, val_batch in enumerate(val_loader):
            vloss = stage1_eval_step(
                args=args,
                base_model=base_model,
                qwen_model=qwen_model,
                processor=processor,
                mlp=global_mlp,
                device=device,
                model_dtype=model_dtype,
                batch_tasks=val_batch,
                train_emb_normed=train_emb_normed,
                train_qids=train_qids,
                qid_to_row=qid_to_row,
                index=index,
                train_task_map=train_task_map,
                writer=writer,
                global_step=global_step,
                reprag_token_id=reprag_token_id,
                eval_prefix="stage1/eval",
                pbar=pbar,
            )
            if vloss is not None:
                val_loss_list.append(vloss)

    return sum(val_loss_list) / max(1, len(val_loss_list))


def evaluate_stage2_global(args, global_mlp, val_loader, base_model, qwen_model, processor, device, model_dtype, train_emb_normed, 
train_qids, qid_to_row, index, train_task_map,writer, global_step, reprag_token_id, eval_client_id=0):
    step_fns = get_dataset_step_functions(args.dataset)
    stage2_eval_step = step_fns["stage2_eval_step"]

    global_mlp.eval()

    val_loss_sum = 0.0
    val_loss_n = 0
    val_acc_sum = 0.0
    val_acc_n = 0

    with tqdm(total=len(val_loader), desc=f"[Stage2][Eval][Client {eval_client_id}]", leave=False) as pbar:
        for i, val_batch in enumerate(val_loader):
            v_bs = len(val_batch) if val_batch is not None else 0
            if v_bs == 0:
                pbar.update(1)
                continue

                # Note: keep this aligned with your dataset-specific function signature
            vloss, vacc = stage2_eval_step(
                args=args,
                epoch=0,
                step=i,
                global_step=global_step,
                batch_tasks=val_batch,
                device=device,
                processor=processor,
                qwen_model=qwen_model,
                base_model=base_model,
                mlp=global_mlp,
                writer=writer,
                model_dtype=model_dtype,
                train_emb_normed=train_emb_normed,
                train_qids=train_qids,
                qid_to_row=qid_to_row,
                index=index,
                train_task_map=train_task_map,
                reprag_token_id=reprag_token_id,
                eval_prefix="stage2/eval",
                pbar=pbar,
            )

            if vacc is not None:
                val_acc_sum += float(vacc) * v_bs
                val_acc_n += v_bs
            if vloss is not None:
                val_loss_sum += float(vloss) * v_bs
                val_loss_n += v_bs

    eval_avg_acc = val_acc_sum / max(1, val_acc_n)
    eval_avg_loss = val_loss_sum / max(1, val_loss_n) if val_loss_n > 0 else None
    return eval_avg_loss, eval_avg_acc