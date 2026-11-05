import os
import sys
import pathlib
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import random
from transformers import AddedToken
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import copy
from collections import OrderedDict

ROOT = pathlib.Path(__file__).resolve().parents[1]  # go up to repo root
sys.path.append(str(ROOT))

from VQA.arg_ScienceQA.ScienceQA_FL_mlp_train_arg import parse_args

from VQA.utilise.model_loading import load_model_standard
from VQA.utilise.retrieval_architecture import PrefixProjector
from VQA.utilise.retrieval_helper import build_client_retrieval_pools
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir

from VQA.utilise.data_process import load_json, get_client_json_files, collate_tasks, TrainInjectionDataset
from VQA.utilise.ScienceQA.ScienceQA_data_process import build_tasks_from_scienceqa

from VQA.FL_Method.fl_method_config import FL_METHOD_REGISTRY


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
REPRAG_TOKEN = "<RepRAG>"


# ---------------------------
# Dataset
# ---------------------------
def build_client_dataloaders(args):
    """
    Returns:
        client_loaders: list[DataLoader]
        client_task_maps: list[dict]
        client_sizes: list[int]
    """
    client_files = get_client_json_files(args.client_train_dir)

    client_loaders = []
    client_task_maps = []
    client_sizes = []

    for client_path in client_files:
        client_probs = load_json(str(client_path))
        client_tasks = build_tasks_from_scienceqa(
            client_probs,
            images_root=args.images_root if args.images_root else None,
            split_filter=None
        )

        client_dataset = TrainInjectionDataset(client_tasks)
        client_loader = DataLoader(
            client_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_tasks
        )

        client_task_map = {str(t["qid"]): t for t in client_tasks}

        client_loaders.append(client_loader)
        client_task_maps.append(client_task_map)
        client_sizes.append(len(client_dataset))

        print(f"[Client] {client_path.name}: {len(client_dataset)} samples")

    return client_loaders, client_task_maps, client_sizes


def build_eval_loader(args):
    val_probs = load_json(args.val_json)
    val_tasks = build_tasks_from_scienceqa(
        val_probs,
        images_root=args.images_root if args.images_root else None,
        split_filter=None
    )
    val_dataset = TrainInjectionDataset(val_tasks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tasks
    )
    print(f"[Eval] val set: {len(val_dataset)}")
    return val_loader, {str(t["qid"]): t for t in val_tasks}


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    args.dataset = "ScienceQA"
    
    # make sure you have log_dir available here
    log_dir = getattr(args, "ckpt_dir", None)
    if log_dir is None:
        # fallback if your code uses another name
        log_dir = getattr(args, "output_dir", "./logs")

    dump_args_to_logdir(args, log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    # 1) Load per-client local embeddings & build FAISS
    # ----------------------------------------------
    client_retrieval_pools = build_client_retrieval_pools(args)

    # ----------------------------------------------
    # 2) Load client train loaders + shared val loader
    # ----------------------------------------------
    client_loaders, client_task_maps, client_sizes = build_client_dataloaders(args)
    val_loader, val_task_map = build_eval_loader(args)

    if len(client_retrieval_pools) != len(client_loaders):
        raise ValueError(
            f"Mismatch: {len(client_retrieval_pools)} client embedding pools vs "
            f"{len(client_loaders)} client dataloaders"
        )

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
    # 4) Global MLP projector
    # ----------------------------------------------
    H = base_model.config.hidden_size
    mlp = PrefixProjector(
        input_size=H,
        output_size=H,
        projector_type="mlp2x_gelu",
        p_drop=0.1
    ).to(device)
    mlp.requires_grad_(True)

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
    # 6) FL training dispatcher
    # ----------------------------------------------
    fl_method = getattr(args, "fl_method", "fedRepRag").lower()
    if fl_method not in FL_METHOD_REGISTRY:
        raise ValueError(f"Unsupported fl_method: {fl_method}")

    if args.stage1_pretrain:
        print(f"[FL] Running Stage1 with method={fl_method}")
        FL_METHOD_REGISTRY[fl_method]["stage1"](
            args=args,
            global_mlp=mlp,
            client_loaders=client_loaders,
            client_task_maps=client_task_maps,
            val_loader=val_loader,
            base_model=base_model,
            qwen_model=qwen_model,
            processor=processor,
            device=device,
            model_dtype=model_dtype,
            client_retrieval_pools=client_retrieval_pools,
            writer=writer,
            reprag_token_id=reprag_token_id,
        )

    if args.stage2_finetuning:
        print(f"[FL] Running Stage2 with method={fl_method}")

        if getattr(args, "stage2_init_ckpt", None):
            ckpt2 = torch.load(args.stage2_init_ckpt, map_location="cpu")
            if "mlp_state" in ckpt2:
                mlp.load_state_dict(ckpt2["mlp_state"], strict=True)
                print(f"[Stage2] Loaded global MLP from {args.stage2_init_ckpt}")

        FL_METHOD_REGISTRY[fl_method]["stage2"](
            args=args,
            global_mlp=mlp,
            client_loaders=client_loaders,
            client_task_maps=client_task_maps,
            val_loader=val_loader,
            base_model=base_model,
            qwen_model=qwen_model,
            processor=processor,
            device=device,
            model_dtype=model_dtype,
            client_retrieval_pools=client_retrieval_pools,
            writer=writer,
            reprag_token_id=reprag_token_id,
        )

    

if __name__ == "__main__":
    main()
