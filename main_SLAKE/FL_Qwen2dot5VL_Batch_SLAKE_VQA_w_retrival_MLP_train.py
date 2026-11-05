import os
import sys
import pathlib
from datetime import datetime
from transformers import AddedToken
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from VQA.arg_SLAKE.Slake_FL_mlp_train_arg import parse_args

from VQA.utilise.model_loading import load_model_standard, load_gemma3_standard
from VQA.utilise.retrieval_architecture import PrefixProjector
from VQA.utilise.retrieval_helper import build_client_retrieval_pools
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir

from VQA.utilise.SLAKE.Slake_data_process import load_slake_items, build_tasks_from_slake, TrainInjectionDataset, collate_tasks

from VQA.FL_Method.fl_method_config import FL_METHOD_REGISTRY


REPRAG_TOKEN = "<RepRAG>"


# ---------------------------
# Dataset
# ---------------------------
def build_client_dataloaders(args):
    client_dir = pathlib.Path(args.client_train_dir)
    client_files = sorted(client_dir.glob("client_*.json"))

    if len(client_files) == 0:
        raise ValueError(f"No client_*.json files found in {args.client_train_dir}")

    lang = getattr(args, "lang", "en")
    images_root = getattr(args, "images_root", None)

    client_loaders, client_task_maps, client_sizes = [], [], []

    for client_path in client_files:
        client_items = load_slake_items(str(client_path))
        client_tasks = build_tasks_from_slake(client_items, images_root=images_root, split_name="train", lang_filter=lang)

        client_dataset = TrainInjectionDataset(client_tasks)
        if len(client_dataset) == 0:
            print(f"[Warning] {client_path.name}: 0 samples kept after filtering")
            continue

        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tasks)
        client_task_map = {str(t["qid"]): t for t in client_dataset.tasks}

        client_loaders.append(client_loader)
        client_task_maps.append(client_task_map)
        client_sizes.append(len(client_dataset))

        print(f"[Client] {client_path.name}: {len(client_dataset)} samples")

    return client_loaders, client_task_maps, client_sizes


def build_eval_loader(args):
    if not getattr(args, "val_json", None):
        raise ValueError("--val_json is required for FL validation.")

    lang = getattr(args, "lang", "en")
    images_root = getattr(args, "images_root", None)

    val_items = load_slake_items(args.val_json)
    val_tasks = build_tasks_from_slake(val_items, images_root=images_root, split_name="val", lang_filter=lang)

    val_dataset = TrainInjectionDataset(val_tasks)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tasks)

    print(f"[Eval] val set: {len(val_dataset)}")
    return val_loader, {str(t["qid"]): t for t in val_tasks}


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    args.dataset = "SLAKE"

    log_dir = getattr(args, "ckpt_dir", None)
    if log_dir is None:
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
    if "gemma" in args.model_name.lower():
        processor, qwen_model = load_gemma3_standard(args.model_name)
    else:
        processor, qwen_model = load_model_standard(args.model_name)
        qwen_model.to(device)
    processor.tokenizer.padding_side = "left"
    base_model = qwen_model.module if hasattr(qwen_model, "module") else qwen_model

    num_added = processor.tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
    if num_added > 0:
        base_model.resize_token_embeddings(len(processor.tokenizer))

    out_emb = base_model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight"):
        base_model.config.vocab_size = int(out_emb.weight.shape[0])
    else:
        base_model.config.vocab_size = int(base_model.get_input_embeddings().weight.shape[0])

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
    cfg = base_model.config
    H = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
    assert H is not None, f"Cannot determine hidden_size from config: {cfg}"
    mlp = PrefixProjector(input_size=H, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1).to(device)
    mlp.requires_grad_(True)

    with torch.no_grad():
        dummy = torch.zeros(2, H, device=device)
        out = mlp(dummy)
        assert out.shape[-1] == H, f"MLP output dim must be H={H}, got {out.shape}"

    # ----------------------------------------------
    # 5) TensorBoard
    # ----------------------------------------------
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
        # Resume from a halfway stage-1 checkpoint if provided
        args.stage1_start_round = 0
        if getattr(args, "stage1_resume_ckpt", None):
            resume_ckpt = torch.load(args.stage1_resume_ckpt, map_location="cpu")
            if "mlp_state" in resume_ckpt:
                mlp.load_state_dict(resume_ckpt["mlp_state"], strict=True)
                print(f"[Stage1 Resume] Loaded MLP weights from {args.stage1_resume_ckpt}")
            args.stage1_start_round = int(resume_ckpt.get("round", 0))
            print(f"[Stage1 Resume] Skipping rounds 1-{args.stage1_start_round}, "
                  f"continuing from round {args.stage1_start_round + 1}")

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