"""
Federated MLP projector training for PubMedQA / PubMedQA (Qwen3, text-only).

Pipeline:
  1. Load per-client local corpus FAISS pools.
  2. Load per-client local corpus embeddings (MLP inputs).
  3. Load per-client Stage-1 corpus tasks.
  4. Load per-client Stage-2 labeled PubMedQA train tasks.
  5. Load shared validation tasks.
  6. Load frozen Qwen3, add <RepRAG>.
  7. Create PrefixProjector(D_content -> H).
  8. Dispatch federated Stage-1 / Stage-2 training.
"""

import os
import re
import sys
import pathlib
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AddedToken
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from VQA.arg_PubMedQA.PubMedQA_FL_mlp_train_arg import parse_args
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index
from VQA.utilise.retrieval_architecture import PrefixProjector

from VQA.utilise.PubMedQA.PubMedQA_data_process import load_pubmedqa_train, load_combined_pool, PubMedQADataset, collate_tasks
from VQA.utilise.PubMedQA.PubMedQA_prompt_message import build_pubmedqa_embedding_message
from VQA.FL_Method.fl_method_config import FL_METHOD_REGISTRY


REPRAG_TOKEN = "<RepRAG>"


# -------------------------------------------------------
# Model loading
# -------------------------------------------------------

def load_model(args):
    model_name = args.model_name
    print(f"[Model] Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if getattr(args, "use_quantization", False):
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
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

_CLIENT_RE = re.compile(r"client_(\d+)\.(pt|json|jsonl)$")


def _extract_client_id(path: pathlib.Path) -> int:
    m = _CLIENT_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected client filename format: {path}")
    return int(m.group(1))


def _index_client_files(
    client_dir: str,
    suffixes: Tuple[str, ...] = (".pt", ".json", ".jsonl"),
) -> Dict[int, pathlib.Path]:
    p = pathlib.Path(client_dir)
    if not p.exists():
        raise ValueError(f"Directory not found: {client_dir}")

    files: List[pathlib.Path] = []
    for suf in suffixes:
        files.extend(sorted(p.glob(f"client_*{suf}")))

    out: Dict[int, pathlib.Path] = {}
    for fp in files:
        cid = _extract_client_id(fp)
        out[cid] = fp
    return out


def _assert_same_client_ids(name_to_files: Dict[str, Dict[int, pathlib.Path]]) -> List[int]:
    if not name_to_files:
        raise ValueError("No client file dictionaries were provided.")

    key_sets = {name: set(d.keys()) for name, d in name_to_files.items()}
    ref_name, ref_set = next(iter(key_sets.items()))

    for name, s in key_sets.items():
        if s != ref_set:
            raise ValueError(
                f"Client ID mismatch between {ref_name}={sorted(ref_set)} and {name}={sorted(s)}"
            )

    client_ids = sorted(ref_set)
    if not client_ids:
        raise ValueError("No matched client ids found.")
    return client_ids


# -------------------------------------------------------
# On-the-fly query encoding for validation
# -------------------------------------------------------

def encode_tasks_online(
    tasks: List[Dict[str, Any]],
    tokenizer,
    base_model,
    device,
    batch_size: int = 8,
    max_length: int = 512,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode query tasks on-the-fly with the frozen LM.
    """
    was_training = base_model.training
    base_model.eval()

    all_embeddings = []
    qids_order = []

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        msgs = [build_pubmedqa_embedding_message(t, include_gold=False) for t in chunk]

        texts = [
            tokenizer.apply_chat_template(
                [m],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
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
            hidden = out.hidden_states[-1].float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            emb = summed / mask.sum(dim=1).clamp(min=1e-6)
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
# Client retrieval pools
# -------------------------------------------------------

def build_client_pubmedqa_retrieval_pools(args):
    """
    Build one retrieval pool per client.

    Each pool contains:
      - local corpus question-embedding FAISS index
      - corpus qids / qid->row
      - corpus embedding map (title+content embeddings, MLP input)
      - corpus task map
      - labeled-train question embeddings for Stage 2
    """
    question_emb_files = _index_client_files(args.client_question_embeddings_dir, suffixes=(".pt",))
    corpus_emb_files = _index_client_files(args.client_content_embeddings_dir, suffixes=(".pt",))
    train_question_emb_files = _index_client_files(args.client_train_embeddings_dir, suffixes=(".pt",))
    corpus_jsonl_files = _index_client_files(args.client_corpus_dir, suffixes=(".jsonl",))

    client_ids = _assert_same_client_ids({
        "question_emb": question_emb_files,
        "corpus_emb": corpus_emb_files,
        "train_question_emb": train_question_emb_files,
        "corpus_jsonl": corpus_jsonl_files,
    })

    pools = []
    d_content_ref = None

    for cid in client_ids:
        # 1) local corpus question embeddings -> FAISS
        question_ckpt = torch.load(question_emb_files[cid], map_location="cpu")
        question_emb = question_ckpt["embeddings"].cpu().numpy().astype("float32")
        question_qids = [str(q) for q in question_ckpt["qids"]]
        question_emb_normed = l2_normalize(question_emb)
        corpus_faiss_index = build_faiss_ip_index(question_emb_normed)
        corpus_qid_to_row = {qid: i for i, qid in enumerate(question_qids)}

        # 2) local labeled-train question embeddings (for stage 2)
        train_ckpt = torch.load(train_question_emb_files[cid], map_location="cpu")
        train_question_emb = train_ckpt["embeddings"].cpu().numpy().astype("float32")
        train_question_qids = [str(q) for q in train_ckpt["qids"]]
        train_query_emb_normed = l2_normalize(train_question_emb)
        train_query_qid_to_row = {qid: i for i, qid in enumerate(train_question_qids)}

        # 3) local corpus/title+content embeddings
        corpus_ckpt = torch.load(corpus_emb_files[cid], map_location="cpu")
        corpus_emb = corpus_ckpt["embeddings"].cpu().numpy().astype("float32")
        corpus_emb_qids = [str(q) for q in corpus_ckpt["qids"]]

        if len(corpus_emb_qids) != len(corpus_emb):
            raise ValueError(f"[Client {cid}] corpus embedding qid / embedding length mismatch")

        if set(question_qids) != set(corpus_emb_qids):
            raise ValueError(
                f"[Client {cid}] question-embedding qids and corpus-embedding qids do not match"
            )

        content_emb_map = {qid: corpus_emb[i] for i, qid in enumerate(corpus_emb_qids)}
        d_content = int(corpus_emb.shape[1])

        if d_content_ref is None:
            d_content_ref = d_content
        elif d_content != d_content_ref:
            raise ValueError(
                f"Corpus embedding dim mismatch across clients: "
                f"{d_content_ref} vs {d_content} (client {cid})"
            )

        # 4) local corpus text/tasks
        corpus_tasks = load_combined_pool(str(corpus_jsonl_files[cid]), idx=0, num_parts=1)
        corpus_task_map = {str(t["qid"]): t for t in corpus_tasks}

        task_qids = set(corpus_task_map.keys())
        if set(question_qids) != task_qids:
            raise ValueError(
                f"[Client {cid}] corpus task qids and question-embedding qids do not match"
            )

        pool = {
            "client_id": cid,
            "corpus_emb_normed": question_emb_normed,
            "corpus_faiss_index": corpus_faiss_index,
            "corpus_qids": question_qids,
            "corpus_qid_to_row": corpus_qid_to_row,
            "content_emb_map": content_emb_map,
            "content_dim": d_content,
            "corpus_task_map": corpus_task_map,
            "train_query_emb_normed": train_query_emb_normed,
            "train_query_qid_to_row": train_query_qid_to_row,
        }
        pools.append(pool)

        print(
            f"[Client {cid}] "
            f"corpus_q={len(question_qids)} | "
            f"train_q={len(train_question_qids)} | "
            f"corpus_emb={len(content_emb_map)} | "
            f"D_content={d_content}"
        )

    return pools


# -------------------------------------------------------
# Datasets
# -------------------------------------------------------

def build_stage1_client_dataloaders(args):
    """
    Stage 1 uses each client's local combined-pool corpus items.
    """
    client_files = _index_client_files(args.client_corpus_dir, suffixes=(".jsonl",))
    if len(client_files) == 0:
        raise ValueError(f"No client corpus files found in {args.client_corpus_dir}")

    client_ids = sorted(client_files.keys())
    client_loaders, client_task_maps, client_sizes = [], [], []

    for cid in client_ids:
        tasks = load_combined_pool(str(client_files[cid]), idx=0, num_parts=1)
        dataset = PubMedQADataset(tasks)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_tasks,
        )
        task_map = {str(t["qid"]): t for t in tasks}

        client_loaders.append(loader)
        client_task_maps.append(task_map)
        client_sizes.append(len(dataset))

        print(f"[Stage1][Client {cid}] {len(dataset)} samples")

    return client_loaders, client_task_maps, client_sizes


def build_stage2_client_dataloaders(args):
    """
    Stage 2 uses each client's labeled PubMedQA train set.
    """
    client_files = _index_client_files(args.client_train_dir, suffixes=(".json",))
    if len(client_files) == 0:
        raise ValueError(f"No client labeled-train files found in {args.client_train_dir}")

    client_ids = sorted(client_files.keys())
    client_loaders, client_task_maps, client_sizes = [], [], []

    for cid in client_ids:
        tasks = load_pubmedqa_train(str(client_files[cid]))
        dataset = PubMedQADataset(tasks)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_tasks,
        )
        task_map = {str(t["qid"]): t for t in tasks}

        client_loaders.append(loader)
        client_task_maps.append(task_map)
        client_sizes.append(len(dataset))

        print(f"[Stage2][Client {cid}] {len(dataset)} samples")

    return client_loaders, client_task_maps, client_sizes


def build_eval_loader(args):
    if not getattr(args, "val_json", None):
        raise ValueError("--val_json is required for FL validation.")

    val_tasks = load_pubmedqa_train(args.val_json)
    for t in val_tasks:
        t["split"] = "val"

    val_dataset = PubMedQADataset(val_tasks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tasks,
    )

    print(f"[Eval] val set: {len(val_dataset)}")
    return val_loader, val_tasks, {str(t["qid"]): t for t in val_tasks}


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    args = parse_args()
    args.dataset = "PubMedQA"

    print("[Paths]")
    print(f"  client_corpus_dir             = {args.client_corpus_dir}")
    print(f"  client_train_dir              = {args.client_train_dir}")
    print(f"  client_question_embeddings_dir = {args.client_question_embeddings_dir}")
    print(f"  client_content_embeddings_dir = {args.client_content_embeddings_dir}")
    print(f"  client_train_embeddings_dir   = {args.client_train_embeddings_dir}")
    print(f"  val_json                      = {args.val_json}")
    print(f"  eval_corpus_embeddings_path   = {getattr(args, 'eval_corpus_embeddings_path', None)}")
    print(f"  eval_content_embeddings_path  = {getattr(args, 'eval_content_embeddings_path', None)}")
    print(f"  eval_corpus_jsonl             = {getattr(args, 'eval_corpus_jsonl', None)}")

    log_dir = getattr(args, "ckpt_dir", None) or "./logs"
    os.makedirs(log_dir, exist_ok=True)
    dump_args_to_logdir(args, log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------
    # 1) Build per-client retrieval pools
    # ----------------------------------------------
    client_retrieval_pools = build_client_pubmedqa_retrieval_pools(args)

    # ----------------------------------------------
    # 2) Build per-client Stage 1 / Stage 2 loaders
    # ----------------------------------------------
    stage1_client_loaders, stage1_client_task_maps, stage1_client_sizes = build_stage1_client_dataloaders(args)
    stage2_client_loaders, stage2_client_task_maps, stage2_client_sizes = build_stage2_client_dataloaders(args)

    val_loader, val_tasks, val_task_map = build_eval_loader(args)

    # ----------------------------------------------
    # 2b) Build global eval retrieval pool.
    #     Uses the full corpus (with test items) so that test queries can
    #     retrieve their own relevant passages at inference time, while the
    #     per-client training corpora remain test-item-free.
    #     Falls back to client 0's pool if no eval corpus args are provided.
    # ----------------------------------------------
    if getattr(args, "eval_corpus_embeddings_path", None):
        print(f"[EvalPool] Loading eval corpus embeddings from {args.eval_corpus_embeddings_path}")
        eval_corp_ckpt = torch.load(args.eval_corpus_embeddings_path, map_location="cpu")
        eval_corpus_emb = eval_corp_ckpt["embeddings"].cpu().numpy().astype("float32")
        eval_corpus_qids = [str(q) for q in eval_corp_ckpt["qids"]]
        eval_corpus_emb_normed = l2_normalize(eval_corpus_emb)
        eval_corpus_faiss_index = build_faiss_ip_index(eval_corpus_emb_normed)

        print(f"[EvalPool] Loading eval content embeddings from {args.eval_content_embeddings_path}")
        eval_cont_ckpt = torch.load(args.eval_content_embeddings_path, map_location="cpu")
        eval_cont_emb = eval_cont_ckpt["embeddings"].cpu().numpy().astype("float32")
        eval_cont_qids = [str(q) for q in eval_cont_ckpt["qids"]]
        eval_content_emb_map = {qid: eval_cont_emb[i] for i, qid in enumerate(eval_cont_qids)}
        d_eval_content = int(eval_cont_emb.shape[1])

        print(f"[EvalPool] Loading eval corpus text from {args.eval_corpus_jsonl}")
        eval_corpus_tasks = load_combined_pool(args.eval_corpus_jsonl, idx=0, num_parts=1)
        eval_corpus_task_map = {str(t["qid"]): t for t in eval_corpus_tasks}

        eval_retrieval_pool = {
            "corpus_emb_normed":   eval_corpus_emb_normed,
            "corpus_faiss_index":  eval_corpus_faiss_index,
            "corpus_qids":         eval_corpus_qids,
            "corpus_qid_to_row":   {qid: i for i, qid in enumerate(eval_corpus_qids)},
            "content_emb_map":     eval_content_emb_map,
            "content_dim":         d_eval_content,
            "corpus_task_map":     eval_corpus_task_map,
        }
        print(f"[EvalPool] Global eval pool built: {len(eval_corpus_qids)} items.")
    else:
        print("[EvalPool] No separate eval corpus provided; evaluation will use client 0's training corpus.")
        eval_retrieval_pool = None

    if len(client_retrieval_pools) != len(stage1_client_loaders):
        raise ValueError(
            f"Mismatch: {len(client_retrieval_pools)} retrieval pools vs "
            f"{len(stage1_client_loaders)} stage1 client loaders"
        )

    if len(client_retrieval_pools) != len(stage2_client_loaders):
        raise ValueError(
            f"Mismatch: {len(client_retrieval_pools)} retrieval pools vs "
            f"{len(stage2_client_loaders)} stage2 client loaders"
        )

    # Explicit qid consistency checks
    for pool, task_map in zip(client_retrieval_pools, stage1_client_task_maps):
        corpus_qids_from_tasks = set(task_map.keys())
        corpus_qids_from_emb = set(pool["corpus_qid_to_row"].keys())
        if corpus_qids_from_tasks != corpus_qids_from_emb:
            raise ValueError(
                f"[Client {pool['client_id']}] Stage1 corpus task qids and corpus question embedding qids do not match"
            )

    for pool, task_map in zip(client_retrieval_pools, stage2_client_task_maps):
        train_qids_from_tasks = set(task_map.keys())
        train_qids_from_emb = set(pool["train_query_qid_to_row"].keys())
        if train_qids_from_tasks != train_qids_from_emb:
            raise ValueError(
                f"[Client {pool['client_id']}] Stage2 train task qids and train question embedding qids do not match"
            )

    # ----------------------------------------------
    # 3) Load frozen Qwen3
    # ----------------------------------------------
    tokenizer, model = load_model(args)
    model.to(device)
    base_model = model.module if hasattr(model, "module") else model

    num_added = tokenizer.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
    if num_added > 0:
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"[Model] Added {REPRAG_TOKEN}; vocab resized to {len(tokenizer)}")

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
    H = int(base_model.config.hidden_size)
    print(f"[Model] Frozen Qwen3 loaded; hidden_size={H}, dtype={model_dtype}")

    # ----------------------------------------------
    # 4) Global MLP projector
    # ----------------------------------------------
    d_content = int(client_retrieval_pools[0]["content_dim"])
    mlp = PrefixProjector(
        input_size=d_content,
        output_size=H,
        projector_type="mlp2x_gelu",
        p_drop=0.1,
    ).to(device)
    mlp.requires_grad_(True)

    with torch.no_grad():
        dummy = torch.zeros(2, d_content, device=device)
        out = mlp(dummy)
        assert out.shape[-1] == H, f"MLP output dim must be H={H}, got {out.shape}"

    print(f"[MLP] PrefixProjector: D_content={d_content} -> H={H}")

    # ----------------------------------------------
    # 5) TensorBoard
    # ----------------------------------------------
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_root = getattr(args, "tb_dir", "./tb_logs")
    tb_log_dir = os.path.join(tb_root, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TB] Logging to {tb_log_dir}")

    # ----------------------------------------------
    # 6) Shared val query encoder hook
    # ----------------------------------------------
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

    # ----------------------------------------------
    # 7) FL training dispatcher
    # ----------------------------------------------
    fl_method = getattr(args, "fl_method", "fedRepRag").lower()
    if fl_method not in FL_METHOD_REGISTRY:
        raise ValueError(f"Unsupported fl_method: {fl_method}")

    if args.stage1_pretrain:
        print(f"[FL] Running PubMedQA Stage1 with method={fl_method}")
        FL_METHOD_REGISTRY[fl_method]["stage1"](
            args=args,
            global_mlp=mlp,
            client_loaders=stage1_client_loaders,
            client_task_maps=stage1_client_task_maps,
            val_loader=val_loader,
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            client_retrieval_pools=client_retrieval_pools,
            writer=writer,
            reprag_token_id=reprag_token_id,
            get_val_query_emb_fn=get_val_query_emb,
            eval_retrieval_pool=eval_retrieval_pool,
        )

    if args.stage2_finetuning:
        print(f"[FL] Running PubMedQA Stage2 with method={fl_method}")

        if getattr(args, "stage2_init_ckpt", None):
            ckpt2 = torch.load(args.stage2_init_ckpt, map_location="cpu")
            if "mlp_state" in ckpt2:
                mlp.load_state_dict(ckpt2["mlp_state"], strict=True)
                print(f"[Stage2] Loaded global MLP from {args.stage2_init_ckpt}")

        FL_METHOD_REGISTRY[fl_method]["stage2"](
            args=args,
            global_mlp=mlp,
            client_loaders=stage2_client_loaders,
            client_task_maps=stage2_client_task_maps,
            val_loader=val_loader,
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            client_retrieval_pools=client_retrieval_pools,
            writer=writer,
            reprag_token_id=reprag_token_id,
            get_val_query_emb_fn=get_val_query_emb,
            eval_retrieval_pool=eval_retrieval_pool,
        )

    writer.close()
    print("[Done]")


if __name__ == "__main__":
    main()