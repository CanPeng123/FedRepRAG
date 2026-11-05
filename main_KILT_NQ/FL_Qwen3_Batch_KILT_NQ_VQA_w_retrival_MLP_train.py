"""
Federated MLP projector training for KILT-NQ (Qwen3, text-only).

Pipeline:
  1. Load per-client local corpus JSONL (from FL split).
  2. Subset global embeddings (question + content) per client.
  3. Build per-client FAISS index from subsetted question embeddings.
  4. Load shared validation JSONL (val queries encoded on-the-fly).
  5. Load frozen Qwen3, add <RepRAG> token.
  6. Create global PrefixProjector(D_content -> H).
  7. Run federated Stage-1 / Stage-2 training.

Unlike PubMedQA, KILT-NQ uses the same per-client JSONL for both Stage-1
(paraphrase pretraining) and Stage-2 (answer CE fine-tuning).
"""

import os
import re
import sys
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AddedToken
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from VQA.arg_KILT_NQ.KILT_NQ_FL_mlp_train_arg import parse_args
from VQA.utilise.mlp_utilise.mlp_log_record import dump_args_to_logdir
from VQA.utilise.retrieval_helper import l2_normalize, build_faiss_ip_index
from VQA.utilise.retrieval_architecture import PrefixProjector

from VQA.utilise.KILT_NQ.KILT_NQ_data_process import load_kilt_nq, KILTNQDataset, collate_records
from VQA.utilise.KILT_NQ.KILT_NQ_prompt_message import build_kilt_nq_embedding_message

from VQA.FL_Method.kilt_nq_fedRepRag import (
    train_kilt_nq_fedRepRag_stage1,
    train_kilt_nq_fedRepRag_stage2,
)

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
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    model.eval()
    return tokenizer, model


# -------------------------------------------------------
# Client file discovery
# -------------------------------------------------------

_CLIENT_RE = re.compile(r"client_(\d+)\.jsonl$")


def _index_client_jsonl_files(client_dir: str) -> Dict[int, pathlib.Path]:
    p = pathlib.Path(client_dir)
    if not p.exists():
        raise ValueError(f"Client corpus directory not found: {client_dir}")
    out: Dict[int, pathlib.Path] = {}
    for fp in sorted(p.glob("client_*.jsonl")):
        m = _CLIENT_RE.match(fp.name)
        if m:
            out[int(m.group(1))] = fp
    if not out:
        raise ValueError(f"No client_*.jsonl files found in {client_dir}")
    return out


# -------------------------------------------------------
# On-the-fly query encoding for validation
# -------------------------------------------------------

def encode_records_online(
    records: List[Dict[str, Any]],
    tokenizer,
    base_model,
    device,
    batch_size: int = 8,
    max_length: int = 512,
) -> Tuple[np.ndarray, Dict[str, int]]:
    was_training = base_model.training
    base_model.eval()

    all_embeddings = []
    qids_order = []

    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        msgs = [build_kilt_nq_embedding_message(r, include_gold=False) for r in chunk]

        texts = [
            tokenizer.apply_chat_template(
                [m], tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            for m in msgs
        ]

        enc = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
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
        for r in chunk:
            qids_order.append(str(r["id"]))

    base_model.train(was_training)

    emb_normed = np.concatenate(all_embeddings, axis=0).astype("float32")
    qid_to_row = {qid: i for i, qid in enumerate(qids_order)}
    return emb_normed, qid_to_row


# -------------------------------------------------------
# Build per-client retrieval pools
# -------------------------------------------------------

def build_client_pools(args, client_jsonl_files, global_q_emb, global_q_qids,
                        global_c_emb, global_c_qids):
    """
    For each client, subset global embeddings to client QIDs and build FAISS.
    Returns list of client pool dicts.
    """
    # Build global qid -> row maps
    global_q_qid_to_row = {qid: i for i, qid in enumerate(global_q_qids)}
    global_c_qid_to_row = {qid: i for i, qid in enumerate(global_c_qids)}

    d_content_ref = None
    pools = []

    for cid in sorted(client_jsonl_files.keys()):
        records = load_kilt_nq(str(client_jsonl_files[cid]))
        client_qids = [str(r["id"]) for r in records]

        # Filter only QIDs present in global embeddings
        valid_q = [(qid, global_q_qid_to_row[qid]) for qid in client_qids if qid in global_q_qid_to_row]
        valid_c = [(qid, global_c_qid_to_row[qid]) for qid in client_qids if qid in global_c_qid_to_row]

        if not valid_q:
            raise ValueError(f"[Client {cid}] No question embeddings found for any client QID.")

        # Question embeddings -> FAISS
        q_qids_local = [q for q, _ in valid_q]
        q_emb_local = np.stack([global_q_emb[row] for _, row in valid_q]).astype("float32")
        q_emb_normed = l2_normalize(q_emb_local)
        faiss_index = build_faiss_ip_index(q_emb_normed)
        q_qid_to_row = {qid: i for i, qid in enumerate(q_qids_local)}

        # Content embeddings -> dict
        c_qids_present = set(qid for qid, _ in valid_c)
        content_emb_map = {qid: global_c_emb[row] for qid, row in valid_c}

        # Check consistency
        if set(q_qids_local) != c_qids_present:
            missing = set(q_qids_local) - c_qids_present
            print(f"[Client {cid}] Warning: {len(missing)} QIDs have question but no content embedding (using zeros).")
            D_c = global_c_emb.shape[1]
            for qid in missing:
                content_emb_map[qid] = np.zeros(D_c, dtype=np.float32)

        d_content = int(global_c_emb.shape[1])
        if d_content_ref is None:
            d_content_ref = d_content
        elif d_content != d_content_ref:
            raise ValueError(f"Content embedding dim mismatch: {d_content_ref} vs {d_content} (client {cid})")

        train_record_map = {str(r["id"]): r for r in records}

        pool = {
            "client_id": cid,
            "query_emb_normed": q_emb_normed,
            "train_faiss_index": faiss_index,
            "train_ids": q_qids_local,
            "query_id_to_row": q_qid_to_row,
            "content_emb_map": content_emb_map,
            "train_record_map": train_record_map,
            "content_dim": d_content,
        }
        pools.append(pool)

        print(f"[Client {cid}] corpus_n={len(records)} | q_emb={len(q_qids_local)} | "
              f"content_emb={len(content_emb_map)} | D_content={d_content}")

    return pools


# -------------------------------------------------------
# DataLoaders
# -------------------------------------------------------

def build_client_dataloaders(args, client_jsonl_files):
    client_ids = sorted(client_jsonl_files.keys())
    client_loaders = []

    for cid in client_ids:
        records = load_kilt_nq(str(client_jsonl_files[cid]))
        # Filter records that have at least one gold answer (needed for CE)
        records = [r for r in records if r.get("gold_answers")]
        dataset = KILTNQDataset(records)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_records,
        )
        client_loaders.append(loader)
        print(f"[DataLoader][Client {cid}] {len(dataset)} samples")

    return client_loaders


def build_val_loader(args):
    val_records = load_kilt_nq(args.val_jsonl)
    val_records = [r for r in val_records if r.get("gold_answers")]
    val_dataset = KILTNQDataset(val_records)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_records,
    )
    print(f"[Eval] val set: {len(val_dataset)} (with gold answers)")
    return val_loader, val_records


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    args = parse_args()
    fl_method = getattr(args, "fl_method", "fedRepRag")
    if fl_method.lower() != "fedreprag":
        raise ValueError(f"Unsupported fl_method for KILT-NQ: {fl_method} (only 'fedRepRag' supported)")

    print("[Paths]")
    print(f"  client_corpus_dir                  = {args.client_corpus_dir}")
    print(f"  global_question_embeddings_path    = {args.global_question_embeddings_path}")
    print(f"  global_content_embeddings_path     = {args.global_content_embeddings_path}")
    print(f"  val_jsonl                          = {args.val_jsonl}")

    log_dir = getattr(args, "ckpt_dir", None) or "./logs"
    os.makedirs(log_dir, exist_ok=True)
    dump_args_to_logdir(args, log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1) Load global embeddings
    # --------------------------------------------------
    print("[Embeddings] Loading global question embeddings ...")
    q_ckpt = torch.load(args.global_question_embeddings_path, map_location="cpu")
    global_q_emb = q_ckpt["embeddings"]
    if isinstance(global_q_emb, torch.Tensor):
        global_q_emb = global_q_emb.numpy()
    global_q_emb = global_q_emb.astype("float32")
    global_q_qids = [str(q) for q in q_ckpt["qids"]]
    print(f"[Embeddings] Q: {global_q_emb.shape}")

    print("[Embeddings] Loading global content embeddings ...")
    c_ckpt = torch.load(args.global_content_embeddings_path, map_location="cpu")
    global_c_emb = c_ckpt["embeddings"]
    if isinstance(global_c_emb, torch.Tensor):
        global_c_emb = global_c_emb.numpy()
    global_c_emb = global_c_emb.astype("float32")
    global_c_qids = [str(q) for q in c_ckpt["qids"]]
    print(f"[Embeddings] Content: {global_c_emb.shape}")

    # --------------------------------------------------
    # 2) Discover client corpus files
    # --------------------------------------------------
    client_jsonl_files = _index_client_jsonl_files(args.client_corpus_dir)
    print(f"[Clients] Found {len(client_jsonl_files)} client corpus files: "
          f"IDs={sorted(client_jsonl_files.keys())}")

    # --------------------------------------------------
    # 3) Build per-client retrieval pools
    # --------------------------------------------------
    client_retrieval_pools = build_client_pools(
        args, client_jsonl_files,
        global_q_emb, global_q_qids,
        global_c_emb, global_c_qids,
    )

    # --------------------------------------------------
    # 4) Build per-client DataLoaders + val loader
    # --------------------------------------------------
    client_loaders = build_client_dataloaders(args, client_jsonl_files)
    val_loader, val_records = build_val_loader(args)

    # --------------------------------------------------
    # 5) Load frozen Qwen3, add <RepRAG>
    # --------------------------------------------------
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
    print(f"[Model] Frozen Qwen3; hidden_size={H}, dtype={model_dtype}")

    # --------------------------------------------------
    # 6) Global MLP projector
    # --------------------------------------------------
    d_content = int(client_retrieval_pools[0]["content_dim"])
    mlp = PrefixProjector(
        input_size=d_content, output_size=H, projector_type="mlp2x_gelu", p_drop=0.1,
    ).to(device)
    mlp.requires_grad_(True)

    with torch.no_grad():
        dummy = torch.zeros(2, d_content, device=device)
        out = mlp(dummy)
        assert out.shape[-1] == H, f"MLP output must be H={H}, got {out.shape}"

    print(f"[MLP] PrefixProjector: D_content={d_content} -> H={H}")

    # --------------------------------------------------
    # 7) TensorBoard
    # --------------------------------------------------
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_root = getattr(args, "tb_dir", "./tb_logs")
    tb_log_dir = os.path.join(tb_root, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TB] Logging to {tb_log_dir}")

    # --------------------------------------------------
    # 8) Val query encoder
    # --------------------------------------------------
    def get_val_query_emb():
        print("[Eval] Encoding val queries on-the-fly ...")
        emb, q2r = encode_records_online(
            val_records, tokenizer=tokenizer, base_model=base_model,
            device=device, batch_size=getattr(args, "val_encode_batch_size", 8),
        )
        print(f"[Eval] Val query embeddings: {emb.shape}")
        return emb, q2r

    # --------------------------------------------------
    # 9) FL training
    # --------------------------------------------------
    if args.stage1_pretrain:
        print(f"[FL] Running KILT-NQ Stage1 FedRepRag")
        train_kilt_nq_fedRepRag_stage1(
            args=args,
            global_mlp=mlp,
            client_loaders=client_loaders,
            val_loader=val_loader,
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            client_retrieval_pools=client_retrieval_pools,
            writer=writer,
            reprag_token_id=reprag_token_id,
            get_val_query_emb_fn=get_val_query_emb,
        )

    if args.stage2_finetuning:
        print(f"[FL] Running KILT-NQ Stage2 FedRepRag")

        if getattr(args, "stage2_init_ckpt", None):
            ckpt2 = torch.load(args.stage2_init_ckpt, map_location="cpu")
            if "mlp_state" in ckpt2:
                mlp.load_state_dict(ckpt2["mlp_state"], strict=True)
                print(f"[Stage2] Loaded global MLP from {args.stage2_init_ckpt}")

        train_kilt_nq_fedRepRag_stage2(
            args=args,
            global_mlp=mlp,
            client_loaders=client_loaders,
            val_loader=val_loader,
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            model_dtype=model_dtype,
            client_retrieval_pools=client_retrieval_pools,
            writer=writer,
            reprag_token_id=reprag_token_id,
            get_val_query_emb_fn=get_val_query_emb,
        )

    writer.close()
    print("[Done]")


if __name__ == "__main__":
    main()