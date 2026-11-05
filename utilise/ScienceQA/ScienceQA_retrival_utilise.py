import torch
import numpy as np
from PIL.Image import Image as PILImage
from typing import List, Dict, Union, Literal, Optional
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

from VQA.utilise.data_process import load_image, load_pil_or_none
from VQA.utilise.ScienceQA.ScienceQA_data_process import discover_q_choice_by_convention, load_captions

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_prefix_layer0_for_batch_from_retrieved(batch_neighbors_qids: list[list[str]], train_task_map: Dict[str, dict], model, 
processor, device, max_tokens_per_example: int = 65536) -> torch.Tensor:
    """
    For each test example, build a prefix sequence by concatenating token embeddings of its retrieved solved examples.

    Returns: prefix_embs [B, Lmax, H], where Lmax is max prefix length in batch (padded with zeros).
    """
    prefix_seqs = []
    H = model.config.hidden_size

    for neighbor_qids in batch_neighbors_qids:
        # For this test item, gather embeddings of each neighbor example
        example_embs_list = []

        for qid in neighbor_qids:
            ex = train_task_map.get(str(qid))
            if ex is None:
                continue
            msg = build_solved_example_message(ex)
            tok_embs = embed_single_message_tokens(msg, model, processor, device, layer="embeddings", strip_padding=True)  # [T, H]

            # Truncate per-example length if desired
            tok_embs = tok_embs[:max_tokens_per_example]
            example_embs_list.append(tok_embs)

        if not example_embs_list:
            # no neighbors → empty prefix
            prefix_seqs.append(torch.zeros(0, H, device=device))
        else:
            # Concatenate all neighbor sequences into one prefix [L, H]
            prefix_seqs.append(torch.cat(example_embs_list, dim=0))

    # Now pad to same length across batch
    B = len(prefix_seqs)
    Lmax = max(seq.size(0) for seq in prefix_seqs) if prefix_seqs else 0
    if Lmax == 0:
        return torch.zeros(B, 0, H, device=device)

    padded = torch.zeros(B, Lmax, H, device=device)
    for i, seq in enumerate(prefix_seqs):
        L = seq.size(0)
        padded[i, :L] = seq

    return padded  # [B, Lmax, H]


def build_prefix_last_layer_for_batch_from_retrieved(batch_neighbors_qids: List[List[str]], batch_neighbors_scores: List[List[float]],
train_task_map: Dict[str, dict], model, processor, device, max_tokens_per_example: int = 65536, l2_normalize: bool = True,
output_mode: Literal["weighted_avg", "concat"] = "concat", k_concat: Optional[int] = None, pad_value: float = 0.0) -> torch.Tensor:
    """
    Build retrieved-example embeddings per test item.

    Modes:
      - output_mode="weighted_avg": returns [B, 1, H] using softmax-weighted average over neighbors
      - output_mode="concat":       returns [B, k_concat, H] by stacking per-neighbor embeddings (pads/truncates)

    Notes for concat:
      - Uses neighbors in the same order as batch_neighbors_qids.
      - If fewer than k_concat valid neighbors, pads with pad_value.
      - If more than k_concat, truncates.
      - If l2_normalize=True, each neighbor embedding is L2-normalized (padding stays pad_value).
    """
    B = len(batch_neighbors_qids)
    assert B == len(batch_neighbors_scores), "qids and scores batch size mismatch"

    if output_mode == "concat":
        if k_concat is None:
            k_concat = max((len(x) for x in batch_neighbors_qids), default=0)
        if k_concat <= 0:
            raise ValueError("k_concat must be a positive int for output_mode='concat'")

    # Underlying HF model (unwrap Deepspeed if needed)
    base_model = model.module if hasattr(model, "module") else model
    H = base_model.config.hidden_size

    pooled_embs = []  # each element: [1, H] for weighted_avg, or [k_concat, H] for concat

    for neighbor_qids, neighbor_scores in zip(batch_neighbors_qids, batch_neighbors_scores):
        example_embs_list = []  # list of [H] embeddings (valid neighbors only)

        # --- collect neighbor embeddings (in provided order) ---
        for qid in neighbor_qids:
            ex = train_task_map.get(str(qid))
            if ex is None:
                continue
            msg = build_solved_example_message(ex)
            neigh_emb = embed_single_message_tokens(msg, model, processor, device, layer="last", average_hidden=True,
            add_generation_prompt=True)  # [H]
            example_embs_list.append(neigh_emb)

            # in concat mode, stop once we have enough
            if output_mode == "concat" and len(example_embs_list) >= k_concat:
                break

        # --- handle no valid neighbors ---
        if not example_embs_list:
            if output_mode == "weighted_avg":
                pooled_embs.append(torch.full((1, H), pad_value, device=device))
            else:
                pooled_embs.append(torch.full((k_concat, H), pad_value, device=device))
            continue

        # ================= weighted_avg =================
        if output_mode == "weighted_avg":
            # Stack neighbor embeddings → [K, H]
            ex_embs = torch.stack(example_embs_list, dim=0)  # [K, H]
            K = ex_embs.size(0)

            # Match scores length to K (some neighbors may have been skipped)
            scores_tensor = torch.tensor(neighbor_scores[:K], dtype=torch.float32, device=device)  # [K]

            # Softmax over neighbors to get weights
            weights = F.softmax(scores_tensor, dim=0)  # [K]

            # Weighted average over neighbors
            ret_emb = (ex_embs * weights.unsqueeze(-1)).sum(dim=0)  # [H]

            if l2_normalize:
                ret_emb = F.normalize(ret_emb, p=2, dim=0)

            pooled_embs.append(ret_emb.unsqueeze(0))  # [1, H]

        # ================= concat =================
        elif output_mode == "concat":
            # We need exactly k_concat rows: pad if fewer.
            K = len(example_embs_list)
            if K == 0:
                ex_embs = torch.full((k_concat, H), pad_value, device=device)
            elif K < k_concat:
                pad = torch.full((k_concat - K, H), pad_value, device=device)
                ex_embs = torch.cat([torch.stack(example_embs_list, dim=0), pad], dim=0)  # [k_concat, H]
            else:
                ex_embs = torch.stack(example_embs_list[:k_concat], dim=0)  # [k_concat, H]
            
            if l2_normalize:
                ex_embs = F.normalize(ex_embs, p=2, dim=-1)  # row-wise
            
            pooled_embs.append(ex_embs)
        
        else:
            raise ValueError(f"Invalid output_mode: {output_mode}. Use 'weighted_avg' or 'concat'.")

    ret_emb_batch = torch.stack(pooled_embs, dim=0).float().to(device)
    return ret_emb_batch  # weighted_avg: [B,1,H], concat: [B,k_concat,H]
