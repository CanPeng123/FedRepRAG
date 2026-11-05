import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

REPRAG_TOKEN = "<RepRAG>"


def _get_solution_text(ex: Dict[str, Any]) -> str:
    # unify all possible fields you may have
    sol = (ex.get("solution") or ex.get("rationale") or "").strip()
    return sol


def build_teacher_evidence_texts(
    batch_neighbors_qids: List[List[str]],
    train_task_map: Dict[str, Dict[str, Any]],
    topk: int,
    max_chars: int = 1200,
    *,
    # Optional: if you want to pass already-extracted texts like your override_texts
    override_texts: Optional[List[List[str]]] = None,
    # Whether to label each retrieved item (helps the model parse multiple chunks)
    add_headers: bool = True,
    # Separator between neighbor solutions
    sep: str = "\n\n",
) -> List[str]:
    """
    Build teacher evidence strings using ONLY solution/rationale text.

    Returns:
      List[str] length B. Each string is concatenated top-k solution texts.

    Notes:
      - Mirrors your stage1/stage2 retrieval embedding source (solution-only).
      - Keeps a minimal "." if nothing is available (avoid empty evidence).
      - Applies max_chars cap at the final string level.
    """
    B = len(batch_neighbors_qids)
    if override_texts is not None:
        assert len(override_texts) == B, f"override_texts batch mismatch: {len(override_texts)} vs B={B}"

    texts: List[str] = []
    for bi, neigh in enumerate(batch_neighbors_qids):
        chunks: List[str] = []
        used = 0

        for j, qid in enumerate(neigh[:topk]):
            if override_texts is not None:
                sol = (override_texts[bi][j] if j < len(override_texts[bi]) else "").strip()
            else:
                ex = train_task_map.get(str(qid))
                sol = _get_solution_text(ex) if ex is not None else ""

            if not sol:
                continue

            if add_headers:
                piece = f"[Retrieved Solution {len(chunks)+1}]\n{sol}"
            else:
                piece = sol

            # early stop if max_chars would be exceeded a lot
            # (still do a final hard cap below)
            used += len(piece) + len(sep)
            chunks.append(piece)
            if used >= max_chars:
                break

        ev = sep.join(chunks).strip()
        if not ev:
            ev = "."

        if len(ev) > max_chars:
            ev = ev[:max_chars].rstrip()

        texts.append(ev)

    return texts

"""
def build_teacher_evidence_texts_full(batch_neighbors_qids: List[List[str]], train_task_map: Dict[str, Dict[str, Any]], topk: int, 
max_chars: int = 1200) -> List[str]:
    texts = []
    for neigh in batch_neighbors_qids:
        chunks = []
        for qid in neigh[:topk]:
            rt = train_task_map.get(str(qid))
            if rt is None:
                continue
            q = (rt.get("question") or "").strip()
            ctx = (rt.get("context") or "").strip()
            sol = (rt.get("solution") or rt.get("rationale") or "").strip()

            piece = ""
            if q:   piece += f"Q: {q}\n"
            if ctx: piece += f"Context: {ctx}\n"
            if sol: piece += f"Solution: {sol}\n"
            if piece:
                chunks.append(piece.strip())

        ev = "\n\n".join(chunks).strip()
        if len(ev) > max_chars:
            ev = ev[:max_chars].rstrip()
        texts.append(ev)
    return texts
"""

def build_teacher_prompt_texts(processor, base_messages, evidence_texts):
    texts = []
    for msg, ev in zip(base_messages, evidence_texts):
        msg2 = dict(msg)  # shallow copy
        # If your msg structure is {"role": "...", "content": ...}, adapt accordingly.
        # For Qwen-style multi-modal messages, you might have msg["content"] list.
        # Here is a safe generic approach for string content:
        if isinstance(msg2.get("content"), str):
            msg2["content"] = f"[Retrieved Evidence]\n{ev}\n\n" + msg2["content"]
        else:
            # If content is list of segments, prepend a text segment
            msg2["content"] = [{"type": "text", "text": f"[Retrieved Evidence]\n{ev}\n\n"}] + msg2["content"]

        prompt = processor.apply_chat_template([msg2], tokenize=False, add_generation_prompt=True)
        texts.append(prompt)
    return texts


def get_kl_loss_reprag_style(teacher_logits: torch.Tensor, student_logits: torch.Tensor, teacher_labels: torch.Tensor, 
student_labels: torch.Tensor, temperature: float = 1.0, distill_topk: Optional[int] = None) -> torch.Tensor:
    """
    Correct autoregressive alignment:
    - CE uses shift_logits = logits[:, :-1], shift_labels = labels[:, 1:]
    - So KL must be computed on the same shifted positions where shift_labels != -100.
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    T = float(temperature)

    # shift to "prediction positions"
    t_log = teacher_logits[:, :-1, :]     # predicts labels[:, 1:]
    s_log = student_logits[:, :-1, :]
    t_lab = teacher_labels[:, 1:]
    s_lab = student_labels[:, 1:]

    # mask on prediction positions
    t_mask = (t_lab != -100)
    s_mask = (s_lab != -100)

    if int(t_mask.sum().item()) != int(s_mask.sum().item()):
        raise RuntimeError(
            f"Teacher/student supervised prediction positions mismatch after shift: "
            f"{int(t_mask.sum().item())} vs {int(s_mask.sum().item())}. "
            f"Likely teacher prompt truncation or different answer length."
        )

    # select [N, V]
    t_sel = t_log[t_mask]   # boolean mask selects last dim automatically => [N, V]
    s_sel = s_log[s_mask]

    if distill_topk is not None:
        _, idx = torch.topk(t_sel, k=int(distill_topk), dim=-1)
        t_sel = torch.gather(t_sel, 1, idx)
        s_sel = torch.gather(s_sel, 1, idx)

    return loss_fct(
        F.log_softmax(s_sel / T, dim=-1),
        F.softmax(t_sel / T, dim=-1),
    ) * (T * T)


def build_student_prompt_texts(processor, base_messages, k_prefix):
    xr = (REPRAG_TOKEN + " ") * k_prefix
    texts = []
    for msg in base_messages:
        msg2 = dict(msg)
        prefix = f"[Retrieved Evidence]\n{xr}\n\n"
        if isinstance(msg2.get("content"), str):
            msg2["content"] = prefix + msg2["content"]
        else:
            msg2["content"] = [{"type":"text","text": prefix}] + msg2["content"]
        texts.append(processor.apply_chat_template([msg2], tokenize=False, add_generation_prompt=True))
    return texts