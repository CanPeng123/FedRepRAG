import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


REPRAG_TOKEN = "<RepRAG>"


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


def _get_slake_solution_text(ex: Dict[str, Any], include_answer_type: bool = True, include_base_type: bool = True,
include_kvqa_hint: bool = True, include_gold: bool = True) -> str:
    """
    Keep full retrieved example content, including question, but this is only
    used for retrieved examples, not for the base query prompt.
    """
    msg = build_solved_example_message_slake(
        ex,
        include_answer_type=include_answer_type,
        include_base_type=include_base_type,
        include_kvqa_hint=include_kvqa_hint,
        include_gold=include_gold,
    )
    content = msg.get("content", [])
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        return str(content[0].get("text", "")).strip()
    return ""


def build_solved_example_message_slake(ex: Dict[str, Any], include_answer_type: bool = True, include_base_type: bool = True, 
include_kvqa_hint: bool = True, include_gold: bool = True) -> Dict[str, Any]:
    q = (ex.get("question") or "").strip()
    a = (ex.get("gold_answer") or ex.get("answer") or "").strip()
    if not a and include_gold:
        a = "."

    parts = ["Solved example:"]
    if q:
        parts.append(f"Question: {q}")

    if include_answer_type:
        at = (ex.get("answer_type") or "").strip()
        if at:
            parts.append(f"Answer type: {at}.")

    bt = (ex.get("base_type") or "").strip()
    if include_base_type and bt:
        parts.append(f"Base type: {bt}.")

    if include_kvqa_hint and bt.lower() == "kvqa":
        triple = ex.get("triple", None)
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            rel = str(triple[1]).strip()
            tail = str(triple[2]).strip()
            if rel or tail:
                parts.append(f"Knowledge query: relation={rel}, tail={tail}")

    if include_gold:
        parts.append(f"Answer: {a}.")

    text = " ".join(parts)
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def build_teacher_evidence_texts(batch_neighbors_qids: List[List[str]], train_task_map: Dict[str, Dict[str, Any]], topk: int, 
max_chars: int = 5000, *, override_texts: Optional[List[List[str]]] = None, add_headers: bool = True, sep: str = "\n\n", 
include_answer_type: bool = True, include_base_type: bool = True, include_kvqa_hint: bool = True, include_gold: bool = True, 
deduplicate: bool = True) -> List[str]:
    """
    Keep full retrieved example content, including question.
    Deduplicate exact repeated retrieved examples if requested.
    """
    B = len(batch_neighbors_qids)
    if override_texts is not None:
        assert len(override_texts) == B, f"override_texts batch mismatch: {len(override_texts)} vs B={B}"

    texts: List[str] = []

    for bi, neigh in enumerate(batch_neighbors_qids):
        chunks: List[str] = []
        seen = set()
        used = 0

        for j, qid in enumerate(neigh[:topk]):
            if override_texts is not None:
                piece_core = (override_texts[bi][j] if j < len(override_texts[bi]) else "").strip()
            else:
                ex = train_task_map.get(str(qid))
                if ex is None:
                    piece_core = ""
                else:
                    piece_core = _get_slake_solution_text(
                        ex,
                        include_answer_type=include_answer_type,
                        include_base_type=include_base_type,
                        include_kvqa_hint=include_kvqa_hint,
                        include_gold=include_gold,
                    )

            if not piece_core:
                continue

            dedup_key = _normalize_ws(piece_core).lower()
            if deduplicate and dedup_key in seen:
                continue
            seen.add(dedup_key)

            piece = f"[Retrieved Example {len(chunks)+1}]\n{piece_core}" if add_headers else piece_core

            projected_len = used + len(piece) + (len(sep) if len(chunks) > 0 else 0)
            if projected_len > max_chars and len(chunks) > 0:
                break

            chunks.append(piece)
            used = projected_len

        ev = sep.join(chunks).strip()
        if not ev:
            ev = "."

        if len(ev) > max_chars:
            ev = ev[:max_chars].rstrip()

        texts.append(ev)

    return texts


def build_teacher_prompt_texts(processor, base_messages, evidence_texts):
    texts = []
    for msg, ev in zip(base_messages, evidence_texts):
        msg2 = dict(msg)

        if isinstance(msg2.get("content"), str):
            msg2["content"] = f"Context:\n{ev}\n\n" + msg2["content"]
        else:
            msg2["content"] = [{"type": "text", "text": f"Context:\n{ev}\n\n"}] + msg2["content"]

        prompt = processor.apply_chat_template([msg2], tokenize=False, add_generation_prompt=True)
        texts.append(prompt)
    return texts


def get_kl_loss_reprag_style(teacher_logits: torch.Tensor, student_logits: torch.Tensor, teacher_labels: torch.Tensor, 
student_labels: torch.Tensor, temperature: float = 1.0, distill_topk: Optional[int] = None) -> torch.Tensor:
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    T = float(temperature)

    t_log = teacher_logits[:, :-1, :]
    s_log = student_logits[:, :-1, :]
    t_lab = teacher_labels[:, 1:]
    s_lab = student_labels[:, 1:]

    t_mask = (t_lab != -100)
    s_mask = (s_lab != -100)

    if int(t_mask.sum().item()) != int(s_mask.sum().item()):
        raise RuntimeError(
            f"Teacher/student supervised prediction positions mismatch after shift: "
            f"{int(t_mask.sum().item())} vs {int(s_mask.sum().item())}. "
            f"Likely teacher prompt truncation or different answer length."
        )

    t_sel = t_log[t_mask]
    s_sel = s_log[s_mask]

    if distill_topk is not None:
        k = min(int(distill_topk), t_sel.size(-1))
        _, idx = torch.topk(t_sel, k=k, dim=-1)
        t_sel = torch.gather(t_sel, 1, idx)
        s_sel = torch.gather(s_sel, 1, idx)

    return loss_fct(
        F.log_softmax(s_sel / T, dim=-1),
        F.softmax(t_sel / T, dim=-1),
    ) * (T * T)


def build_student_prompt_texts(processor, base_messages, k_prefix: int):
    xr = " ".join([REPRAG_TOKEN] * k_prefix)
    texts = []

    for msg in base_messages:
        msg2 = dict(msg)

        prefix_text = f"[Retrieved Evidence]\n{xr}\n\n"

        if isinstance(msg2.get("content"), str):
            msg2["content"] = prefix_text + msg2["content"]
        else:
            msg2["content"] = [{"type": "text", "text": prefix_text}] + msg2["content"]

        texts.append(processor.apply_chat_template([msg2], tokenize=False, add_generation_prompt=True))

    return texts