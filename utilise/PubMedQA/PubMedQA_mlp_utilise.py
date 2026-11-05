"""
PubMedQA-specific MLP training utilities.

Unlike SLAKE (vision-language, free-form answers), PubMedQA/PubMedQA is:
  - Text-only (Qwen3-8B, AutoTokenizer + AutoModelForCausalLM)
  - Yes/No/Maybe answers mapped to option letters A/B/C
  - Corpus = PubMed passages (title + abstract)
"""

import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

REPRAG_TOKEN = "<RepRAG>"

PUBMEDQA_OPTIONS     = {"yes": "A", "no": "B", "maybe": "C"}
PUBMEDQA_OPTIONS_INV = {"A": "yes", "B": "no", "C": "maybe"}


# -------------------------------------------------------
# Answer helpers
# -------------------------------------------------------

def gold_to_option_letter(gold: str) -> str:
    """Convert 'yes'/'no'/'maybe' to 'A'/'B'/'C'."""
    return PUBMEDQA_OPTIONS.get(gold.strip().lower(), gold.strip().upper())


def build_answer_token_ids_pubmedqa(tokenizer, gold_texts: List[str]) -> List[List[int]]:
    """Tokenise gold answers (option letters A/B/C) without special tokens."""
    letters = [" " + gold_to_option_letter(g) for g in gold_texts]
    tok = tokenizer(letters, add_special_tokens=False, padding=False)
    return tok["input_ids"]


def compute_pubmedqa_batch_acc(
    batch_tasks: List[Dict[str, Any]],
    pred_texts: List[str],
) -> float:
    """Exact-match accuracy on A/B/C option letters."""
    correct = 0
    for t, pred in zip(batch_tasks, pred_texts):
        gold_letter = gold_to_option_letter(t.get("gold_answer", ""))
        pred_clean = pred.strip().upper()
        m = re.search(r"[ABC]", pred_clean)
        pred_letter = m.group(0) if m else pred_clean[:1]
        if pred_letter == gold_letter:
            correct += 1
    return correct / max(len(batch_tasks), 1)


# -------------------------------------------------------
# Base message builder (text-only)
# -------------------------------------------------------

def build_mcq_vqa_message_text(task: Dict[str, Any]) -> Dict[str, Any]:
    """Plain-text VQA message for PubMedQA (no images)."""
    opts = "\n".join(f"{k}. {v}" for k, v in
                     {"A": "yes", "B": "no", "C": "maybe"}.items())
    content = (
        "Answer the following biomedical yes/no/maybe question.\n"
        "Reply with ONLY the option letter (A, B, or C). Do not explain.\n\n"
        f"Question: {task['question']}\n\nOptions:\n{opts}\n\nAnswer:"
    )
    return {"role": "user", "content": content}


# -------------------------------------------------------
# Corpus evidence formatter
# -------------------------------------------------------

def build_corpus_evidence_text(entry: Dict[str, Any], max_chars: int = 2000) -> str:
    """Format a retrieved corpus entry as evidence text.

    All pool entries (pqa_labeled, pqa_unlabeled, pubmed_corpus) now share
    the same title/content fields sourced from the PubMed corpus JSONL.
    """
    title   = entry.get("title",   "").strip()
    content = entry.get("content", "").strip()
    text = f"{title}\n{content}".strip() if title else content
    return text[:max_chars]


# -------------------------------------------------------
# Teacher prompt builders (passage text injected literally)
# -------------------------------------------------------

def build_teacher_evidence_texts_pubmedqa(
    batch_neighbors_qids: List[List[str]],
    corpus_task_map: Dict[str, Dict[str, Any]],
    topk: int,
    max_chars: int = 3000,
) -> List[str]:
    """Build teacher evidence strings from retrieved passage title+abstract."""
    texts = []
    for neighbor_qids in batch_neighbors_qids:
        chunks = []
        for j, qid in enumerate(neighbor_qids[:topk]):
            entry = corpus_task_map.get(str(qid))
            if entry is None:
                continue
            piece = build_corpus_evidence_text(entry)
            if piece:
                chunks.append(f"[Passage {j+1}]\n{piece}")
        ev = "\n\n".join(chunks).strip() or "."
        if len(ev) > max_chars:
            ev = ev[:max_chars].rstrip()
        texts.append(ev)
    return texts


def build_teacher_prompt_texts_pubmedqa(
    tokenizer,
    base_messages: List[Dict[str, Any]],
    evidence_texts: List[str],
) -> List[str]:
    """Build teacher prompts: prepend passage evidence to each question."""
    prompts = []
    for msg, ev in zip(base_messages, evidence_texts):
        msg2 = dict(msg)
        msg2["content"] = f"Reference passages:\n{ev}\n\n{msg2['content']}"
        prompt = tokenizer.apply_chat_template(
            [msg2], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


# -------------------------------------------------------
# Student prompt builder (<RepRAG> injection)
# -------------------------------------------------------

def build_student_prompt_texts_pubmedqa(
    tokenizer,
    base_messages: List[Dict[str, Any]],
    k_prefix: int,
) -> List[str]:
    """Build student prompts: prepend K <RepRAG> tokens before the question."""
    xr = " ".join([REPRAG_TOKEN] * k_prefix)
    prompts = []
    for msg in base_messages:
        msg2 = dict(msg)
        msg2["content"] = f"[Retrieved Evidence]\n{xr}\n\n{msg2['content']}"
        prompts.append(
            tokenizer.apply_chat_template(
                [msg2], tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )
    return prompts


# -------------------------------------------------------
# KL distillation loss
# -------------------------------------------------------

def get_kl_loss_pubmedqa(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    student_labels: torch.Tensor,
    temperature: float = 1.0,
    distill_topk: Optional[int] = None,
) -> torch.Tensor:
    T = float(temperature)
    t_log = teacher_logits[:, :-1, :]
    s_log = student_logits[:, :-1, :]
    t_lab = teacher_labels[:, 1:]
    s_lab = student_labels[:, 1:]

    t_mask = t_lab != -100
    s_mask = s_lab != -100

    if int(t_mask.sum().item()) != int(s_mask.sum().item()):
        raise RuntimeError(
            f"Teacher/student supervised token mismatch: "
            f"{int(t_mask.sum().item())} vs {int(s_mask.sum().item())}"
        )

    t_sel = t_log[t_mask]
    s_sel = s_log[s_mask]

    if distill_topk is not None:
        k = min(int(distill_topk), t_sel.size(-1))
        _, idx = torch.topk(t_sel, k=k, dim=-1)
        t_sel = torch.gather(t_sel, 1, idx)
        s_sel = torch.gather(s_sel, 1, idx)

    return torch.nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(s_sel / T, dim=-1),
        F.softmax(t_sel / T, dim=-1),
    ) * (T * T)
