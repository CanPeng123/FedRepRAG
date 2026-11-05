"""
Shared helpers for KILT-NQ MLP training (stage 1 + stage 2).
"""
import re
import string
from typing import List, Dict, Any

REPRAG_TOKEN = "<RepRAG>"


def build_kilt_nq_vqa_message_text(record: Dict[str, Any]) -> Dict[str, str]:
    """Base VQA message (no retrieval) used as the student prompt skeleton."""
    return {
        "role": "user",
        "content": (
            "Answer the following question with a short, concise phrase or entity. "
            "Do not explain — just give the answer.\n\n"
            f"Question: {record['question']}\nAnswer:"
        ),
    }


def build_student_prompt_texts_kilt_nq(
    tokenizer,
    base_messages: List[Dict[str, str]],
    k_prefix: int,
) -> List[str]:
    """
    Build student prompt strings with k_prefix <RepRAG> tokens prepended
    before the question, so the MLP injection lands before the query text.
    """
    reprag_str = " ".join([REPRAG_TOKEN] * k_prefix)
    prompts = []
    for msg in base_messages:
        content = f"[Retrieved Evidence]\n{reprag_str}\n\n{msg['content']}"
        augmented = [{"role": "user", "content": content}]
        text = tokenizer.apply_chat_template(
            augmented,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)
    return prompts


def build_answer_token_ids_kilt_nq(
    tokenizer,
    records: List[Dict[str, Any]],
    max_tokens: int = 16,
) -> List[List[int]]:
    """Tokenize the first gold answer for each record (stage-2 CE target)."""
    texts = []
    for r in records:
        ans = r["gold_answers"][0] if r.get("gold_answers") else ""
        texts.append(ans.strip() or ".")
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_tokens,
    )
    return enc["input_ids"]


def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def compute_kilt_nq_batch_acc(
    records: List[Dict[str, Any]],
    pred_texts: List[str],
) -> float:
    """Normalized exact match accuracy over a batch."""
    correct = 0
    for r, pred in zip(records, pred_texts):
        pred_norm = _normalize_answer(pred)
        golds = r.get("gold_answers") or []
        if any(_normalize_answer(g) == pred_norm for g in golds):
            correct += 1
    return correct / len(records) if records else 0.0


def compute_kilt_nq_em_batch(
    pred_ids_batch: List[List[int]],
    gold_ids_batch: List[List[int]],
) -> float:
    """Token-level exact match (all tokens must match) over a batch."""
    correct = sum(p == g for p, g in zip(pred_ids_batch, gold_ids_batch))
    return correct / len(pred_ids_batch) if pred_ids_batch else 0.0