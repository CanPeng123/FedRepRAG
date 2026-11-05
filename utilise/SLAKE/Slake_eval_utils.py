import re
from typing import List, Dict, Any, Tuple, Optional
from VQA.utilise.SLAKE.Slake_post_process import normalize_answer_with_numbers, answers_match
from VQA.utilise.SLAKE.Slake_result_record import norm_cat, update_stats


# =========================================================
# Helper
# =========================================================
_ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer\s*[:\-]\s*|answer\s+is\s+)", re.IGNORECASE)

def strip_answer_prefix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # take first line only (as you already do)
    s = s.splitlines()[0].strip()
    # remove leading "Answer: " / "answer - " / "Answer is "
    s = _ANSWER_PREFIX_RE.sub("", s).strip()
    # optionally strip trailing period for short closed answers (Yes./No.)
    # s = s.rstrip()
    return s


# =========================================================
# Evaluation
# =========================================================
def score_and_record_batch(batch_tasks: List[Dict[str, Any]], pred_texts: List[str], rows: List[Dict[str, Any]], stats: Dict[str, Dict[str, Any]], state: Dict[str, int], 
retrieval_mode: str, *, verbose: bool = False) -> None:
    """
    Updates in-place:
      - rows: appended
      - stats: updated
      - state: counters updated

    state keys required:
      correct_all, seen_all, correct_open, seen_open, correct_closed, seen_closed
    """
    assert len(batch_tasks) == len(pred_texts), "batch size mismatch"

    for t, pred in zip(batch_tasks, pred_texts):
        gold = t["gold_answer"]

        pred_n = normalize_answer_with_numbers(pred)
        gold_n = normalize_answer_with_numbers(gold)

        print("---------------------------------------------------")
        print("score_and_record_batch: pred: {0}, pred_n: {1}".format(pred, pred_n))
        print("score_and_record_batch: gold: {0}, gold_n: {1}".format(gold, gold_n))
        print("---------------------------------------------------")
        is_correct = int(answers_match(pred, gold) or answers_match(pred_n, gold_n))

        at = (t.get("answer_type") or "").upper()

        m   = norm_cat(t.get("modality"))
        loc = norm_cat(t.get("location"))
        ct  = norm_cat(t.get("content_type"))
        bt  = norm_cat(t.get("base_type"))

        # ALL
        update_stats(stats["modality"]["all"], m, is_correct)
        update_stats(stats["location"]["all"], loc, is_correct)
        update_stats(stats["content_type"]["all"], ct, is_correct)
        update_stats(stats["base_type"]["all"], bt, is_correct)

        # OPEN/CLOSED
        if at == "OPEN":
            update_stats(stats["modality"]["open"], m, is_correct)
            update_stats(stats["location"]["open"], loc, is_correct)
            update_stats(stats["content_type"]["open"], ct, is_correct)
            update_stats(stats["base_type"]["open"], bt, is_correct)
        elif at == "CLOSED":
            update_stats(stats["modality"]["closed"], m, is_correct)
            update_stats(stats["location"]["closed"], loc, is_correct)
            update_stats(stats["content_type"]["closed"], ct, is_correct)
            update_stats(stats["base_type"]["closed"], bt, is_correct)

        # counters
        state["correct_all"] += is_correct
        state["seen_all"] += 1
        if at == "OPEN":
            state["correct_open"] += is_correct
            state["seen_open"] += 1
        elif at == "CLOSED":
            state["correct_closed"] += is_correct
            state["seen_closed"] += 1

        rows.append({
            "qid": t["qid"],
            "split": t.get("split", ""),
            "question": t.get("question", ""),
            "image_path": t.get("image_path", ""),
            "answer_type": t.get("answer_type", ""),
            "base_type": t.get("base_type", ""),
            "content_type": t.get("content_type", ""),
            "modality": t.get("modality", ""),
            "location": t.get("location", ""),
            "pred_answer": pred,
            "pred_norm": pred_n,
            "gold_answer": gold,
            "gold_norm": gold_n,
            "correct": is_correct,
            "retrieval_mode": retrieval_mode,
        })

        if verbose:
            print(f"[qid={t['qid']}] type={at} | Q: {t.get('question','')}")
            print(f"  pred : '{pred}' (norm='{pred_n}')")
            print(f"  gold : '{gold}' (norm='{gold_n}')")
            print(f"  ok?  : {is_correct}")


def compute_slake_batch_acc(batch_tasks: List[Dict[str, Any]],pred_texts: List[str], *, return_breakdown: bool = False) -> float | Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    SLAKE-style accuracy following score_and_record_batch's logic.
    - Overall acc (ALL)
    - Optional OPEN/CLOSED breakdown + counters (state-like)

    Returns:
      - if return_breakdown=False: acc_all (float)
      - else: (acc_all, breakdown, state)
    """
    assert len(batch_tasks) == len(pred_texts), "batch size mismatch"

    state = {
        "correct_all": 0, "seen_all": 0,
        "correct_open": 0, "seen_open": 0,
        "correct_closed": 0, "seen_closed": 0,
    }

    for t, pred in zip(batch_tasks, pred_texts):
        gold = t.get("gold_answer", t.get("answer", "")) or ""

        pred = (pred or "")
        gold = (gold or "")

        pred_n = normalize_answer_with_numbers(pred)
        gold_n = normalize_answer_with_numbers(gold)

        # score_and_record_batch does: int(answers_match(pred, gold))
        # Keep that behavior, but add normalized fallback to be safer.
        is_correct = int(answers_match(pred, gold) or answers_match(pred_n, gold_n))

        at = (t.get("answer_type") or "").upper()

        state["correct_all"] += is_correct
        state["seen_all"] += 1

        if at == "OPEN":
            state["correct_open"] += is_correct
            state["seen_open"] += 1
        elif at == "CLOSED":
            state["correct_closed"] += is_correct
            state["seen_closed"] += 1

    acc_all = state["correct_all"] / state["seen_all"] if state["seen_all"] > 0 else 0.0

    if not return_breakdown:
        return float(acc_all)

    acc_open = state["correct_open"] / state["seen_open"] if state["seen_open"] > 0 else 0.0
    acc_closed = state["correct_closed"] / state["seen_closed"] if state["seen_closed"] > 0 else 0.0

    breakdown = {
        "acc_all": float(acc_all),
        "acc_open": float(acc_open),
        "acc_closed": float(acc_closed),
    }
    return float(acc_all), breakdown, state