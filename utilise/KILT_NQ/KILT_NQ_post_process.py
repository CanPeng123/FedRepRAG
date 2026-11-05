import os
import re
import csv
import json
import string
from collections import Counter
from typing import List, Dict, Any


# -------------------------------------------------------
# Answer normalisation (standard NQ/SQuAD convention)
# -------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def exact_match(pred: str, gold_answers: List[str]) -> bool:
    pred_n = _normalize(pred)
    return any(_normalize(g) == pred_n for g in gold_answers)


def token_f1(pred: str, gold_answers: List[str]) -> float:
    """Best F1 over all gold answers (token-level)."""
    pred_tokens = _normalize(pred).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            continue
        p = num_common / len(pred_tokens) if pred_tokens else 0.0
        r = num_common / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        best = max(best, f1)
    return best


def extract_short_answer(raw: str) -> str:
    """
    Strip chain-of-thought / reasoning preamble.
    Take the first non-empty line after stripping, capped at 100 chars.
    """
    raw = raw.strip()
    # If the model prefixes with "Answer:" strip it
    raw = re.sub(r'^[Aa]nswer\s*:\s*', '', raw)
    # Take the first line
    first_line = raw.split('\n')[0].strip()
    return first_line[:100]


# -------------------------------------------------------
# Output writers
# -------------------------------------------------------

def write_predictions_csv(results: List[Dict[str, Any]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'kilt_nq_preds.csv')
    fieldnames = ['id', 'split', 'question', 'gold_answers', 'pred_raw', 'pred_clean', 'em', 'f1']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = dict(r)
            row['gold_answers'] = ' | '.join(r['gold_answers'])
            writer.writerow(row)
    return path


def write_final_results(results: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    total = len(results)
    em_score = sum(r['em'] for r in results) / total if total else 0.0
    f1_score = sum(r['f1'] for r in results) / total if total else 0.0

    lines = [
        '===== KILT-NQ Evaluation Results =====',
        f'Total questions : {total}',
        f'Exact Match (EM): {em_score:.4f}',
        f'Token F1        : {f1_score:.4f}',
    ]
    text = '\n'.join(lines)
    print(text)

    with open(os.path.join(out_dir, 'final_results.txt'), 'w') as f:
        f.write(text + '\n')

    summary = {'total': total, 'exact_match': round(em_score, 4), 'f1': round(f1_score, 4)}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def save_arguments(args, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'argument.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')