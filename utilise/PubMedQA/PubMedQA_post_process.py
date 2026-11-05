import os
import re
import csv
import json
from typing import List, Dict, Any, Optional


# -------------------------------------------------------
# Answer parsing
# -------------------------------------------------------

def parse_answer_letter(output: str) -> str:
    """
    Extract a single answer letter (A-D) from model output.
    Falls back to 'A' if nothing is found.
    """
    if not output:
        return 'A'

    s = output.strip()

    # Exact single letter
    m = re.match(r'^\s*([A-D])\s*$', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # "Answer: X" or "Answer is X"
    m = re.search(r'[Aa]nswer\s*(?:is\s*)?[:\-]?\s*([A-D])\b', s)
    if m:
        return m.group(1).upper()

    # "Option X" or "(X)"
    m = re.search(r'[Oo]ption\s+([A-D])\b', s)
    if m:
        return m.group(1).upper()
    m = re.search(r'\(([A-D])\)', s)
    if m:
        return m.group(1).upper()

    # First occurrence of a standalone letter A-D
    m = re.search(r'\b([A-D])\b', s)
    if m:
        return m.group(1).upper()

    return 'A'


def is_correct(pred: str, gold: str) -> bool:
    return parse_answer_letter(pred) == gold.strip().upper()


# -------------------------------------------------------
# Result recording
# -------------------------------------------------------

def write_predictions_csv(
    results: List[Dict[str, Any]],
    out_dir: str,
    filename: str = 'pubmedqa_preds.csv',
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    fieldnames = [
        'qid', 'dataset', 'split', 'question',
        'gold_answer', 'pred_raw', 'pred_letter', 'correct',
        'retrieval_mode',
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    return path


def write_final_results(
    results: List[Dict[str, Any]],
    out_dir: str,
    retrieval_mode: str = 'none',
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    overall_acc = correct / total if total > 0 else 0.0

    # Per sub-dataset accuracy
    per_ds: Dict[str, Dict[str, int]] = {}
    for r in results:
        ds = r.get('dataset', 'unknown')
        if ds not in per_ds:
            per_ds[ds] = {'correct': 0, 'total': 0}
        per_ds[ds]['total'] += 1
        if r.get('correct', False):
            per_ds[ds]['correct'] += 1

    lines = [
        '===== PubMedQA Evaluation Results =====',
        f'Retrieval mode : {retrieval_mode}',
        f'Total questions: {total}',
        f'Overall accuracy: {overall_acc:.4f} ({correct}/{total})',
        '',
        'Per sub-dataset:',
    ]
    for ds, counts in sorted(per_ds.items()):
        acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        lines.append(f"  {ds:<12}: {acc:.4f}  ({counts['correct']}/{counts['total']})")

    text = '\n'.join(lines)
    print(text)

    with open(os.path.join(out_dir, 'final_results.txt'), 'w') as f:
        f.write(text + '\n')

    # Also save per-dataset JSON for easy downstream use
    summary = {
        'retrieval_mode': retrieval_mode,
        'total': total,
        'overall_accuracy': round(overall_acc, 4),
        'per_dataset': {
            ds: {
                'accuracy': round(c['correct'] / c['total'], 4) if c['total'] > 0 else 0.0,
                'correct': c['correct'],
                'total': c['total'],
            }
            for ds, c in per_ds.items()
        },
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def save_arguments(args, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'argument.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')