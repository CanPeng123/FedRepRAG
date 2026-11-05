import os
from typing import Dict, Any, Optional
from collections import defaultdict
import datetime


def _safe_acc(correct: int, total: int) -> float:
    return (100.0 * correct / total) if total > 0 else 0.0

def norm_cat(x) -> str:
    """Normalize category values to stable strings."""
    if x is None:
        return "UNKNOWN"
    s = str(x).strip()
    return s if s else "UNKNOWN"

def make_stats_bucket():
    # returns dict: key -> [correct, total]
    return defaultdict(lambda: [0, 0])

def update_stats(stats_bucket, key: str, is_correct: int):
    stats_bucket[key][0] += int(is_correct)
    stats_bucket[key][1] += 1

def format_breakdown_table(title: str, stats_bucket, topk: int = 50) -> str:
    """
    stats_bucket: dict[key] = [correct, total]
    """
    lines = [f"----- {title} -----"]
    items = [(k, v[0], v[1]) for k, v in stats_bucket.items()]
    items.sort(key=lambda t: (t[2], _safe_acc(t[1], t[2])), reverse=True)

    for k, c, n in items[:topk]:
        lines.append(f"{k:<18} : {c:>5}/{n:<5} = {_safe_acc(c, n):>6.2f}%")

    if len(items) > topk:
        lines.append(f"... ({len(items)-topk} more)")
    lines.append("")  # blank line
    return "\n".join(lines)


def write_final_results_txt(out_dir: str, args, split: Optional[str], lang: str, total_questions: int, correct_all: int, seen_all: int, 
correct_open: int, seen_open: int, correct_closed: int, seen_closed: int, out_csv_path: str, stats: dict):
    
    out_path = os.path.join(out_dir, "final_results.txt")

    lines = []
    lines.append("===== SLAKE VQA Final Results =====")
    lines.append(f"time            : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"split           : {split or 'all'}")
    lines.append(f"lang            : {lang}")
    lines.append("")
    lines.append("----- Counts -----")
    lines.append(f"total_loaded     : {total_questions}")
    lines.append(f"evaluated_total  : {seen_all}")
    lines.append(f"open_questions   : {seen_open}")
    lines.append(f"closed_questions : {seen_closed}")
    lines.append("")
    lines.append("----- Accuracy -----")
    lines.append(f"overall          : {correct_all}/{seen_all} = {_safe_acc(correct_all, seen_all):.2f}%")
    lines.append(f"open             : {correct_open}/{seen_open} = {_safe_acc(correct_open, seen_open):.2f}%")
    lines.append(f"closed           : {correct_closed}/{seen_closed} = {_safe_acc(correct_closed, seen_closed):.2f}%")
    lines.append("")

    # breakdowns (ALL / OPEN / CLOSED)
    lines.append("----- Breakdown Performance -----")
    for field in ["modality", "location", "content_type", "base_type"]:
        lines.append(f"==================== {field} ====================")
        lines.append(format_breakdown_table(f"{field} (ALL)",    stats[field]["all"]))
        lines.append(format_breakdown_table(f"{field} (OPEN)",   stats[field]["open"]))
        lines.append(format_breakdown_table(f"{field} (CLOSED)", stats[field]["closed"]))

    lines.append("----- Run args (key) -----")
    lines.append(f"model            : {getattr(args, 'model', '')}")
    lines.append(f"batch_size       : {getattr(args, 'batch_size', '')}")
    lines.append(f"do_sample        : {getattr(args, 'do_sample', False)}")
    lines.append(f"temperature      : {getattr(args, 'temperature', None)}")
    lines.append(f"max_new_tokens   : {getattr(args, 'max_new_tokens', None)}")
    lines.append(f"num_beams        : {getattr(args, 'num_beams', None)}")
    lines.append(f"num_parts/idx    : {getattr(args, 'num_parts', 1)}/{getattr(args, 'idx', 0)}")
    lines.append("")
    lines.append("----- Output files -----")
    lines.append(f"pred_csv         : {out_csv_path}")
    lines.append("===================================")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Final] Results saved to: {out_path}")

