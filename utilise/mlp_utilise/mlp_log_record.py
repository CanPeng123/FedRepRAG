import os
import torch
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime


# --------------------------
# Debug for printing
# --------------------------
def debug_print_answer_pred_span_packed(processor, outputs, input_ids, prefix_len, prompt_lens, ans_lens):
    """
    Packed version: each sample has its own prompt_len and answer start due to left padding.
    Prints predicted tokens for the whole answer span.
    """
    logits = outputs.logits  # [B, prefix_len+T_total, vocab]
    B, T_total = input_ids.shape

    total_lens = (prompt_lens + ans_lens).tolist()

    for b in range(B):
        p_len = int(prompt_lens[b].item())
        a_len = int(ans_lens[b].item())
        if a_len <= 0:
            continue

        total = int(total_lens[b])
        seq_start = T_total - total
        ans_start = seq_start + p_len
        ans_end = ans_start + a_len

        # GT ids in unprefixed packed input_ids
        gt_ids = input_ids[b, ans_start:ans_end].tolist()
        gt_text = processor.tokenizer.decode(gt_ids)

        # In prefixed space, answer positions shift by prefix_len
        start_p = prefix_len + ans_start
        end_p = prefix_len + ans_end

        pred_ids = []
        gt_probs = []
        for pos in range(start_p, end_p):
            logit_pos = pos - 1  # causal shift
            pred_id = int(torch.argmax(logits[b, logit_pos]).item())
            pred_ids.append(pred_id)

            probs = torch.softmax(logits[b, logit_pos], dim=-1)
            gt_id = gt_ids[pos - start_p]
            gt_probs.append(float(probs[gt_id].item()))

        pred_text = processor.tokenizer.decode(pred_ids)
        mean_p = sum(gt_probs) / max(1, len(gt_probs))

        print(
            f"[DBG b={b}] ans_span(unpref)=[{ans_start},{ans_end}) "
            f"ans_span(pref)=[{start_p},{end_p}) | "
            f"GT='{gt_text}' ids={gt_ids} | "
            f"PRED='{pred_text}' ids={pred_ids} | "
            f"mean_P(GT)={mean_p:.4f}"
        )

        # after gt_probs is computed
        if len(gt_probs) == 2:
            print(f"    P(letter)={gt_probs[0]:.4f}  P(eos)={gt_probs[1]:.4f}")


def debug_print_loss_details_span_packed(processor, outputs, labels, prefix_len, prompt_lens, ans_lens, T_total):
    """
    Computes per-token NLL over the labeled answer span (same positions you label).
    """
    logits = outputs.logits
    B = labels.size(0)

    total_lens = (prompt_lens + ans_lens).tolist()

    for b in range(B):
        p_len = int(prompt_lens[b].item())
        a_len = int(ans_lens[b].item())
        if a_len <= 0:
            continue

        total = int(total_lens[b])
        seq_start = T_total - total
        ans_start = seq_start + p_len
        ans_end = ans_start + a_len

        start_p = prefix_len + ans_start
        end_p = prefix_len + ans_end

        nlls = []
        gt_ids = []
        for pos in range(start_p, end_p):
            gt_id = int(labels[b, pos].item())
            if gt_id == -100:
                continue
            logit_pos = pos - 1
            log_probs = torch.log_softmax(logits[b, logit_pos], dim=-1)
            nlls.append(float(-log_probs[gt_id].item()))
            gt_ids.append(gt_id)

        gt_text = processor.tokenizer.decode(gt_ids)
        mean_nll = (sum(nlls) / max(1, len(nlls)))

        print(f"[DBG loss b={b}] answer_GT='{gt_text}' | tokens={len(nlls)} | mean_NLL={mean_nll:.6f}")


def dump_args_to_logdir(args, log_dir: str, filename: str = "arg_info.txt"):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)

    # convert argparse.Namespace -> dict
    args_dict = vars(args)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Argument dump ===\n")
        f.write(f"time: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"log_dir: {log_dir}\n\n")

        # 1) human-readable (sorted)
        f.write("[args: pretty]\n")
        for k in sorted(args_dict.keys()):
            f.write(f"{k}: {args_dict[k]}\n")

        # 2) machine-readable JSON (sorted)
        f.write("\n\n[args: json]\n")
        json.dump(args_dict, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    print(f"[Log] Saved args to: {path}")