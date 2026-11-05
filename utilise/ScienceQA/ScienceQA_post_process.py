import re
import os
import pandas as pd
import json
from typing import List, Dict, Tuple, Any


# ---- Parse model output to a choice index ----
def parse_option_letter(output: str, num_choices: int) -> int:
    """
    Parse the model output and return a 0-based option index.
    Accepts forms like 'A', 'Answer: B', '(c)', 'D)', 'option e', etc.
    Returns -1 if no valid option letter is found within [A .. A+num_choices-1].
    """
    if output is None:
        return -1

    text = str(output).strip()
    if not text:
        return -1

    # 1) Try to find a standalone capital letter first (best signal)
    m = re.search(r"\b([A-Z])\b", text)
    if m:
        letter = m.group(1)
    else:
        # 2) Fallback: grab the first alphabetic character anywhere, then uppercase it
        m2 = re.search(r"[A-Za-z]", text)
        if not m2:
            return -1
        letter = m2.group(0).upper()

    idx = ord(letter) - ord("A")
    return idx if 0 <= idx < int(num_choices) else -1


def parse_trace_json_loose(s: str) -> Dict[str, Any]:
    """Recover a single JSON object from a raw string (robust to extra text)."""
    s = (s or "").strip()
    # fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # find first {...}
    try:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j != -1 and j > i:
            frag = s[i:j+1]
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass
    return {}


# -----------------------------
# Result Store
# -----------------------------
def save_split_reports_with_tag(df: pd.DataFrame, out_dir: str, mode: str, prompt_type: str):
    # normalize
    def _norm_subject(s: str) -> str:
        s = (s or "").strip().lower()
        if "natural" in s: return "natural_science"
        if "social"  in s: return "social_science"
        if "language" in s or "lingu" in s or s == "lan": return "language"
        return s or "unknown"

    def _band(g):
        try:
            g = int(g)
        except:
            return "Unknown"
        if 1 <= g <= 6:  return "Grades_1_6"
        if 7 <= g <= 12: return "Grades_7_12"
        return "Unknown"

    def _acc(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.mean()) if not x.empty else None

    df = df.copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df["subject_norm"] = df["subject"].apply(_norm_subject)
    df["grade_band"] = df["grade"].apply(_band)

    # -------- Contextual modalities (mutually exclusive) --------
    # NOTE: If you have the raw dataset hint flag, prefer:
    #   df["has_txt"] = df["has_hint_dataset"].astype(bool)
    df["has_txt"] = df["context"].fillna("").astype(str).str.strip().ne("")
    df["has_img"] = df["image_path"].fillna("").astype(str).str.strip().ne("")
    def _ctx_mod(row):
        if row["has_txt"] and row["has_img"]: return "TXT+IMG"
        if row["has_txt"]:  return "TXT"
        if row["has_img"]:  return "IMG"
        return "NO"
    df["ctx_modality"] = df.apply(_ctx_mod, axis=1)

    wanted = ["natural_science", "social_science", "language"]

    for split_name, g in df.groupby("split"):
        lines = []
        lines.append(f"Split: {split_name}  |  Mode: {mode} | Prompt type: {prompt_type}")
        lines.append(f"Total items: {int(g.shape[0])}")

        overall = _acc(g["correct"])
        lines.append(f"Overall accuracy: {overall:.4f}" if overall is not None else "Overall accuracy: N/A")
        lines.append("")

        # ----- By subject -----
        lines.append("By subject:")
        subject_accs = []
        for s in wanted:
            sub = g[g["subject_norm"] == s]
            acc = _acc(sub["correct"]); n = int(sub.shape[0])
            lines.append(f"  {s:17s}  acc={acc:.4f}  (n={n})" if acc is not None else f"  {s:17s}  acc=N/A      (n={n})")
            if acc is not None: subject_accs.append(acc)
        lines.append(f"Macro average (subjects): {sum(subject_accs)/len(subject_accs):.4f}" if subject_accs else "Macro average (subjects): N/A")
        lines.append("")

        # ----- By grade band (overall) -----
        lines.append("By grade band (overall):")
        for band in ["Grades_1_6", "Grades_7_12"]:
            sub = g[g["grade_band"] == band]
            acc = _acc(sub["correct"]); n = int(sub.shape[0])
            lines.append(f"  {band:14s}  acc={acc:.4f}  (n={n})" if acc is not None else f"  {band:14s}  acc=N/A      (n={n})")
        lines.append("")

        # ----- By contextual modality (mutually exclusive) -----
        lines.append("By contextual modality (exclusive buckets):")
        ctx_order = ["TXT", "IMG", "TXT+IMG", "NO"]
        ctx_accs = []
        for cm in ctx_order:
            sub = g[g["ctx_modality"] == cm]
            acc = _acc(sub["correct"]); n = int(sub.shape[0])
            lines.append(f"  {cm:7s}  acc={acc:.4f}  (n={n})" if acc is not None else f"  {cm:7s}  acc=N/A      (n={n})")
            if acc is not None and n > 0 and cm != "NO":
                ctx_accs.append(acc)
        # Macro average across contextual modalities (excluding NO)
        lines.append(f"Macro average (modalities, TXT/IMG/TXT+IMG): {sum(ctx_accs)/len(ctx_accs):.4f}" if ctx_accs else "Macro average (modalities): N/A")

        out_path = os.path.join(out_dir, f"metrics_{split_name}_mode-{mode}_prompt-{prompt_type}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote split report: {out_path}")


def write_per_split_outputs(df: pd.DataFrame, out_dir: str, split: str, mode: str, prompt_type: str=""):
    # TSV
    out_path = os.path.join(out_dir, f"scienceqa_{split}_mode-{mode}_prompt-{prompt_type}.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved predictions to: {out_path}")

    # model outputs
    for split_name, split_df in df.groupby("split"):
        txt_path = os.path.join(out_dir, f"model_outputs_{split_name}_mode-{mode}_prompt-{prompt_type}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                f.write(f"{row['qid']}: {row['raw_output'].strip()}\n")
        print(f"Saved model outputs for split={split_name} to {txt_path}")

    save_split_reports_with_tag(df, out_dir, mode=mode, prompt_type=prompt_type)


