import os
import re
import string
from typing import List, Dict, Any, Optional, Tuple
import torch
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import json
from torch.utils.data import DataLoader, Dataset


def load_slake_items(json_path: str) -> list[dict]:
    """Load BoKelvin/SLAKE style json: a list of dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"SLAKE json should be a list, got {type(data)}")
    return data


def build_tasks_from_slake(items: list[dict], images_root: Optional[str] = None, split_name: Optional[str] = None, lang_filter: Optional[str] = "en") -> list[dict]:
    """
    Build unified task dicts:
      - qid
      - split
      - question
      - image_path
      - gold_answer
      - answer_type / base_type / modality / location / content_type / triple
    """
    tasks = []
    for it in items:
        if lang_filter is not None and str(it.get("q_lang", "")).lower() != str(lang_filter).lower():
            continue

        img_name = it.get("img_name") or it.get("image") or it.get("image_path")
        if not img_name:
            # skip if no image
            continue

        image_path = os.path.join(images_root, img_name) if images_root else img_name
        qid = it.get("qid", None)
        # fallback qid if missing
        if qid is None:
            qid = f"{it.get('img_id','')}_{len(tasks)}"

        tasks.append({
            "qid": str(qid),
            "split": split_name or "",
            "question": str(it.get("question", "")),
            "image_path": image_path,
            "gold_answer": str(it.get("answer", "")),
            "answer_type": str(it.get("answer_type", "")),   # OPEN / CLOSED
            "base_type": str(it.get("base_type", "")),       # vqa / kvqa
            "content_type": str(it.get("content_type", "")),
            "modality": str(it.get("modality", "")),
            "location": str(it.get("location", "")),
            "triple": it.get("triple", None),                # often [vhead, rel, tail]
        })
    return tasks


def load_pil_or_none(path: str | None) -> PILImage | None:
    if not path:
        return None
    try:
        return PILImageModule.open(path).convert("RGB")
    except Exception:
        return None


def load_captions_map(path: str) -> Dict[str, str]:
    """
    Supports:
      - dict: { "qid": "caption", ... }
      - list: [ {"qid":..., "caption":...}, ... ]
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    if isinstance(data, list):
        m = {}
        for x in data:
            qid = str(x.get("qid"))
            cap = str(x.get("caption", ""))
            if qid:
                m[qid] = cap
        return m

    return {}


class TrainInjectionDataset(Dataset):
    """
    Keep only samples that have supervision.
    SLAKE:      gold_answer (string) or answer
    """
    def __init__(self, tasks: List[Dict[str, Any]], gold_keys=("gold", "gold_answer", "answer")):
        kept = []
        for t in tasks:
            ok = False
            for k in gold_keys:
                v = t.get(k, None)
                if v is None:
                    continue
                if isinstance(v, str) and v.strip() == "":
                    continue
                ok = True
                break
            if ok:
                kept.append(t)

        self.tasks = kept
        print(f"[Dataset] kept {len(self.tasks)}/{len(tasks)} tasks (keys={gold_keys})")

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.tasks[idx]


def collate_tasks(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch