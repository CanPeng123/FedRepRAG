import torch
from PIL import Image
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import os
import json
import pandas as pd
from typing import List, Dict, Any, Union, Optional


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_scienceqa_problems(problems_json_path: str) -> Dict[str, Any]:
    with open(problems_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_tasks_from_scienceqa(problems: Dict[str, Any], images_root: str | None = None, split_filter: str | None = None) -> list[Dict[str, Any]]:
    tasks = []
    for qid, ex in problems.items():
        if split_filter and ex.get("split") != split_filter:
            continue
        img_path = ex.get("image", "")  # if the "image" key is missing from the ex dictionary return ""
        if images_root and img_path and not os.path.isabs(img_path):
            img_path = os.path.join(images_root, ex.get("split") or split_filter or "", str(qid), img_path)
        if img_path and not os.path.exists(img_path):
            img_path = ""  # handle text-only items

        print("build_tasks_from_scienceqa | images_root: {0}".format(images_root))
        print("build_tasks_from_scienceqa | img_path: {0}".format(img_path))

        # tolerant subject & grade extraction
        subject = (ex.get("subject")).strip()
        grade = ex.get("grade")
        try:
            grade = int(grade[5:]) if grade is not None else None
        except Exception:
            grade = None

        tasks.append(dict(
            qid=qid,
            question=ex.get("question", ""),
            choices=ex.get("choices", []),
            image_path=img_path,
            context=ex.get("hint", None),
            gold=ex.get("answer", None),
            split=ex.get("split", None),
            subject=subject,
            grade=grade,
            lecture=ex.get("lecture", ""),
            solution=ex.get("solution", ""),
            topic=ex.get("topic", ""),
            skill=ex.get("skill", ""),
        ))
    return tasks


def discover_q_choice_by_convention(image_dir: str | None, num_choices: int) -> tuple[str | None, list[str | None]]:
    """
    Deterministically find:
      - question image: 'image' + ext
      - choice images:  'choice_0'..'choice_{n-1}' + ext
    Returns (q_path_or_None, [choice0_path_or_None, ...]).
    """
    if not image_dir or not os.path.isdir(image_dir):
        return None, [None] * num_choices

    # question image
    q_path = None
    for ext in IMG_EXTS:
        p = os.path.join(image_dir, f"image{ext}")
        if os.path.isfile(p):
            q_path = p
            break

    # choices
    choice_paths: list[str | None] = []
    for i in range(num_choices):
        found = None
        for ext in IMG_EXTS:
            p = os.path.join(image_dir, f"choice_{i}{ext}")
            if os.path.isfile(p):
                found = p
                break
        choice_paths.append(found)

    return q_path, choice_paths


def load_captions(dir_path: str, image_only: bool = True) -> Dict[str, str]:
    """
    Load caption text files from a directory.

    - If image_only=True, read only 'image.txt' and return {'image': <content>}.
    - Otherwise, read all *.txt files and return dict {stem: <content>}.

    Each file is expected to start with:
        [Description]\n
    and we return only the content AFTER that line.
    """
    if not dir_path or not os.path.isdir(dir_path):
        return {}

    def read_caption_after_header(path: str) -> str:
        """Read file and return content after the '[Description]\\n' header."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            header = "[Description]\n"
            if text.startswith(header):
                return text[len(header):].strip()
            else:
                # If header missing, return whole cleaned file
                return text.strip()
        except Exception:
            return ""

    captions: Dict[str, str] = {}

    # --- Case 1: Only image.txt requested ---
    if image_only:
        image_txt = os.path.join(dir_path, "image.txt")
        if os.path.isfile(image_txt):
            captions["image"] = read_caption_after_header(image_txt)
        return captions  # dict, possibly empty

    # --- Case 2: Load all txt files ---
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".txt"):
            continue

        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue

        stem, _ = os.path.splitext(fname)  # filename without extension
        captions[stem] = read_caption_after_header(fpath)

    return captions
