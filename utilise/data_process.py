
import torch
from PIL import Image
import PIL.Image as PILImageModule
from PIL.Image import Image as PILImage
import os
import json
import pathlib
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# --------------------------
# ----- Dataset Helper -----
# --------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def load_pil_or_none(path: str | None) -> PILImage | None:
    if not path:
        return None
    try:
        return PILImageModule.open(path).convert("RGB")
    except Exception:
        return None


def collate_tasks(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


# ---------------------------
# ----- Dataset Process -----
# ---------------------------
class TrainInjectionDataset(Dataset):
    def __init__(self, tasks: List[Dict[str, Any]]):
        self.tasks = [t for t in tasks if t.get("gold") is not None]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.tasks[idx]


def get_client_json_files(client_train_dir: str):
    client_train_dir = pathlib.Path(client_train_dir)
    client_files = sorted(client_train_dir.glob("client_*.json"))
    if len(client_files) == 0:
        raise ValueError(f"No client_*.json found in {client_train_dir}")
    return client_files


