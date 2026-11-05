import os
import glob
import pandas as pd
from typing import Tuple, Optional, Set


def infer_event_from_path(p: str) -> str:
    name = os.path.basename(p)
    # e.g., "mexico_earthquake_final_data.tsv" -> "mexico_earthquake"
    if name.endswith("_final_data.tsv"):
        return name.replace("_final_data.tsv", "")
    return os.path.splitext(name)[0]


def read_one_tsv(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    df["__event"] = infer_event_from_path(path)
    return df


def load_annotations_multi(annotations_dir: str, tsv_list: str, sep: str) -> pd.DataFrame:
    """Load 7 TSVs either from a directory (glob *.tsv) or a comma-separated list of paths."""
    paths = []
    if tsv_list:
        paths = [p.strip() for p in tsv_list.split(",") if p.strip()]
    elif annotations_dir:
        paths = sorted(glob.glob(os.path.join(annotations_dir, "*.tsv")))
    else:
        raise ValueError("Provide either --annotations_dir or --tsv_list.")

    if not paths:
        raise ValueError("No TSV files found.")

    frames = [read_one_tsv(p, sep) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def _read_id_set(tsv_path: str, id_col: str = "tweet_id", sep: str = "\t") -> Set[str]:
    """Read a TSV and return the set of tweet_id strings."""
    if not tsv_path or not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Split file not found: {tsv_path}")
    df_ids = pd.read_csv(tsv_path, sep=sep, dtype={id_col: str})
    if id_col not in df_ids.columns:
        raise KeyError(f"Column '{id_col}' not found in {tsv_path}. Columns: {list(df_ids.columns)}")
    return set(df_ids[id_col].astype(str).tolist())


def split_df_by_tweet_ids(df: pd.DataFrame, train_split_tsv: str, dev_split_tsv: str, test_split_tsv: str, *, 
    id_col: str = "tweet_id", sep: str = "\t", strict: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a full annotations DataFrame into train/dev/test using tweet_id membership
    from three split TSVs (one per split). Doesn't modify your existing loaders.

    - If the same tweet_id appears in multiple split files, precedence is:
      train > dev > test (we de-overlap in that order).
    - If 'strict' is True, raises if any split id is not present in df[id_col].
    """
    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found in annotations df. Columns: {list(df.columns)}")

    # Normalize df id column to str for reliable matching
    df = df.copy()
    df[id_col] = df[id_col].astype(str)

    ids_all = set(df[id_col].tolist())
    ids_train = _read_id_set(train_split_tsv, id_col=id_col, sep=sep)
    ids_dev   = _read_id_set(dev_split_tsv,   id_col=id_col, sep=sep)
    ids_test  = _read_id_set(test_split_tsv,  id_col=id_col, sep=sep)

    # De-overlap with precedence: train > dev > test
    ids_dev  = ids_dev  - ids_train
    ids_test = ids_test - ids_train - ids_dev

    # Sanity checks / diagnostics
    unknown = (ids_train | ids_dev | ids_test) - ids_all
    if unknown:
        msg = f"[split] {len(unknown)} tweet_id(s) in split files not found in annotations df (showing up to 10): {list(sorted(unknown))[:10]}"
        if strict:
            raise ValueError(msg)
        else:
            print("WARN:", msg)

    # Build splits
    df_train = df[df[id_col].isin(ids_train)].copy()
    df_dev   = df[df[id_col].isin(ids_dev)].copy()
    df_test  = df[df[id_col].isin(ids_test)].copy()

    # Report coverage
    covered = set(df_train[id_col]) | set(df_dev[id_col]) | set(df_test[id_col])
    uncovered = ids_all - covered
    print(f"[split] train/dev/test sizes: {len(df_train)}/{len(df_dev)}/{len(df_test)} (rows)")
    print(f"[split] unique tweet_ids covered: {len(covered)} / {len(ids_all)} in annotations")
    if uncovered:
        print(f"[split] {len(uncovered)} tweet_id(s) in annotations not present in any split file.")

    return df_train, df_dev, df_test


def auto_split_from_dir(df: pd.DataFrame, split_dir: str, task_prefix: Optional[str] = None, *, id_col: str = "tweet_id",
    sep: str = "\t", strict: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience: find split files in a directory and call split_df_by_tweet_ids.

    If task_prefix is given, expect files like:
      {task_prefix}_train.tsv, {task_prefix}_dev.tsv, {task_prefix}_test.tsv
    Otherwise try to locate files containing: 'train', 'dev', 'test' respectively.
    """
    def _find(patterns):
        for fname in os.listdir(split_dir):
            low = fname.lower()
            if all(p in low for p in patterns):
                return os.path.join(split_dir, fname)
        return None

    if task_prefix:
        train_path = os.path.join(split_dir, f"{task_prefix}_train.tsv")
        dev_path   = os.path.join(split_dir, f"{task_prefix}_dev.tsv")
        test_path  = os.path.join(split_dir, f"{task_prefix}_test.tsv")
    else:
        train_path = _find(["train", ".tsv"])
        dev_path   = _find(["dev", ".tsv"])
        test_path  = _find(["test", ".tsv"])

    if not train_path or not dev_path or not test_path:
        raise FileNotFoundError(
            f"Could not find split files in {split_dir}. "
            f"Found train={train_path}, dev={dev_path}, test={test_path}"
        )

    return split_df_by_tweet_ids(df, train_split_tsv=train_path, dev_split_tsv=dev_path, test_split_tsv=test_path, 
    id_col=id_col, sep=sep, strict=strict)
