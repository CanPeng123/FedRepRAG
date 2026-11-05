import json
import os
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset

# Supported dataset keys in benchmark.json
MEDICAL_QA_DATASETS = ['medqa', 'medmcqa', 'pubmedqa', 'bioasq', 'mmlu']


def load_benchmark(benchmark_json: str, dataset_filter: str = 'all') -> List[Dict[str, Any]]:
    """
    Load benchmark.json and flatten into a list of task dicts.

    benchmark.json structure:
        { "medqa": {"id": {question, options, answer}, ...}, "medmcqa": {...}, ... }

    Returns list of task dicts with unified fields.
    """
    with open(benchmark_json, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)

    tasks = []
    datasets = MEDICAL_QA_DATASETS if dataset_filter == 'all' else [dataset_filter]

    for ds_name in datasets:
        if ds_name not in benchmark:
            continue
        for qid, item in benchmark[ds_name].items():
            tasks.append({
                'qid': f'{ds_name}-{qid}',
                'dataset': ds_name,
                'split': 'test',
                'question': item['question'],
                'options': item['options'],           # dict: {"A": ..., "B": ..., ...}
                'gold_answer': item['answer'],         # single letter: "A"/"B"/"C"/"D"
                'pmid': item.get('PMID', []),
            })

    return tasks


def load_train(train_json: str) -> List[Dict[str, Any]]:
    """
    Load PubMedQA train.json (MedQA training split, used as retrieval corpus).

    train.json structure: list of dicts with keys:
        qid, question, options, answer, dataset, split, meta_info
    """
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = []
    for item in data:
        tasks.append({
            'qid': item['qid'],
            'dataset': item.get('dataset', 'medqa'),
            'split': item.get('split', 'train'),
            'question': item['question'],
            'options': item['options'],
            'gold_answer': item['answer'],
            'meta_info': item.get('meta_info', ''),
        })

    return tasks


def load_pubmedqa_train(json_path: str) -> List[Dict[str, Any]]:
    """
    Load PubMedQA labeled train set.

    Handles two formats automatically:
      - Original format (labeled_train_set.json): keys QUESTION, CONTEXTS, LONG_ANSWER, …
      - FL pool format (client_*.json):           keys question, title, content, qid, pmid, …
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Detect format from the first entry
    first = next(iter(data.values()))
    is_pool_format = 'question' in first and 'QUESTION' not in first

    tasks = []
    for pubid, item in data.items():
        if is_pool_format:
            tasks.append({
                'qid':        item.get('qid', pubid),
                'pmid':       item.get('pmid', ''),
                'dataset':    'pubmedqa',
                'split':      'train',
                'question':   item.get('question', ''),
                'title':      item.get('title', ''),
                'content':    item.get('content', ''),
                'gold_answer': item.get('final_decision', ''),
                'source':     item.get('source', ''),
            })
        else:
            tasks.append({
                'qid':        pubid,
                'dataset':    'pubmedqa',
                'split':      'train',
                'question':   item['QUESTION'],
                'contexts':   item.get('CONTEXTS', []),
                'long_answer': item.get('LONG_ANSWER', ''),
                'gold_answer': item.get('final_decision', ''),
                'meshes':     item.get('MESHES', []),
            })
    return tasks


def load_pubmedqa_fl_train(json_path: str) -> List[Dict[str, Any]]:
    """
    Load a PubMedQA FL client train JSON (pool format).

    Format: dict keyed by an arbitrary id, each value has:
        qid, pmid, question, title, content, final_decision, source
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = []
    for _, item in data.items():
        tasks.append({
            'qid':            item.get('qid', ''),
            'pmid':           item.get('pmid', ''),
            'dataset':        'pubmedqa',
            'split':          'train',
            'question':       item.get('question', ''),
            'title':          item.get('title', ''),
            'content':        item.get('content', ''),
            'gold_answer':    item.get('final_decision', ''),
            'source':         item.get('source', ''),
        })
    return tasks


def load_pubmed_corpus(
    chunk_dir: str,
    idx: int = 0,
    num_parts: int = 1,
    pubmedqa_train_json: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load PubMed retrieval corpus from JSONL chunk files.

    Each JSONL entry has: id, title, content, contents, PMID.
    Shards by file index: only loads files where file_i % num_parts == idx.

    If pubmedqa_train_json is provided, each corpus entry is enriched with the
    PubMedQA question for its PMID (both train and test sets are looked up, so
    all 1000 articles are indexed). The 'question' field is then used for
    embedding instead of the raw passage text, keeping query and index in the
    same semantic space.
    """
    # Build PMID -> question map from train + test sets
    pmid_to_question: dict = {}
    if pubmedqa_train_json:
        train_json_path = pubmedqa_train_json
        test_json_path = os.path.join(os.path.dirname(train_json_path), 'test_set.json')
        with open(train_json_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        pmid_to_question = {pmid: v['QUESTION'] for pmid, v in train_data.items()}
        if os.path.exists(test_json_path):
            with open(test_json_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            pmid_to_question.update({pmid: v['QUESTION'] for pmid, v in test_data.items()})

    chunk_files = sorted(
        f for f in os.listdir(chunk_dir) if f.endswith('.jsonl')
    )
    assigned = [f for i, f in enumerate(chunk_files) if i % num_parts == idx]

    tasks = []
    for fname in assigned:
        fpath = os.path.join(chunk_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                pmid = str(item.get('PMID', ''))
                tasks.append({
                    'qid': item.get('id', ''),
                    'dataset': 'pubmed_corpus',
                    'split': 'corpus',
                    'title': item.get('title', ''),
                    'content': item.get('content', ''),
                    'contents': item.get('contents', ''),
                    'pmid': pmid,
                    # question populated only when pubmedqa_train_json is given
                    # prefer lookup dict; fall back to question baked into JSONL
                    'question': pmid_to_question.get(pmid, item.get('question', '')),
                })
    return tasks


def load_combined_pool(
    jsonl_path: str,
    idx: int = 0,
    num_parts: int = 1,
) -> List[Dict[str, Any]]:
    """
    Load the combined pool JSONL (pqa_labeled train + pqa_unlabeled).

    Each entry has: qid, pmid, question, long_answer, abstract,
                    final_decision (empty string for unlabeled), source.
    """
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i % num_parts != idx:
                continue
            item = json.loads(line)
            tasks.append({
                "qid":            item["qid"],
                "pmid":           item.get("pmid", ""),
                "dataset":        "pubmedqa_combined",
                "split":          "pool",
                "question":       item.get("question", ""),
                "title":          item.get("title", ""),
                "content":        item.get("content", ""),
                "final_decision": item.get("final_decision", ""),
                "source":         item.get("source", ""),
            })
    return tasks


def build_train_task_map(train_tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return {qid: task} lookup dict for O(1) neighbour resolution."""
    return {t['qid']: t for t in train_tasks}


def shard_tasks(tasks: List[Dict[str, Any]], idx: int, num_parts: int) -> List[Dict[str, Any]]:
    """Split tasks into num_parts shards and return the idx-th shard."""
    return [t for i, t in enumerate(tasks) if i % num_parts == idx]


class PubMedQADataset(Dataset):
    def __init__(self, tasks: List[Dict[str, Any]]):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_tasks(batch):
    return batch