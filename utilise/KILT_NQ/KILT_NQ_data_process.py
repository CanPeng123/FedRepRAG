import json
from torch.utils.data import Dataset


def load_kilt_nq(jsonl_path):
    """
    Load KILT-NQ JSONL.  Returns a list of dicts with keys:
        id, question, gold_answers (list of non-empty answer strings)
    """
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            gold = [
                o['answer'].strip()
                for o in ex.get('output', [])
                if o.get('answer', '').strip()
            ]
            records.append({
                'id': str(ex['id']),
                'question': ex['input'].strip(),
                'gold_answers': gold,
            })
    return records


def shard_records(records, idx, num_parts):
    return records[idx::num_parts]


class KILTNQDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def collate_records(batch):
    return batch