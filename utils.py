import os, random, time, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int = 42):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    return elapsed


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


class PadTruncateCollate:
    def __init__(self, max_len: int):
        self.max_len = max_len

    def __call__(self, batch):
        # batch: list of (tensor_ids, label)
        xs, ys = zip(*batch)
        max_len = self.max_len
        out = []
        for x in xs:
            if len(x) >= max_len:
                out.append(x[:max_len])
            else:
                pad = torch.zeros(max_len - len(x), dtype=torch.long)
                out.append(torch.cat([x, pad], dim=0))
        X = torch.stack(out, dim=0)
        y = torch.tensor(ys, dtype=torch.float32)
        return X, y


class ListDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def save_row(path_csv, row_dict, header_order=None):
    import csv
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    write_header = not os.path.exists(path_csv)
    if header_order is None:
        header_order = list(row_dict.keys())
    with open(path_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
