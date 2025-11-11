
"""
Preprocessing for IMDb dataset (CSV version).
- Reads `data/IMDB Dataset.csv` containing columns: 'review', 'sentiment'
- Lowercase, remove punctuation/special characters
- Tokenize (nltk.word_tokenize)
- Keep top 10,000 tokens to build vocab
- Convert to ids; OOV -> 1 (reserve 0 for PAD)
- Split into train/test (25k each, random shuffle)
- Provide DataLoaders for seq_len in {25, 50, 100}
"""

import os, re, argparse, json, random
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils import ListDataset, PadTruncateCollate

CLEAN_RE = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    text = text.lower()
    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(tokenized_texts, vocab_size=10000, specials=("<pad>", "<unk>")):
    counter = Counter()
    for toks in tokenized_texts:
        counter.update(toks)
    most_common = [w for w, _ in counter.most_common(vocab_size - len(specials))]
    stoi = {specials[0]: 0, specials[1]: 1}
    for i, w in enumerate(most_common, start=2):
        stoi[w] = i
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def tokens_to_ids(tokens, stoi):
    unk_id = stoi.get("<unk>", 1)
    return torch.tensor([stoi.get(t, unk_id) for t in tokens], dtype=torch.long)


def load_csv_dataset(path_csv='data/IMDB Dataset.csv', seed=42):
    df = pd.read_csv(path_csv)
    assert {'review', 'sentiment'}.issubset(df.columns), "CSV must have 'review' and 'sentiment' columns"
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    texts = df['review'].tolist()
    labels = [1 if s.strip().lower() == 'positive' else 0 for s in df['sentiment']]
    train_texts, test_texts = texts[:25000], texts[25000:]
    train_labels, test_labels = labels[:25000], labels[25000:]
    return train_texts, train_labels, test_texts, test_labels


def preprocess_and_cache(seq_len=50, vocab_size=10000, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'IMDB Dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected file not found: {csv_path}")

    train_texts, train_labels, test_texts, test_labels = load_csv_dataset(csv_path)

    train_tokens = [word_tokenize(clean_text(t)) for t in train_texts]
    test_tokens  = [word_tokenize(clean_text(t)) for t in test_texts]

    stoi, itos = build_vocab(train_tokens, vocab_size=vocab_size)

    train_ids = [tokens_to_ids(toks, stoi) for toks in train_tokens]
    test_ids  = [tokens_to_ids(toks, stoi) for toks in test_tokens]

    # Save vocab
    with open(os.path.join(out_dir, 'vocab.json'), 'w') as f:
        json.dump(stoi, f)

    # Save tensors to pt files
    torch.save({'ids': train_ids, 'labels': train_labels}, os.path.join(out_dir, 'train.pt'))
    torch.save({'ids': test_ids,  'labels': test_labels},  os.path.join(out_dir, 'test.pt'))

    return make_dataloaders(seq_len=seq_len, data_dir=out_dir)


def make_dataloaders(seq_len=50, data_dir='data', batch_size=32):
    train = torch.load(os.path.join(data_dir, 'train.pt'))
    test  = torch.load(os.path.join(data_dir, 'test.pt'))
    train_ds = ListDataset(train['ids'], train['labels'])
    test_ds  = ListDataset(test['ids'],  test['labels'])
    collate = PadTruncateCollate(max_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=50, choices=[25,50,100])
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--out_dir', type=str, default='data')
    args = parser.parse_args()

    train_loader, test_loader = preprocess_and_cache(seq_len=args.seq_len, vocab_size=args.vocab_size, out_dir=args.out_dir)
    print(f"Preprocessing done. Batches in train: {len(train_loader)} | test: {len(test_loader)}")









