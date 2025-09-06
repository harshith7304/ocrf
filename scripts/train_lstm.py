import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from src.utils.text_cleaning import normalize_text
from src.models.lstm_model import LSTMClassifier
from sklearn.metrics import classification_report, f1_score

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"


class Vocab:
    def __init__(self, min_freq=2, max_size=50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def build(self, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        most = [w for w, f in cnt.most_common(self.max_size) if f >= self.min_freq]
        for w in most:
            if w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, text):
        return [self.stoi.get(w, 1) for w in text.split()]


class SeqDataset(Dataset):
    def __init__(self, df, vocab: Vocab, max_len=400):
        self.texts = df.text.tolist()
        self.labels = df.label.astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx])[: self.max_len]
        return ids, self.labels[idx]


def collate(batch):
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
    max_len = lengths.max().item() if len(lengths) > 0 else 1
    padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded, lengths, labels


def load(split: str) -> pd.DataFrame:
    p_parquet = PROC_DIR / f"{split}.parquet"
    p_csv = PROC_DIR / f"{split}.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    elif p_csv.exists():
        return pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"Missing split files: {p_parquet} or {p_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=50000)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--limit_train", type=int, default=0, help="Subsample N training rows for a fast run (0=full)")
    parser.add_argument("--early_stop_patience", type=int, default=2, help="Stop after N epochs without improvement")
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--target_metric", type=str, default="loss", choices=["loss", "f1"], help="Metric to monitor for early stopping")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_df = load("train")
    val_df = load("val")
    test_df = load("test")

    if args.limit_train and args.limit_train < len(train_df):
        train_df = train_df.sample(args.limit_train, random_state=args.seed).reset_index(drop=True)
        print(f"[fast] Using {len(train_df)} training rows (subset)")

    vocab = Vocab(min_freq=args.min_freq, max_size=args.max_size)
    vocab.build(train_df.text)

    train_ds = SeqDataset(train_df, vocab, max_len=args.max_len)
    val_ds = SeqDataset(val_df, vocab, max_len=args.max_len)
    test_ds = SeqDataset(test_df, vocab, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        vocab_size=len(vocab.itos),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_score = None  # best (lower for loss, higher for f1)
    bad_epochs = 0
    patience = args.early_stop_patience
    best_path = BASE / "models" / "lstm_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    def run_epoch(dl, train=True):
        model.train(train)
        total_loss = 0.0
        for x, lengths, labels in dl:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(dl.dataset)

    def eval_f1(dl):
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for x, lengths, labels in dl:
                x, lengths = x.to(device), lengths.to(device)
                logits = model(x, lengths)
                probs = torch.sigmoid(logits)
                pred = (probs.cpu().numpy() >= 0.5).astype(int)
                yh.extend(pred.tolist())
                ys.extend(labels.numpy().astype(int).tolist())
        return f1_score(ys, yh, average="macro")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        if args.target_metric == "f1":
            metric_val = eval_f1(val_loader)
        else:
            metric_val = val_loss
        improved = False
        if best_score is None:
            improved = True
        else:
            if args.target_metric == "loss":
                improved = metric_val < best_score
            else:
                improved = metric_val > best_score
        if improved:
            best_score = metric_val
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "vocab": vocab.itos}, best_path)
            flag = "*"
        else:
            bad_epochs += 1
            flag = ""
        dt = time.time() - t0
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} metric({args.target_metric})={metric_val:.4f} bad_epochs={bad_epochs} time={dt:.1f}s {flag}"
        )
        if epoch >= args.min_epochs and bad_epochs >= patience:
            print("Early stopping.")
            break

    # Test evaluation
    if best_path.exists():
        model.load_state_dict(torch.load(best_path)["model"])
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, lengths, labels in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            probs = torch.sigmoid(logits)
            yh.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
            ys.extend(labels.numpy().astype(int).tolist())
    print(classification_report(ys, yh, digits=4))


if __name__ == "__main__":
    main()
