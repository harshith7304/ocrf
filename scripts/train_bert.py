import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df.text.tolist()
        self.labels = df.label.astype(int).tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


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
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_df = load("train")
    val_df = load("val")
    test_df = load("test")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    train_ds = TextDataset(train_df, tokenizer, args.max_len)
    val_ds = TextDataset(val_df, tokenizer, args.max_len)
    test_ds = TextDataset(test_df, tokenizer, args.max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(0, int(0.1 * num_training_steps)), num_training_steps=num_training_steps)

    best_val = float("inf")
    best_path = BASE / "models" / "bert_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    def run_epoch(dl, train=True):
        model.train(train)
        total_loss = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.set_grad_enabled(train):
                out = model(**batch)
                loss = out.loss
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            total_loss += loss.item() * batch["input_ids"].size(0)
        return total_loss / len(dl.dataset)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best to {best_path}")

    # Test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(dim=-1).cpu().numpy().tolist()
            yh.extend(preds)
            ys.extend(labels.numpy().astype(int).tolist())

    print(classification_report(ys, yh, digits=4))


if __name__ == "__main__":
    main()
