import argparse
from pathlib import Path
import sys, time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def load_split(name: str):
    p = PROC / f"{name}.parquet"
    return pd.read_parquet(p)


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok, max_len=256):
        self.texts = df.text.tolist()
        self.labels = df.label.astype(int).tolist()
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=0, help="Subsample N training rows (0=full)")
    ap.add_argument("--target_metric", choices=["f1", "loss"], default="f1")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    if args.limit_train and args.limit_train < len(train_df):
        train_df = train_df.sample(args.limit_train, random_state=args.seed).reset_index(drop=True)
        print(f"[fast] Using train subset: {len(train_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, use_auth_token=False)

    train_ds = TextDataset(train_df, tokenizer, args.max_len)
    val_ds = TextDataset(val_df, tokenizer, args.max_len)
    test_ds = TextDataset(test_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Class weights
    counts = train_df.label.value_counts()
    w0 = 1.0
    w1 = (counts[0] / counts[1]) if 1 in counts else 1.0
    class_weights = torch.tensor([w0, w1], dtype=torch.float, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_metric = None
    bad_epochs = 0
    best_dir = MODELS / "bert_best"
    best_dir.mkdir(parents=True, exist_ok=True)

    def run_epoch(dl, train=True):
        model.train(train)
        total = 0.0
        for batch in dl:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            with torch.set_grad_enabled(train):
                out = model(**batch)
                loss = criterion(out.logits, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            total += loss.item() * labels.size(0)
        return total / len(dl.dataset)

    def macro_f1(dl):
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for batch in dl:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                pred = logits.argmax(-1).cpu().tolist()
                ys.extend(labels.tolist())
                yh.extend(pred)
        return f1_score(ys, yh, average="macro")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        metric_val = macro_f1(val_loader) if args.target_metric == "f1" else val_loss
        improved = False
        if best_metric is None:
            improved = True
        else:
            if args.target_metric == "loss":
                improved = metric_val < best_metric
            else:
                improved = metric_val > best_metric
        if improved:
            best_metric = metric_val
            bad_epochs = 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            flag = "*"
        else:
            bad_epochs += 1
            flag = ""
        print(
            f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} metric({args.target_metric})={metric_val:.4f} bad_epochs={bad_epochs} time={time.time()-t0:.1f}s {flag}"
        )
        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

    # Test evaluation
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred = logits.argmax(-1).cpu().tolist()
            ys.extend(labels.tolist())
            yh.extend(pred)
    print(classification_report(ys, yh, digits=4))


if __name__ == "__main__":
    main()
