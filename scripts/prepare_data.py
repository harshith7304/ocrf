import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.text_cleaning import normalize_text, concat_fields

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


KAGGLE_CANDIDATES = [
    "fake_job_postings.csv",
    "Fake.csv",
]
EMSCAD_CANDIDATES = [
    "EMSCAD.csv",
    "emscad.csv",
]


def find_dataset_file():
    # prefer Kaggle, else EMSCAD
    for name in KAGGLE_CANDIDATES + EMSCAD_CANDIDATES:
        p = RAW_DIR / name
        if p.exists():
            return p
    # wildcard scan
    for p in RAW_DIR.glob("*.csv"):
        return p
    return None


def load_and_unify(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # Identify label
    label_col = None
    for k in ["fraudulent", "label", "is_fraud", "target", "fraud"]:
        if k in cols:
            label_col = cols[k]
            break
    if label_col is None:
        raise ValueError("Could not find label column (expected one of: fraudulent,label,is_fraud,target,fraud)")

    # Identify text fields
    text_fields_priority = [
        ["title", "description", "requirements", "benefits", "company_profile"],
        ["title", "description", "requirements"],
        ["description"],
    ]
    present = set(c.lower() for c in df.columns)
    for group in text_fields_priority:
        if all(g in present for g in group):
            fields = [cols[g] for g in group]
            break
    else:
        # fallback to all string cols except label
        fields = [c for c in df.columns if df[c].dtype == object and c != label_col]

    # Create unified text
    df["text_raw"] = df.apply(lambda r: concat_fields(r, fields), axis=1)
    df["text"] = df["text_raw"].map(normalize_text)

    # Map label to int {0,1}
    y = df[label_col]
    # Try robust conversions
    if y.dtype == bool:
        y = y.astype(int)
    elif y.dtype.kind in {"i", "u", "f"}:
        # already numeric; clamp to {0,1}
        y = (y.astype(float) > 0.5).astype(int)
    else:
        y_str = y.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "true": 1,
            "t": 1,
            "yes": 1,
            "y": 1,
            "fake": 1,
            "fraud": 1,
            "fraudulent": 1,
            "0": 0,
            "false": 0,
            "f": 0,
            "no": 0,
            "n": 0,
            "real": 0,
            "legit": 0,
            "legitimate": 0,
            "non-fraudulent": 0,
        }
        y = y_str.map(mapping)
    # Fallbacks for any remaining NaNs
    if y.isna().any():
        # Try majority class; if still empty, set 0
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0).astype(int)

    # Light sanity print
    print(f"Detected label column: {label_col}. Value counts (head):\n{y.value_counts(dropna=False).head().to_dict()}")

    out = df[["text", "text_raw"]].copy()
    out["label"] = y
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = find_dataset_file()
    if not path:
        raise SystemExit(f"No CSV found under {RAW_DIR}. Please run download script or place CSV manually.")
    print(f"Using dataset file: {path}")

    df = load_and_unify(path)

    # Stratified split: train/val/test
    train_df, temp_df = train_test_split(df, test_size=args.val_size + args.test_size, stratify=df["label"], random_state=args.seed)
    rel_val = args.val_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(temp_df, test_size=1 - rel_val, stratify=temp_df["label"], random_state=args.seed)

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_parquet = PROC_DIR / f"{name}.parquet"
        try:
            d.to_parquet(out_parquet, index=False)
        except Exception as e:
            print(f"Parquet save failed for {name} ({e}), falling back to CSV.")
            d.to_csv(PROC_DIR / f"{name}.csv", index=False)

    # Class balance report
    report = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_pos_ratio": float(train_df["label"].mean()),
        "val_pos_ratio": float(val_df["label"].mean()),
        "test_pos_ratio": float(test_df["label"].mean()),
    }
    (PROC_DIR / "split_report.json").write_text(pd.Series(report).to_json(indent=2))
    print("Saved processed splits and report:", report)


if __name__ == "__main__":
    main()
