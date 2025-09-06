import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"


def load(split: str) -> pd.DataFrame:
    p_parquet = PROC_DIR / f"{split}.parquet"
    p_csv = PROC_DIR / f"{split}.csv"
    if p_parquet.exists():
        print(f"Loading {p_parquet}")
        return pd.read_parquet(p_parquet)
    elif p_csv.exists():
        print(f"Loading {p_csv}")
        return pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"Missing split files: {p_parquet} or {p_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    args = parser.parse_args()

    train_df = load("train")
    val_df = load("val")
    test_df = load("test")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, args.ngram_max))),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)),
    ])

    pipe.fit(train_df.text, train_df.label)
    y_pred = pipe.predict(test_df.text)

    print(classification_report(test_df.label, y_pred, digits=4))

    # Save model
    import joblib
    MODELS_DIR = BASE / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "baseline_tfidf_logreg.joblib")


if __name__ == "__main__":
    main()
