from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

PROC_DIR = BASE / "data" / "processed"
PLOTS_DIR = BASE / "data" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


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
    test_df = load("test")
    # Placeholder: baseline predictions only (extend as needed)
    import joblib
    pipe = joblib.load(BASE / "models" / "baseline_tfidf_logreg.joblib")
    y_pred = pipe.predict(test_df.text)

    print(classification_report(test_df.label, y_pred, digits=4))

    cm = confusion_matrix(test_df.label, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {PLOTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
