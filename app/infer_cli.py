import argparse
from pathlib import Path
import joblib
from src.utils.text_cleaning import normalize_text

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    pipe = joblib.load(BASE / "models" / "baseline_tfidf_logreg.joblib")
    text = normalize_text(args.text)
    pred = pipe.predict([text])[0]
    proba = max(pipe.predict_proba([text])[0]) if hasattr(pipe, "predict_proba") else None
    label = "FAKE" if pred == 1 else "LEGIT"
    print(f"Prediction: {label}")
    if proba is not None:
        print(f"Confidence: {proba:.3f}")


if __name__ == "__main__":
    main()
