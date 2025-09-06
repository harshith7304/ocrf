import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory of the saved HF model (e.g., models/bert_best)")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    enc = tok(args.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    label = int(torch.argmax(logits, dim=-1))
    print(f"Prediction: {'FAKE' if label==1 else 'LEGIT'}  probs={probs}")


if __name__ == "__main__":
    main()
