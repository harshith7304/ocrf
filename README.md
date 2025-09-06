# Fake Job Posting Detection (LSTM + BERT)

A complete, reproducible project to detect fraudulent job postings using deep learning. Includes data preparation, classical baselines, LSTM, and DistilBERT models, with evaluation and a simple inference app.

## Quick start (Windows PowerShell)

1. Create venv and install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Download/prepare dataset (default: EMSCAD)

```powershell
python scripts\download_dataset.py --dataset emscad
python scripts\prepare_data.py
```

3. Train models

```powershell
# Baseline
python scripts\train_baseline.py
# LSTM
python scripts\train_lstm.py
# DistilBERT
python scripts\train_bert.py
```

4. Evaluate and run inference

```powershell
python scripts\evaluate.py
python app\infer_cli.py --text "Example job description text..."
```

Optional: run Streamlit app

```powershell
streamlit run app\app.py
```

## Repo layout

- `scripts/` — data download, prep, training, evaluation, inference
	- `train_bert_fast.py` — DistilBERT (or other HF model) with class weighting & early stopping
	- `infer_bert.py` — single-text inference for saved HF model
	- `COLAB_INSTRUCTIONS.md` — end-to-end Colab workflow
- `src/` — reusable library code (preprocessing, datasets, models, metrics)
- `data/` — raw and processed data (git-ignored)
- `models/` — saved checkpoints
- `app/` — CLI and Streamlit app

## Notes

- Default dataset: EMSCAD (no credentials). Kaggle alternative supported with a Kaggle API token.
- CPU works; GPU recommended for faster BERT training.
- Reproducible seeds and deterministic splits.

## Speed tips / fast mode

Baseline (TF-IDF) is already fast. For LSTM and DistilBERT on limited hardware:

LSTM fast run example (subset + smaller dims + early stopping):

```powershell
python scripts\train_lstm.py --epochs 8 --limit_train 4000 --embed_dim 64 --hidden_dim 64 --target_metric f1 --early_stop_patience 2 --batch_size 128 --max_len 300
```

DistilBERT quick pass (1 epoch, shorter sequences):

```powershell
python scripts\train_bert.py --epochs 1 --batch_size 8 --max_len 128
```

Then scale up epochs or remove --limit_train once satisfied.

## Colab usage

If local training is slow, open Google Colab and run:

```bash
!git clone <your_repo_clone_url> fake-job-detector
%cd fake-job-detector
!pip install -r requirements.txt
!python scripts/prepare_data.py  # upload your CSV into data/raw first (Colab: use file browser)
!python scripts/train_bert.py --epochs 3 --batch_size 16 --max_len 256
```

After training, download `models/` artifacts from Colab for local inference.
