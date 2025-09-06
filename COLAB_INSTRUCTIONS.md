## Colab Guide: Fake Job Posting Detection (LSTM + DistilBERT)

This guide lets you (or a collaborator) reproduce training quickly on Google Colab (with GPU) and bring back models locally.

### 1. Runtime
Runtime > Change runtime type > GPU.

### 2. Clone (if repo is public)
```bash
!git clone <REPLACE_WITH_REPO_URL> fake-job-detector
%cd fake-job-detector
```
If not pushing to Git yet, instead upload the project folder via Colab file browser and `%cd` into it.

### 3. Install dependencies (trimmed set)
```bash
!pip install -q transformers==4.56.1 datasets==4.0.0 scikit-learn==1.7.1 pandas pyarrow torch accelerate==1.10.1 matplotlib seaborn bs4 emoji html5lib
```

### 4. Upload dataset
Left panel > Files > Upload `EMSCAD.csv` (or `fake_job_postings.csv`) then run:
```python
import pathlib, shutil
pathlib.Path('data/raw').mkdir(parents=True, exist_ok=True)
shutil.move('EMSCAD.csv','data/raw/EMSCAD.csv')  # adjust name if different
```

### 5. Prepare data
```python
!python scripts/prepare_data.py
```
Outputs: `data/processed/{train,val,test}.parquet` + `split_report.json`.

### 6. Train baseline (sanity check)
```python
!python scripts/train_baseline.py
```

### 7. Fast LSTM (subset) example
```python
!python scripts/train_lstm.py --epochs 6 --limit_train 6000 --embed_dim 64 --hidden_dim 64 --target_metric f1 --early_stop_patience 2 --batch_size 128 --max_len 300
```

### 8. DistilBERT training (enhanced fast script)
`train_bert_fast.py` adds class weighting + early stopping.
```python
!python scripts/train_bert_fast.py --model distilbert-base-uncased --epochs 3 --batch_size 16 --max_len 256 --patience 2 --lr 5e-5
```

### 9. Evaluate baseline or BERT
Baseline confusion matrix:
```python
!python scripts/evaluate.py
```
BERT quick inference after training:
```python
!python scripts/infer_bert.py --model_dir models/bert_best --text "Work from home pay fee first"
```

### 10. Download trained BERT model
```python
import shutil, zipfile, os
shutil.make_archive('bert_best','zip','models/bert_best')
from google.colab import files
files.download('bert_best.zip')
```

### 11. (Optional) Push to Hugging Face Hub
```python
from huggingface_hub import login
login()  # paste token
!pip install -q huggingface_hub
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_dir = 'models/bert_best'
from huggingface_hub import HfApi
# Example: skip if private
!huggingface-cli upload <your-username>/fake-job-detector $model_dir/ --repo-type model --private
```

### 12. Local inference after download
Unzip `bert_best.zip` into `models/bert_best/` locally, then:
```powershell
python scripts\infer_bert.py --model_dir models\bert_best --text "Enter job description here"
```

### Troubleshooting
| Issue | Fix |
|-------|-----|
| OOM on GPU | Reduce `--batch_size`, `--max_len` |
| Low fraud recall | Increase epochs, use full train set, ensure class weighting (fast script) |
| Hugging Face 401 | Ensure no invalid HF_TOKEN env var; login with `huggingface_hub.login()` |

### Next Ideas
- Switch to `microsoft/deberta-v3-small`
- Add focal loss for extreme imbalance
- Export ONNX for CPU inference speed

---
Maintainer: Update `<REPLACE_WITH_REPO_URL>` before sharing.
