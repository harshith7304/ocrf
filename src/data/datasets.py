from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset


class TextLabelDataset(Dataset):
    def __init__(self, parquet_path: Path):
        self.df = pd.read_parquet(parquet_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {"text": r["text"], "label": int(r["label"])}
