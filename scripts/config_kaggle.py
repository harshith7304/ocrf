import json
import os
from pathlib import Path


def write_kaggle_json(username: str, key: str):
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = kaggle_dir / "kaggle.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    # On Linux/Mac, you would set permissions 600. On Windows it's OK.
    return cfg_path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    p = write_kaggle_json(args.username, args.key)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
