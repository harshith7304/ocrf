import argparse
import os
import sys
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def info(msg: str):
    print(f"[download_dataset] {msg}")


def save_readme():
    readme = RAW_DIR / "README.txt"
    if not readme.exists():
        readme.write_text(
            """
This folder stores raw datasets.

Supported options:
- EMSCAD (preferred, no credentials): Manually download and place CSV here.
  Suggested sources: search for "Employment Scam Aegean Dataset (EMSCAD)".
  Typical filenames might include 'EMSCAD.csv' or similar.

- Kaggle: Fake Job Posting Prediction
  Requires Kaggle credentials. Set env vars KAGGLE_USERNAME & KAGGLE_KEY
  or place kaggle.json under %USERPROFILE%\.kaggle\kaggle.json.

If automatic downloads fail, place one of the following files here:
- fake_job_postings.csv (Kaggle)
- emscad.csv or EMSCAD.csv (EMSCAD)

The preparation script will auto-detect schema and proceed.
            """.strip()
        )


def try_kaggle():
    try:
        import subprocess
        # Ensure kaggle is installed
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "kaggle"], check=True)
        # Download dataset
        info("Using Kaggle to download 'shubhamchandak94/fake-job-postings' (or fallback)")
        # Multiple popular slugs to try
        slugs = [
            # Known mirrors/variants
            "shivamb/real-or-fake-jobs",
            "amansaxena/fake-job-postings",
            "arashnic/fake-job-postings",
            "adityakhatri/fake-job-postings",
            "shubhamchandak94/fake-job-postings",
        ]
        for slug in slugs:
            r = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(RAW_DIR), "-w"], capture_output=True, text=True)
            if r.returncode == 0:
                info(f"Downloaded from Kaggle: {slug}")
                # Unzip any zips
                for p in RAW_DIR.glob("*.zip"):
                    subprocess.run(["powershell", "-NoProfile", "-Command", f"Expand-Archive -Path \"{p}\" -DestinationPath \"{RAW_DIR}\" -Force"], check=False)
                return True
            else:
                info(f"Kaggle slug failed: {slug} -> {r.stderr.strip()[:200]}")
        return False
    except Exception as e:
        info(f"Kaggle download error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["emscad", "kaggle"], default="emscad")
    args = parser.parse_args()

    save_readme()

    if args.dataset == "kaggle":
        ok = try_kaggle()
        if not ok:
            info("Kaggle download failed. Please ensure Kaggle API credentials are configured or place CSV in data/raw.")
            sys.exit(1)
        info("Done.")
        return

    # EMSCAD path (manual friendly). Provide guidance only.
    info("EMSCAD selected. Attempting community mirrors is disabled to avoid unreliable links.")
    info("Please download EMSCAD CSV manually and place it under data/raw (e.g., EMSCAD.csv). The next step 'prepare_data.py' will detect it.")
    # Exit 0 so pipeline can continue; prep script will check for files.


if __name__ == "__main__":
    main()
