"""Download MBPP (notebook URL) and CoDocBench."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.codoc import download_codoc
from src.data.mbpp import download_mbpp


def main() -> None:
    print("Downloading MBPP (sanitized-mbpp.json via requests)...")
    download_mbpp()
    print("Downloading CoDocBench (GitHub train/test JSONL)...")
    download_codoc()
    print("All datasets ready under data/processed/")


if __name__ == "__main__":
    main()
