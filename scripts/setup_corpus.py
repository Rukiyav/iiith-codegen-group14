import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import REPOS, REPO_CORPUS_DIR

def clone_repos():
    REPO_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS:
        target_dir = REPO_CORPUS_DIR / name
        if target_dir.exists():
            print(f"Repository {name} already exists at {target_dir}. Skipping clone.")
            continue
        print(f"Cloning {name} from {url}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target_dir)],
                check=True
            )
            print(f"Successfully cloned {name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    clone_repos()
