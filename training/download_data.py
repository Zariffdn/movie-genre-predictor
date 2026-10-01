"""Download and unpack the CMU Movie Summary Corpus into training/data/.

The corpus is about 46 MB compressed, so it is not committed to git.
Run this once before train.py:

    python training/download_data.py
"""

import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
DATA_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE = DATA_DIR / "MovieSummaries.tar.gz"
CORPUS_DIR = DATA_DIR / "MovieSummaries"


def main() -> None:
    if (CORPUS_DIR / "plot_summaries.txt").exists():
        print(f"Corpus already present in {CORPUS_DIR}")
        return

    DATA_DIR.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL} ...")
        # Download under a temporary name and rename at the end. An interrupted
        # download then leaves a .part file behind, never a broken archive.
        partial = ARCHIVE.with_name(ARCHIVE.name + ".part")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(ARCHIVE)

    print(f"Extracting {ARCHIVE.name} ...")
    with tarfile.open(ARCHIVE) as archive:
        # filter="data" refuses paths that would escape DATA_DIR (e.g. "../x").
        archive.extractall(DATA_DIR, filter="data")
    print(f"Done: {CORPUS_DIR}")


if __name__ == "__main__":
    main()
