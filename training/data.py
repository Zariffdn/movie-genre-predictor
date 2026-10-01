"""Load the CMU Movie Summary Corpus and split it into train / validation / test."""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from genres import map_labels

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "MovieSummaries"


@dataclass(frozen=True)
class Movie:
    wikipedia_id: str
    title: str
    plot: str
    genres: tuple[str, ...]  # a tuple, so a frozen Movie really can't change


def load_movies(corpus_dir: Path = CORPUS_DIR) -> list[Movie]:
    """Join plot summaries with their genres. Returns movies that have at least one of our genres."""
    # movie.metadata.tsv: one movie per line. Column 3 is the title and
    # column 9 is a JSON object like {"/m/07s9rl0": "Drama", ...}.
    metadata: dict[str, tuple[str, list[str]]] = {}
    with open(corpus_dir / "movie.metadata.tsv", encoding="utf-8", newline="") as f:
        # QUOTE_NONE: titles can contain '"', which must not be treated as CSV quoting.
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            wikipedia_id, _, title, *_, genre_json = row
            metadata[wikipedia_id] = (title, list(json.loads(genre_json).values()))

    movies = []
    seen_plots = set()
    # plot_summaries.txt: "<wikipedia id>\t<summary>" per line.
    with open(corpus_dir / "plot_summaries.txt", encoding="utf-8") as f:
        for line in f:
            wikipedia_id, plot = line.rstrip("\n").split("\t", 1)
            # A few summaries appear under several ids. For example, five film
            # versions of "Madame X" share one. Only the first copy (with
            # metadata) is considered, so one text can't land in both train
            # and test. If that copy has none of our genres, the text is
            # dropped completely. That happens to one movie.
            if wikipedia_id not in metadata or plot in seen_plots:
                continue
            seen_plots.add(plot)
            title, labels = metadata[wikipedia_id]
            genres = tuple(map_labels(labels))
            if genres:
                movies.append(Movie(wikipedia_id, title, plot, genres))
    return movies


def split(movies: list[Movie], seed: int, val_fraction: float = 0.15,
          test_fraction: float = 0.15) -> tuple[list[Movie], list[Movie], list[Movie]]:
    """Shuffle with a fixed seed, then cut into train / validation / test."""
    shuffled = list(movies)
    random.Random(seed).shuffle(shuffled)
    n_test = round(len(shuffled) * test_fraction)
    n_val = round(len(shuffled) * val_fraction)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test
