"""The app's prediction code, written in plain Python and run on model.json.

No NumPy and no scikit-learn: just a dictionary, math.log, math.exp and
loops. app/lib/src/genre_model.dart is a line-by-line translation of this
file. That makes this the easiest place to read how a prediction works.

train.py uses it twice:
  1. Run it on every test plot and check it gives the same probabilities as
     scikit-learn. That proves model.json holds the whole model.
  2. Ask it for the top words behind each genre when writing the fixture
     the Dart test compares against.
"""

import json
import math
from pathlib import Path

from tokenizer import tokenize

TOP_WORDS = 5


class ReferenceModel:
    def __init__(self, model: dict):
        self.vocabulary = model["vocabulary"]
        # word -> column number, e.g. {"aboard": 12, ...}
        self.index = {word: i for i, word in enumerate(self.vocabulary)}
        self.idf = model["idf"]
        self.genres = model["genres"]

    @classmethod
    def load(cls, path: Path) -> "ReferenceModel":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def vectorize(self, text: str) -> dict[int, float]:
        """TF-IDF vector as {column: value}. Words the model doesn't know are skipped."""
        counts: dict[int, int] = {}
        for token in tokenize(text):
            i = self.index.get(token)
            if i is not None:
                counts[i] = counts.get(i, 0) + 1

        vector = {i: (1 + math.log(count)) * self.idf[i] for i, count in counts.items()}

        # Scale to length 1, so long and short plots are comparable.
        # (Explicit loops instead of sum() on purpose: Python 3.12's sum()
        # uses extra-precise addition, and the Dart code adds in a plain loop.)
        squared = 0.0
        for value in vector.values():
            squared += value * value
        norm = math.sqrt(squared)
        if norm > 0:
            vector = {i: value / norm for i, value in vector.items()}
        return vector

    def predict(self, text: str) -> list[dict]:
        """One result per genre: probability, threshold, yes/no, and the top words behind it."""
        vector = self.vectorize(text)
        results = []
        for genre in self.genres:
            weights = genre["weights"]
            # A linear model's score is a sum of one term per word, so each
            # word's share of the score is exact, not an approximation.
            contributions = {i: weights[i] * value for i, value in vector.items()}
            logit = genre["intercept"]
            for contribution in contributions.values():
                logit += contribution
            probability = sigmoid(logit)
            results.append({
                "genre": genre["name"],
                "probability": probability,
                "threshold": genre["threshold"],
                "predicted": probability >= genre["threshold"],
                "top_words": self._top_words(contributions),
            })
        return results

    def _top_words(self, contributions: dict[int, float]) -> list[tuple[str, float]]:
        """The words that pushed hardest toward this genre (positive contributions only)."""
        positive = [(self.vocabulary[i], c) for i, c in contributions.items() if c > 0]
        # Largest contribution first; ties broken alphabetically so Dart gets the same order.
        positive.sort(key=lambda pair: (-pair[1], pair[0]))
        return positive[:TOP_WORDS]


def sigmoid(z: float) -> float:
    """1 / (1 + e^-z), written so that e^x never overflows for large |z|."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)
