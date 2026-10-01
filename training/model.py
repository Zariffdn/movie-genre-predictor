"""The machine-learning pieces: TF-IDF features, one logistic regression per genre, and per-genre thresholds."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tokenizer import tokenize

# A word must appear in at least MIN_DF training plots (drops typos and
# one-off names) and in at most MAX_DF of them (drops "the", "and", "his",
# which appear in nearly every plot). Of what's left, keep the MAX_FEATURES
# most frequent words. That caps the app's model file at a few megabytes.
MIN_DF = 5
MAX_DF = 0.5
MAX_FEATURES = 20_000

# Candidate decision thresholds: 0.05, 0.06, ..., 0.95.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def make_vectorizer() -> TfidfVectorizer:
    """TF-IDF with settings that the Dart code can reproduce exactly.

    For a plot, each vocabulary word w gets the weight

        (1 + ln(count of w in the plot)) * idf(w)

    and then the whole vector is scaled to length 1 (L2 norm). idf(w) is
    ln((1 + n) / (1 + df(w))) + 1, where n is the number of training plots
    and df(w) is how many of them contain w. It is learned here and exported,
    so the app never needs to recompute it.
    """
    return TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,     # tokenize() already lowercases
        token_pattern=None,  # we supply our own tokenizer
        sublinear_tf=True,   # 1 + ln(count) rather than the raw count
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
    )


def train_models(X, Y: np.ndarray, C: float, class_weight: str | None = "balanced") -> list[LogisticRegression]:
    """Fit one independent yes/no classifier per genre (column of Y).

    class_weight="balanced" gives the positive examples of a rare genre
    (Western is about 3% of movies) as much total weight as the negatives,
    so the model can't do well just by always saying "no".
    """
    models = []
    for g in range(Y.shape[1]):
        model = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)
        model.fit(X, Y[:, g])
        models.append(model)
    return models


def predict_proba(models: list[LogisticRegression], X) -> np.ndarray:
    """An (n_movies, n_genres) matrix of P(genre | plot)."""
    return np.column_stack([model.predict_proba(X)[:, 1] for model in models])


def tune_thresholds(Y_true: np.ndarray, P: np.ndarray) -> np.ndarray:
    """For each genre, pick the threshold on THRESHOLD_GRID that maximises F1.

    Call this on validation data, never on test data. Otherwise the test
    score would be measured with a threshold that was picked by looking at
    the test answers.
    """
    thresholds = []
    for g in range(Y_true.shape[1]):
        scores = [f1_score(Y_true[:, g], P[:, g] >= t, zero_division=0) for t in THRESHOLD_GRID]
        thresholds.append(THRESHOLD_GRID[int(np.argmax(scores))])
    return np.array(thresholds)
