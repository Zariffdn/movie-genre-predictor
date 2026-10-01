"""Scoring. Every function compares Y_true with Y_pred: 0/1 matrices of shape (n_movies, n_genres).

Scores are returned at full precision and rounded only when they're
written out (see export.rounded), so no number is ever rounded twice.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from genres import GENRES


def score(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    """Precision, recall and F1 per genre, plus the usual multi-label summaries.

    micro: pool every (movie, genre) decision, then compute P/R/F1 once.
           Common genres like Drama dominate it.
    macro: compute F1 per genre, then average. Every genre counts equally,
           so it shows whether rare genres (Western, War) work too.
    samples_f1: F1 per movie, averaged over movies.
    exact_match: share of movies whose whole genre set was exactly right.
                 Strict, so it is always low for multi-label problems.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0)
    per_genre = {
        name: {
            "precision": float(precision[g]),
            "recall": float(recall[g]),
            "f1": float(f1[g]),
            "support": int(support[g]),
        }
        for g, name in enumerate(GENRES)
    }
    summary = {}
    for average in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(Y_true, Y_pred, average=average, zero_division=0)
        summary[average] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    return {
        "per_genre": per_genre,
        **summary,
        "samples_f1": float(f1_score(Y_true, Y_pred, average="samples", zero_division=0)),
        "exact_match": float(accuracy_score(Y_true, Y_pred)),
        "share_with_no_genre_predicted": float(np.mean(Y_pred.sum(axis=1) == 0)),
    }


def bootstrap_f1(Y_true: np.ndarray, Y_pred: np.ndarray, seed: int, n_resamples: int = 1000) -> dict:
    """95% confidence intervals for micro and macro F1.

    Draw a new test set of the same size by picking movies at random with
    replacement, then score it. Repeat that 1,000 times. The middle 95% of
    those scores shows how much the result depends on which movies happened
    to land in the test set.
    """
    rng = np.random.default_rng(seed)
    n = len(Y_true)
    micro, macro = [], []
    for _ in range(n_resamples):
        rows = rng.integers(0, n, n)
        micro.append(f1_score(Y_true[rows], Y_pred[rows], average="micro", zero_division=0))
        macro.append(f1_score(Y_true[rows], Y_pred[rows], average="macro", zero_division=0))
    return {
        "micro_f1_95ci": [float(np.percentile(micro, 2.5)), float(np.percentile(micro, 97.5))],
        "macro_f1_95ci": [float(np.percentile(macro, 2.5)), float(np.percentile(macro, 97.5))],
    }
