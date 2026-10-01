import numpy as np
import pytest

from evaluate import score
from model import tune_thresholds


def test_threshold_tuning_finds_the_gap_between_classes():
    # Positives score 0.70-0.90, negatives 0.10-0.30. Any threshold in
    # (0.30, 0.70] is perfect, and the search returns the first one it tries.
    Y_true = np.array([[1], [1], [1], [0], [0], [0]])
    P = np.array([[0.9], [0.8], [0.7], [0.3], [0.2], [0.1]])
    [threshold] = tune_thresholds(Y_true, P)
    assert threshold == pytest.approx(0.31)


def test_always_yes_baseline_f1_is_2p_over_1_plus_p():
    # Predicting "yes" for everything: precision = prevalence p, recall = 1.
    Y_true = np.zeros((100, 15), dtype=int)
    Y_true[:20, 0] = 1  # genre 0 has prevalence 0.2
    result = score(Y_true, np.ones_like(Y_true))
    assert result["per_genre"]["Action"]["f1"] == pytest.approx(2 * 0.2 / 1.2, abs=1e-4)
