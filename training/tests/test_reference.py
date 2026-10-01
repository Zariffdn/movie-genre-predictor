"""reference.py must do exactly what scikit-learn does. These tests use tiny hand-made data, so no corpus is needed."""

import math

import numpy as np
import pytest

from export import model_to_json
from model import make_vectorizer, predict_proba, train_models
from reference import ReferenceModel, sigmoid

TINY_MODEL = {
    "vocabulary": ["alien", "ship"],
    "idf": [2.0, 1.0],
    "genres": [{"name": "Science Fiction", "threshold": 0.5, "intercept": -1.0, "weights": [3.0, -1.0]}],
}


def test_tfidf_by_hand():
    vector = ReferenceModel(TINY_MODEL).vectorize("Alien alien ship, and a cowboy")
    # "alien" twice: (1 + ln 2) * idf 2.  "ship" once: (1 + ln 1) * idf 1.
    # "and", "cowboy" aren't in the vocabulary, and "a" is too short to be a token.
    alien, ship = (1 + math.log(2)) * 2.0, 1.0
    norm = math.sqrt(alien ** 2 + ship ** 2)
    assert vector == pytest.approx({0: alien / norm, 1: ship / norm})


def test_probability_and_top_words_by_hand():
    [result] = ReferenceModel(TINY_MODEL).predict("alien alien ship")
    vector = ReferenceModel(TINY_MODEL).vectorize("alien alien ship")
    logit = -1.0 + 3.0 * vector[0] - 1.0 * vector[1]
    assert result["probability"] == pytest.approx(1 / (1 + math.exp(-logit)))
    assert result["predicted"] is True
    # "ship" has a negative weight, so it pushes away from the genre and isn't listed.
    assert [word for word, _ in result["top_words"]] == ["alien"]


def test_text_with_no_known_words_gives_the_intercept_alone():
    [result] = ReferenceModel(TINY_MODEL).predict("zzz qqq")
    assert result["probability"] == pytest.approx(sigmoid(-1.0))
    assert result["top_words"] == []


def test_sigmoid_does_not_overflow():
    assert sigmoid(1000) == 1.0
    assert sigmoid(-1000) == 0.0
    assert sigmoid(0) == 0.5


def test_exported_json_reproduces_scikit_learn():
    texts = [
        "the alien ship lands and the crew fights the alien",
        "a cowboy rides into town to face the outlaw gang",
        "aliens from space invade the earth",
        "the sheriff and the cowboy chase the outlaw",
        "a space ship crew finds an alien planet",
        "the outlaw gang robs the town bank at noon",
    ] * 2
    Y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]] * 2)
    vectorizer = make_vectorizer().set_params(min_df=1, max_df=1.0)  # the real settings need a big corpus
    X = vectorizer.fit_transform(texts)
    models = train_models(X, Y, C=1.0)
    exported = model_to_json(vectorizer, ["Science Fiction", "Western"], models, [0.5, 0.5], test_metrics={})
    reference = ReferenceModel(exported)

    unseen = ["an alien cowboy!", "", "nothing known here"]
    expected = predict_proba(models, vectorizer.transform(texts + unseen))
    for text, row in zip(texts + unseen, expected):
        got = [result["probability"] for result in reference.predict(text)]
        assert got == pytest.approx(row, abs=1e-6)
