"""Side experiments behind the settings in model.py. Validation data only.

    python training/experiments.py     (about 4 minutes)

Each line prints validation macro-F1, with thresholds tuned on validation,
for one variation of the model:

  1. class_weight None vs "balanced", across C. Also shows the score at a
     plain 0.5 threshold, to see how much each one depends on tuning.
  2. Vocabulary size: 5k, 10k, 20k words, or every word that passes min_df/max_df.
  3. Single words only vs single words plus word pairs (bigrams).

The output is also saved to results/experiments.txt, and STUDY.md quotes
it. The test set is never used here. If it were, picking settings by their
test score would make the test score optimistic.
"""

from sklearn.metrics import f1_score

from data import load_movies, split
from model import make_vectorizer, predict_proba, train_models, tune_thresholds
from train import RESULTS_DIR, SEED, genre_matrix

_lines: list[str] = []


def report(line: str) -> None:
    """Print a line and keep it for results/experiments.txt."""
    print(line, flush=True)
    _lines.append(line)


def validation_macro_f1(X_train, Y_train, X_val, Y_val, C=1.0, class_weight="balanced"):
    models = train_models(X_train, Y_train, C, class_weight)
    P_val = predict_proba(models, X_val)
    thresholds = tune_thresholds(Y_val, P_val)
    tuned = f1_score(Y_val, P_val >= thresholds, average="macro", zero_division=0)
    at_half = f1_score(Y_val, P_val >= 0.5, average="macro", zero_division=0)
    return tuned, at_half


def main() -> None:
    train, val, _ = split(load_movies(), seed=SEED)
    Y_train, Y_val = genre_matrix(train), genre_matrix(val)
    train_text, val_text = [m.plot for m in train], [m.plot for m in val]

    report("Validation macro-F1 (thresholds tuned on validation unless marked 'at 0.5')")
    report("1. class_weight and C (20k words)")
    vectorizer = make_vectorizer()
    X_train, X_val = vectorizer.fit_transform(train_text), vectorizer.transform(val_text)
    for class_weight in (None, "balanced"):
        for C in (0.5, 1.0, 2.0, 4.0, 8.0):
            tuned, at_half = validation_macro_f1(X_train, Y_train, X_val, Y_val, C, class_weight)
            report(f"   class_weight={str(class_weight):8}  C={C:<4}  tuned {tuned:.4f}   at 0.5 {at_half:.4f}")

    report("2. vocabulary size (C=1, balanced)")
    for max_features in (5_000, 10_000, 20_000, None):
        vectorizer = make_vectorizer().set_params(max_features=max_features)
        X_train, X_val = vectorizer.fit_transform(train_text), vectorizer.transform(val_text)
        tuned, _ = validation_macro_f1(X_train, Y_train, X_val, Y_val)
        report(f"   {len(vectorizer.vocabulary_):>6} words  tuned {tuned:.4f}")

    report("3. word pairs (C=1, balanced)")
    for max_features in (40_000, 100_000):
        vectorizer = make_vectorizer().set_params(ngram_range=(1, 2), max_features=max_features)
        X_train, X_val = vectorizer.fit_transform(train_text), vectorizer.transform(val_text)
        tuned, _ = validation_macro_f1(X_train, Y_train, X_val, Y_val)
        report(f"   words + pairs, {len(vectorizer.vocabulary_):>6} features  tuned {tuned:.4f}")

    (RESULTS_DIR / "experiments.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
