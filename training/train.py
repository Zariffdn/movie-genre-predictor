"""Train the genre model, score it honestly, and export it to the Flutter app.

    python training/download_data.py   # once
    python training/train.py

1. Load about 38k plots with their genres, and split them 70% train,
   15% validation and 15% test.
2. Learn the TF-IDF vocabulary and IDF weights from the training plots only.
3. For each C in C_GRID, train 15 logistic regressions on train, pick each
   genre's threshold on validation, and record validation macro-F1. Keep
   the C that scores best.
4. Score that model on the test set, once. No choice about the model was
   made by looking at test data, so this estimates how it does on plots it
   has never seen.
5. Write the model to a staging file and check that it reproduces
   scikit-learn. Only if it does, replace the app's model.json. Then write
   the fixture for the Dart parity test.
"""

import time
from pathlib import Path

import numpy as np

from data import load_movies, split
from evaluate import bootstrap_f1, score
from export import model_to_json, parity_fixture, rounded, write_json
from genres import GENRES
from model import make_vectorizer, predict_proba, train_models, tune_thresholds
from reference import ReferenceModel

SEED = 42
# C is the inverse of regularisation strength. Small C keeps weights small,
# giving a simpler model. Large C lets the model fit the training plots more
# closely, and eventually overfit.
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PARITY_PLOTS = 40
MAX_PROBABILITY_DIFFERENCE = 1e-6

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "app" / "assets" / "model.json"
FIXTURE_PATH = ROOT / "app" / "test" / "fixtures" / "parity_cases.json"
RESULTS_DIR = ROOT / "training" / "results"

# Odd inputs for the parity test, on top of real test-set plots.
EDGE_CASES = [
    "",
    "!!! ??? 123 ...",
    "Zzyzx qwertyuiop blorfing",
    "A LONELY COWBOY RIDES INTO A FRONTIER TOWN TO FACE THE OUTLAW GANG.",
    "zombie zombie zombie zombie survivors",
    # Accents, an emoji, a dotted capital I and a Kelvin sign. None of them may be
    # lowercased into ASCII, so the last two words must come out as "stanbul" and "elvin".
    "Amélie, une serveuse à Montmartre, décide de changer la vie des autres. 🎬 "
    "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}stanbul \N{KELVIN SIGN}elvin",
]


def genre_matrix(movies) -> np.ndarray:
    """0/1 matrix: one row per movie, one column per genre (in GENRES order)."""
    return np.array([[genre in movie.genres for genre in GENRES] for movie in movies], dtype=int)


def main() -> None:
    started = time.time()

    # 1. Data --------------------------------------------------------------
    movies = load_movies()
    train, val, test = split(movies, seed=SEED)
    Y_train, Y_val, Y_test = genre_matrix(train), genre_matrix(val), genre_matrix(test)
    print(f"{len(movies)} movies -> train {len(train)}, validation {len(val)}, test {len(test)}")

    # 2. Features: fitted on train only, so no information leaks from val/test.
    vectorizer = make_vectorizer()
    X_train = vectorizer.fit_transform([m.plot for m in train])
    X_val = vectorizer.transform([m.plot for m in val])
    X_test = vectorizer.transform([m.plot for m in test])
    print(f"vocabulary: {len(vectorizer.vocabulary_)} words")

    # 3. Choose C and the thresholds using validation data ------------------
    search = []
    best = None
    for C in C_GRID:
        models = train_models(X_train, Y_train, C)
        P_val = predict_proba(models, X_val)
        thresholds = tune_thresholds(Y_val, P_val)
        val_macro_f1 = score(Y_val, P_val >= thresholds)["macro"]["f1"]
        search.append({"C": C, "val_macro_f1": val_macro_f1})
        print(f"  C={C:<5} validation macro-F1 {val_macro_f1:.4f}")
        if best is None or val_macro_f1 > best["val_macro_f1"]:
            best = {"C": C, "val_macro_f1": val_macro_f1, "models": models, "thresholds": thresholds}
    C, models, thresholds = best["C"], best["models"], best["thresholds"]
    print(f"chosen C={C}")

    # 4. Test: the first and only time the test labels are used -------------
    P_test = predict_proba(models, X_test)
    Y_pred = P_test >= thresholds
    test_scores = score(Y_test, Y_pred)
    test_scores.update(bootstrap_f1(Y_test, Y_pred, seed=SEED))
    baselines = {
        # Same model, but every genre uses 0.5: shows what threshold tuning bought.
        "same_model_threshold_0.5": score(Y_test, P_test >= 0.5),
        # Say "yes" to every genre for every movie: the score to beat.
        "always_predict_every_genre": score(Y_test, np.ones_like(Y_test)),
    }

    # 5. Export to a staging file, verify it, then hand it to the app -------
    staging = MODEL_PATH.with_name(MODEL_PATH.name + ".tmp")
    write_json(model_to_json(vectorizer, GENRES, models, thresholds,
                             rounded(app_metrics(test_scores, baselines, thresholds, len(test)))),
               staging, compact=True)
    reference = ReferenceModel.load(staging)  # read back from disk, like the app does
    export_check = verify_export(reference, test, P_test, thresholds)
    print(f"export check: max probability difference {export_check['max_probability_difference']:.1e}, "
          f"{export_check['decisions_that_differ']} of {export_check['decisions_checked']} decisions differ")
    if export_check["max_probability_difference"] > MAX_PROBABILITY_DIFFERENCE:
        staging.unlink()
        raise SystemExit("The export does not reproduce scikit-learn's predictions; app/assets/model.json was left as it was.")
    staging.replace(MODEL_PATH)

    texts = [test[i].plot for i in parity_plot_rows(P_test, thresholds)] + EDGE_CASES
    P_fixture = predict_proba(models, vectorizer.transform(texts))
    write_json(parity_fixture(texts, P_fixture, thresholds, reference), FIXTURE_PATH, compact=True)

    results = {
        "data": {"movies": len(movies), "train": len(train), "validation": len(val), "test": len(test),
                 "vocabulary": len(vectorizer.vocabulary_), "seed": SEED},
        "c_search": search,
        "chosen_C": C,
        "thresholds": {name: float(t) for name, t in zip(GENRES, thresholds)},
        "test": test_scores,
        "test_baselines": baselines,
        "export_check": export_check,
        "model_file_bytes": MODEL_PATH.stat().st_size,
    }
    write_json(rounded(results), RESULTS_DIR / "metrics.json")
    (RESULTS_DIR / "results.md").write_text(results_markdown(results), encoding="utf-8", newline="\n")
    print()
    print(results_markdown(results))
    print(f"Wrote {MODEL_PATH.relative_to(ROOT)} ({results['model_file_bytes'] / 1e6:.1f} MB), "
          f"{FIXTURE_PATH.relative_to(ROOT)} and training/results/ in {time.time() - started:.0f}s")


def parity_plot_rows(P_test: np.ndarray, thresholds: np.ndarray) -> list[int]:
    """Test rows for the parity fixture: the first PARITY_PLOTS, plus, for any
    genre that none of those is predicted as, the first test plot that is.
    Then the Dart test sees every genre say "yes" at least once."""
    rows = list(range(PARITY_PLOTS))
    for g in range(len(GENRES)):
        if not any(P_test[i, g] >= thresholds[g] for i in rows):
            row = next((i for i in range(len(P_test)) if P_test[i, g] >= thresholds[g]), None)
            if row is None:
                raise SystemExit(f"No test plot is predicted as {GENRES[g]}, so the parity fixture can't cover it.")
            rows.append(row)
    return rows


def verify_export(reference: ReferenceModel, movies, P_sklearn: np.ndarray, thresholds: np.ndarray) -> dict:
    """Predict every test plot from model.json (via reference.py) and compare with scikit-learn."""
    max_difference = 0.0
    differ = 0
    for movie, row in zip(movies, P_sklearn):
        for result, p, threshold in zip(reference.predict(movie.plot), row, thresholds):
            max_difference = max(max_difference, abs(result["probability"] - p))
            differ += int(result["predicted"] != (p >= threshold))
    return {
        "max_probability_difference": float(max_difference),
        "decisions_checked": len(movies) * len(GENRES),
        "decisions_that_differ": differ,
    }


def app_metrics(test_scores: dict, baselines: dict, thresholds: np.ndarray, n_test: int) -> dict:
    """The small subset of test results that the app's About screen shows."""
    always_yes = baselines["always_predict_every_genre"]["per_genre"]
    return {
        "test_movies": n_test,
        "micro_f1": test_scores["micro"]["f1"],
        "macro_f1": test_scores["macro"]["f1"],
        "macro_f1_95ci": test_scores["macro_f1_95ci"],
        "per_genre": {
            name: {**test_scores["per_genre"][name],
                   "threshold": float(t),
                   "always_yes_f1": always_yes[name]["f1"]}
            for name, t in zip(GENRES, thresholds)
        },
    }


def results_markdown(results: dict) -> str:
    test, base = results["test"], results["test_baselines"]
    at_half = base["same_model_threshold_0.5"]["per_genre"]
    always = base["always_predict_every_genre"]["per_genre"]
    d = results["data"]
    lines = [
        f"Test set: {d['test']:,} movies the model never saw during training or tuning "
        f"(train {d['train']:,}, validation {d['validation']:,}). Chosen C = {results['chosen_C']}.",
        "",
        "| Genre | Test movies | Threshold | Precision | Recall | F1 | F1 at 0.5 | F1 always-yes |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in GENRES:
        g = test["per_genre"][name]
        lines.append(f"| {name} | {g['support']:,} | {results['thresholds'][name]:.2f} | {g['precision']:.3f} "
                     f"| {g['recall']:.3f} | **{g['f1']:.3f}** | {at_half[name]['f1']:.3f} | {always[name]['f1']:.3f} |")
    for avg in ("micro", "macro"):
        s = test[avg]
        lines.append(f"| *{avg} average* | | | {s['precision']:.3f} | {s['recall']:.3f} | **{s['f1']:.3f}** "
                     f"| {base['same_model_threshold_0.5'][avg]['f1']:.3f} "
                     f"| {base['always_predict_every_genre'][avg]['f1']:.3f} |")
    lo_mi, hi_mi = test["micro_f1_95ci"]
    lo_ma, hi_ma = test["macro_f1_95ci"]
    check = results["export_check"]
    lines += [
        "",
        f"- 95% bootstrap interval: micro-F1 {lo_mi:.3f} to {hi_mi:.3f}, macro-F1 {lo_ma:.3f} to {hi_ma:.3f}",
        f"- Per-movie (samples) F1: {test['samples_f1']:.3f}. Exact genre set right: {test['exact_match']:.1%} of movies. "
        f"No genre predicted: {test['share_with_no_genre_predicted']:.1%} of movies.",
        f"- Export check: on all {check['decisions_checked']:,} test (movie, genre) pairs, the exported model.json "
        f"gives probabilities within {check['max_probability_difference']:.1e} of scikit-learn's, "
        f"and {check['decisions_that_differ']} yes/no decisions differ.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
