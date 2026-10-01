# Study guide

This walks through the project in the order the data flows, from raw corpus to a genre on a phone screen. Read each section with the file it names open next to it.

Every number here comes from a real run:
- the scores come from `training/train.py` and are saved in `training/results/`;
- the comparisons between settings come from `training/experiments.py` and are saved in `training/results/experiments.txt`;
- the two tables in §13 come from `training/top_words.py`;
- the rest come from one-off checks that no script repeats: corpus counts, timings, file sizes, and the Crime-label, Stooges and Bollywood breakdowns (§3, §8 and §13).

**Contents**

1. [The 30-second pitch](#1-the-30-second-pitch)
2. [The big picture](#2-the-big-picture)
3. [The data](#3-the-data) · `download_data.py`, `data.py`, `genres.py`
4. [The tokenizer](#4-the-tokenizer) · `tokenizer.py`, `tokenizer.dart`
5. [Features: TF-IDF](#5-features-tf-idf) · `model.py`
6. [The model: one logistic regression per genre](#6-the-model-one-logistic-regression-per-genre) · `model.py`
7. [Thresholds](#7-thresholds) · `model.py`
8. [Evaluation](#8-evaluation) · `evaluate.py`, `train.py`
9. [Export and the reference implementation](#9-export-and-the-reference-implementation) · `export.py`, `reference.py`
10. [Prediction in Dart](#10-prediction-in-dart) · `genre_model.dart`
11. [The Flutter app](#11-the-flutter-app) · `main.dart`, `home_page.dart`, `about_page.dart`
12. [Tests](#12-tests)
13. [What the model learned, and its biases](#13-what-the-model-learned-and-its-biases) · `top_words.py`
14. [Limitations and next steps](#14-limitations-and-next-steps)
15. [Exercises](#15-exercises)

---

## 1. The 30-second pitch

> "It's a Flutter app that predicts a movie's genres from its plot summary, and it runs entirely on the phone. I built a TF-IDF plus logistic regression model in Python from about 38,000 Wikipedia plot summaries, split into training, validation and test sets. It's one yes/no classifier per genre, with a decision threshold tuned for each genre on validation data. I exported the weights to JSON and wrote the prediction in plain Dart. There's no server, and the Android release build doesn't even request internet permission. A test checks that the Dart code reproduces the Python model's probabilities to within a millionth. On a held-out test set it gets a micro-F1 of 0.64 and a macro-F1 of 0.60, against 0.26 and 0.24 for a baseline that tags every genre. The app also shows which words drove each prediction. Because the model is linear, that breakdown of the score is exact."

## 2. The big picture

```
training/ (Python)                                        app/ (Flutter)
corpus → map labels → split → tokenize → TF-IDF → 15 × LR → model.json → tokenize → TF-IDF → 15 × sigmoid → genres
```

**Why on-device?** It works offline, it's instant, it costs nothing to run, and the user's text never leaves the phone. The price is that the model has to be small and simple enough to re-implement by hand.

**Why TF-IDF + logistic regression, and not a neural network?**
- It's a strong baseline for text classification and trains in about a minute on a laptop.
- Prediction is a dot product. It takes about 100 lines of Dart and needs no ML runtime (no TensorFlow Lite, no ONNX).
- It's interpretable. A genre's score is exactly a sum of one term per word, so you can show *which words* pushed it up.
- The model file is 3.5 MB. A sentence-embedding model would be tens of MB and need a native runtime.

**Repo map:**
- `training/` is Python and produces `app/assets/model.json`.
- `app/` is Flutter and consumes it.
- `shared/` holds the test cases both languages must pass.

## 3. The data

Open `training/download_data.py`, `training/data.py` and `training/genres.py`.

**The corpus.** The CMU Movie Summary Corpus (Bamman, O'Connor & Smith, ACL 2013) has 42,306 plot summaries from a 2012 English Wikipedia dump, plus metadata for 81,741 movies from Freebase. Two files matter here:
- `plot_summaries.txt`: `wikipedia_id <TAB> summary`
- `movie.metadata.tsv`: nine tab-separated columns. Column 9 is a JSON object of genres, like `{"/m/07s9rl0": "Drama", ...}`

It's licensed CC BY-SA 3.0 US. That means you must credit it (the README and the app's About page do) and share adaptations under the same licence, which is why `model.json` and the fixture file carry CC BY-SA.

**Downloading (`download_data.py`).** The archive is saved under a temporary `.part` name and renamed only once it's complete. An interrupted download can't leave a broken archive that later runs would try to unpack.

**Loading (`load_movies`).** It reads the metadata into a dictionary keyed by Wikipedia ID, then streams the plots and joins them to it. Two details are worth being able to explain:
- **`quoting=csv.QUOTE_NONE`:** nine titles begin with `"`, such as `"Manos": The Hands of Fate`. The default CSV reader treats that quote as the start of a quoted field and silently drops it (no lines merge, but those titles change). Turning quoting off keeps every field exactly as written.
- **De-duplication:** the 42,306 summaries contain only 42,298 distinct texts, because 5 texts appear under several IDs. For example, five film versions of *Madame X* share one summary. If one copy landed in train and another in test, the model would be "tested" on a plot it had memorised. Only the first copy (with metadata) is considered. If that copy has none of our genres, the text is dropped entirely. That happens once, costing one movie (*The Warrens of Virginia*); a slightly smarter rule would keep the first copy that *has* genres. This only catches exact copies. §8 covers what it misses.

**Genres (`genres.py`).** Freebase has 363 genre labels. Many describe the *production* rather than the *story* ("Black-and-white", "World cinema", "Short Film", "Bollywood", "Japanese Movies"), so they're left out. The rule used here is to keep story genres with at least 1,000 movies and fold obvious sub-genres into them. That gives 15:

Action, Adventure, Comedy, Crime, Drama, Family, Fantasy, Horror, Musical, Mystery, Romance, Science Fiction, Thriller, War, Western.

A sub-genre can count toward several genres ("Romantic comedy" → Comedy and Romance; "Crime Thriller" → Crime and Thriller). The map is hand-written, and that's deliberate: it's short, readable, and every choice in it can be defended. It has one known gap: the plain label "Crime" (on 26 movies with plots) is missing, so 18 of the kept movies lack a Crime tag they should have. Fixing it means retraining, so the report lists it as a limitation instead.

A movie whose labels map to none of the 15 is **dropped**, not treated as "no genre". An empty label list in Freebase usually means "nobody tagged it", so using it as a negative example would teach the model something false.

**What's left:** 38,397 movies, 2.21 genres per movie on average. Prevalence ranges from Drama (51.6%) down to Western (2.7%). That imbalance drives several later decisions.

**The split (`split`).** The movies are shuffled with a fixed seed (42) and cut 70 / 15 / 15:

| Set | Movies | Used for |
|---|--:|---|
| train | 26,877 | learning the vocabulary, IDF and weights |
| validation | 5,760 | choosing C and each genre's threshold |
| test | 5,760 | one final, honest score |

There are three sets because every choice you make by looking at a dataset's score slightly overfits to that dataset. Validation absorbs that, so the test set stays untouched until the end.

## 4. The tokenizer

Open `training/tokenizer.py`, `app/lib/src/tokenizer.dart` and `shared/tokenizer_cases.json`.

**The rule:** a token is a run of 2 or more ASCII letters, lowercased. Everything else is a separator.

```
"Spider-Man's 2nd fight!"  →  ["spider", "man", "nd", "fight"]
```

Python: `[m.lower() for m in re.findall(r"[A-Za-z]{2,}", text)]`
Dart: `[for (final m in RegExp(r'[A-Za-z]{2,}').allMatches(text)) m[0]!.toLowerCase()]`

**Why so strict?** The model is trained in Python and runs in Dart. If the two tokenizers disagreed on even one word, the phone would look up different vocabulary entries than the model learned from, and its predictions would drift from the scores we measured without any error message. Unicode is where languages disagree most:
- **Case rules differ.** In Python, `"\N{KELVIN SIGN}".lower()` gives an ASCII `'k'`, and `"İ".lower()` gives two characters (an `i` plus a combining dot). Dart has its own case tables.
- **`\w` differs.** It matches different characters in Python's `re` and Dart's `RegExp`.
- **Strings are counted differently.** Python strings are sequences of code points. Dart strings are UTF-16 code units, so an emoji is *two* units.

Matching only `[A-Za-z]` avoids all of that. A match contains nothing but ASCII letters, and lowercasing ASCII is the same everywhere.

**The cost:** accented words get split (`café` → `caf`). In English Wikipedia plots that mostly affects names, and it hardly matters to the model.

**One-letter words** ("a", "I") are dropped because they carry no genre signal.

**How we know the two agree:** `shared/tokenizer_cases.json` holds 19 tricky inputs (accents, emoji, the Kelvin sign, `İ`, full-width letters, digits, CRLF…), and *both* test suites check them. Its non-ASCII characters are stored as `\u` escapes, so a Kelvin sign can't pass for a K in an editor. The parity fixture also stores the Python tokens for all 48 of its texts (42 real plots plus 6 edge cases), and the Dart test compares them.

## 5. Features: TF-IDF

Open `training/model.py`, function `make_vectorizer`.

A model needs numbers, not text. TF-IDF turns a plot into a vector with one slot per vocabulary word (20,000 of them). Most slots are 0 for any given plot.

For each word *w* in a plot:

```
tf(w)  = 1 + ln(count of w in this plot)          ← sublinear_tf=True
idf(w) = ln((1 + n) / (1 + df(w))) + 1            ← n = training plots, df = plots containing w
value  = tf(w) × idf(w)
```

Then the whole vector is divided by its length (L2 norm), so it has length 1.

- **tf, and why the log:** "zombie" 10 times is more zombie-ish than once, but not 10 times more. With the log, 10 occurrences count about 3.3× as much as one.
- **idf:** rare words are more informative. "saloon" is in few plots, so it gets a high idf. "life" is in many, so it gets a low one.
- **L2 normalisation:** without it, long plots would have bigger vectors and more extreme scores just for being long.

**A worked example** (it's `test_tfidf_by_hand` in `tests/test_reference.py`). Take a vocabulary of `alien` (idf 2) and `ship` (idf 1), and the text "alien alien ship":
- alien: (1 + ln 2) × 2 = 3.386
- ship: (1 + ln 1) × 1 = 1.000
- length = √(3.386² + 1²) = 3.531, so the vector is **alien 0.959, ship 0.283**

**Vocabulary settings:**
- `min_df=5`: a word must appear in at least 5 training plots. That drops typos and one-off character names.
- `max_df=0.5`: a word in more than half of all plots is dropped ("the", "and", "his"). That's why there's no stop-word list. scikit-learn's built-in English list has known problems anyway (it includes "fire", "system" and "bill").
- `max_features=20_000`: of what's left, keep the 20,000 most frequent words.

**Fitted on training data only.** `fit_transform(train)`, then `transform(val)` and `transform(test)`. If the IDF were computed on test plots too, information about the test set would leak into the features.

**Why 20,000 words and single words only?** From `experiments.py` (validation macro-F1):

| Vocabulary | Validation macro-F1 |
|---|--:|
| 5,000 words | 0.5998 |
| 10,000 words | 0.6093 |
| **20,000 words** | **0.6149** |
| all 31,633 words | 0.6166 |
| 40,000 features: words + word pairs | 0.6125 |
| 100,000 features: words + word pairs | 0.6145 |

Going from 20k to every word gains 0.002 and makes the file about 60% bigger. Word pairs (bigrams) gain nothing and would make the Dart side more complicated. So the choice is 20k single words.

## 6. The model: one logistic regression per genre

Open `training/model.py`, functions `train_models` and `predict_proba`.

**Multi-label, not multi-class.** Multi-class means pick one of N labels. Multi-label means any subset of them, which fits here: a movie can be Comedy *and* Romance *and* Drama. The simplest approach, called *binary relevance* or *one-vs-rest*, trains an independent yes/no classifier for each genre. That's 15 models.

**Logistic regression** for one genre:

```
score z = b + w₁x₁ + w₂x₂ + … + w₂₀₀₀₀x₂₀₀₀₀      (x = the TF-IDF vector, w = learned weights, b = intercept)
P(genre) = sigmoid(z) = 1 / (1 + e^(−z))            (squashes any number into 0–1)
```

Training finds the `w` and `b` that minimise *log loss* (it punishes confident wrong answers heavily), plus an L2 penalty on the size of the weights. The score `z` is in *log-odds*: 0 means 50%, and each +1 multiplies the odds by e ≈ 2.7.

Continuing the worked example with weights alien = 3, ship = −1 and intercept −1: z = −1 + 3(0.959) − 1(0.283) = 1.594, and sigmoid(1.594) = **0.83**.

**C (regularisation).** C is the *inverse* of the penalty strength. Small C keeps the weights small, giving a simpler model that might underfit. Large C lets the weights grow to fit the training plots closely, and the model might overfit. `train.py` tries six values, and validation macro-F1 peaks in the middle, which is the textbook shape:

| C | 0.25 | 0.5 | **1** | 2 | 4 | 8 |
|---|--:|--:|--:|--:|--:|--:|
| val macro-F1 | 0.6066 | 0.6128 | **0.6149** | 0.6132 | 0.6081 | 0.6012 |

**`class_weight="balanced"`.** Westerns are 2.7% of movies. Unweighted, a model can get 97% accuracy by always saying "no". Balanced weighting scales each class by `n / (2 × count of that class)`, so the 2.7% of positives carry as much total weight as the 97.3% of negatives. From `experiments.py` (validation macro-F1):

| | tuned thresholds | plain 0.5 threshold |
|---|--:|--:|
| no weighting, C=1 | 0.6102 | 0.4376 |
| no weighting, best C (2) | 0.6151 | 0.4916 |
| balanced, C=1 | 0.6149 | 0.5901 |

With tuned thresholds the two end up the same, because tuning compensates. Balanced weighting makes the model much less dependent on the thresholds being right, though, so it's the safer choice.

**Solver:** scikit-learn's default, L-BFGS. It doesn't penalise the intercept, which you want. The older `liblinear` solver does penalise it.

## 7. Thresholds

Open `training/model.py`, function `tune_thresholds`.

The model outputs a probability, but the app has to say yes or no. 0.5 isn't special. The best cut-off depends on how common the genre is and what you're optimising. Here we optimise **F1** (below), per genre: try every threshold from 0.05 to 0.95 in steps of 0.01 on the *validation* set and keep the best.

The chosen thresholds range from 0.39 (Drama) to 0.72 (War):
- **Rare genres get high thresholds** because balanced weighting inflates their probabilities (the model was told positives matter a lot), and a higher threshold compensates.
- **Drama gets a low one** because half of all movies are dramas. Predicting "Drama" generously costs little precision and gains a lot of recall.

**An honest result:** tuning helped only a little on test, taking macro-F1 from 0.588 at 0.5 to 0.598 tuned.
- **Helped in 9 genres**, most for Western (+0.064) and Family (+0.047).
- **Tied for Thriller**, whose tuned threshold happens to be 0.50.
- **Did slightly worse than 0.5 in 5 genres:** Fantasy (−0.026), and Crime, Romance, Musical and War (by 0.006 or less).

A threshold picked on 5,760 validation movies is itself a noisy estimate, especially for small genres (Fantasy has 347 test movies).

## 8. Evaluation

Open `training/evaluate.py` and step 4 of `training/train.py`.

**Per genre**, using Western as the example (precision 0.775, recall 0.719):
- **Precision:** of the movies we *called* Western, 77.5% really are.
- **Recall:** of the movies that *are* Westerns, we found 71.9%.
- **F1:** the harmonic mean of the two, 2PR / (P + R) = 0.746. The harmonic mean punishes imbalance, so you can't get a high F1 by maxing just one of them.

**Across genres:**
- **Micro-F1 (0.643):** pool all 86,400 (movie, genre) decisions and compute F1 once. Common genres dominate it.
- **Macro-F1 (0.598):** compute F1 per genre and average them. Every genre gets the same weight, so Western counts as much as Drama. That's why we tuned C on it.
- **Samples F1 (0.631):** F1 per movie, averaged over movies. It's closest to how a single user experiences the app.
- **Exact match (17.0%):** the whole genre set is right. It's very strict with 15 labels, so it's always low. Report it anyway.
- **No genre predicted:** 0.7% of test movies.

**Baselines give the numbers meaning.**
- *Always yes* (tag every movie with every genre): recall is 1 and precision equals prevalence, so F1 = 2p / (1 + p). That gives micro 0.257 and macro 0.239. Drama's always-yes F1 is 0.688 just because half of all movies are dramas, which is why Drama's 0.776 is less impressive than Western's 0.746 against a baseline of 0.052.
- *Same model at 0.5*: shows what threshold tuning bought (see §7).

**Bootstrap confidence intervals** (`bootstrap_f1`). The test set is one sample of movies, and a different sample would give a slightly different score. The code resamples the 5,760 test movies *with replacement* 1,000 times and scores each resample. The middle 95% is the interval: micro-F1 0.636–0.649, macro-F1 0.588–0.608. The ±0.01 tells you not to read much into differences smaller than that.

**Why validation (0.615) scores higher than test (0.598).** C and the thresholds were *chosen* because they scored best on validation. Some of that "best" is luck that doesn't carry over, a form of the winner's curse. That's exactly why the test set is kept separate: it's the number you can believe.

**Where the test score is still a little optimistic.** The split is random by movie, and the de-duplication only removes *identical* summaries. Two things slip through:
- **Near-duplicates.** Remakes often share almost the same summary (the Belgian and American versions of *Loft*, or Three Stooges shorts that were remade from stock footage). 8 groups of summaries share their first 30 words, and in 2 of them one copy is in train and another in test (a one-off check).
- **Film series, the bigger effect.** Series shorts are spread across train *and* test. For example, the 202 plots that mention the Stooges split 135 train, 28 validation, 39 test. The model gets test credit for recognising character names like "daffy" and "stooges" (§13), which wouldn't carry over to a new kind of film.

The fix is a *group-aware* split that keeps each series in one set, plus near-duplicate removal (for example, dropping pairs whose word sets overlap by more than 90%).

**Label noise.** The labels come from Freebase and are often incomplete. A thriller tagged only "Drama" turns a correct Thriller prediction into a "false positive". So the real precision is somewhat higher than the measured one, but you can't know by how much without re-labelling.

**Round once.** `evaluate.py` returns full-precision scores, and they're rounded only when written out (`export.rounded`). An earlier version rounded to 4 decimals and then printed 3, and that double rounding shifted a few printed values by 0.001 (micro-F1 showed 0.642 instead of 0.643).

## 9. Export and the reference implementation

Open `training/export.py` and `training/reference.py`.

**`model.json`** holds everything the app needs:

```json
{
  "format_version": 1,
  "vocabulary": ["aadhi", "aakash", "aaron", ...],        // 20,000 words, alphabetical; position = column
  "idf":        [9.4073039, 9.4073039, 7.1386204, ...],   // one per word
  "genres": [
    {"name": "Action", "threshold": 0.55, "intercept": -1.1663603,
     "weights": [0.0581672, -0.044776, 0.6169935, ...]},  // one per word
    ...                                                   // 15 genres
  ],
  "test_metrics": { ... }                                 // shown on the About screen
}
```

That's 300,000 weights plus 20,000 IDF values. Numbers are **rounded to 7 decimal places**, which takes about 9.5 characters per weight instead of 19.2 at full precision. That roughly halves the file: 3.5 MB instead of 6.6 MB (a one-off measurement). Is it safe? `train.py` measures it (below).

**Reproducible.**
- **Same machine:** running `train.py` twice produced a byte-for-byte identical `model.json` (checked by comparing file hashes). The seed is fixed and every package is pinned in `requirements.txt`.
- **Another machine:** a different CPU or operating system can give different low-level maths results in the last digit, so the file could differ in its final decimal places. The export check's 1e-6 tolerance is far larger than that.

**`reference.py` is the specification.** It's the app's prediction algorithm in plain Python, with no NumPy and no scikit-learn, reading `model.json`. The Dart code is a line-by-line translation of it. Two details:
- **Explicit loops instead of `sum()`.** Since Python 3.12, `sum()` of floats uses compensated (extra-precise) addition, while Dart's loop is ordinary addition. The difference is around 1e-16, but the reference should do exactly what Dart does.
- **Numerically stable sigmoid.** `1 / (1 + e^(−z))` computes `e^1000` when z = −1000, which overflows in Python and raises an exception. The code uses `e^z / (1 + e^z)` for negative z, so the exponent is always ≤ 0.

**`verify_export` (in `train.py`).**
1. `train.py` writes the model to a staging file (`model.json.tmp`) and reads it back from disk.
2. It predicts all 5,760 test plots with `reference.py` and compares the results with scikit-learn's. **The largest probability difference is 4.1e-08, and 0 of 86,400 yes/no decisions differ.**
3. If the difference ever exceeds 1e-6, it deletes the staging file and stops, leaving the app's `model.json` untouched. Only a model that passes replaces it.

This catches export bugs, such as weights in the wrong order, a missing intercept or a mismatched vocabulary, before the app ever sees the file.

**The parity fixture (`parity_fixture`).** For 48 texts it writes the tokens, scikit-learn's probability for each genre, the yes/no decision, and the top words (from `reference.py`, because scikit-learn has no explain function). The texts are:
- the first 40 test plots;
- 2 more test plots, so that every genre is predicted at least once (the first 40 had no Musical and no Western);
- 6 edge cases: empty text, punctuation only, only unknown words, ALL CAPS, a repeated word, and a French sentence with an emoji, a dotted `İ` and a Kelvin sign.

## 10. Prediction in Dart

Open `app/lib/src/genre_model.dart` side by side with `training/reference.py`.

| Step | Python (`reference.py`) | Dart (`genre_model.dart`) |
|---|---|---|
| word → column | `self.index = {word: i ...}` | `_index = {for (final (i, word) in ...) word: i}` |
| count known words | `counts[i] = counts.get(i, 0) + 1` | `counts[i] = (counts[i] ?? 0) + 1` |
| tf × idf | `(1 + math.log(count)) * self.idf[i]` | `(1 + math.log(count)) * _idf[i]` |
| L2 normalise | loop, `math.sqrt`, divide | loop, `math.sqrt`, `updateAll` |
| score | `logit = intercept; logit += c` | `var logit = genre.intercept; logit += c` |
| probability | `sigmoid(logit)` | `sigmoid(logit)` |
| top words | sort by (−contribution, word) | `compareTo`, then word order as a tie-break |

Details worth knowing:
- **Map order.** Both Python `dict` and Dart's default `Map` (a `LinkedHashMap`) keep insertion order, so both add up the terms in the same order.
- **`Float64List`** (from `dart:typed_data`) stores 20,000 weights as one packed block of memory (160 KB), not 20,000 separate objects.
- **`format_version` check.** If a future `model.json` changes shape, the app fails loudly instead of predicting nonsense.
- **What the top words mean.** A genre's score (in log-odds) is the intercept plus one term per word, so `weight × value` is *exactly* that word's share of the score, measured from the starting point of an empty text (the intercept). Three caveats worth knowing:
  - **Words depend on each other.** The vector is scaled to length 1, so each word's value depends on the other words in the text. A word's share is therefore not the same as the change you'd see by deleting it.
  - **SHAP uses a different starting point.** SHAP, the standard explanation method, measures each word against the *average* plot, `weight × (value − average value)`. For a linear model that is exact too.
  - **Approximation is for non-linear models.** Approximating methods like LIME and KernelSHAP are what non-linear models need.
- **The empty-text starting point matters.** With no known words, every probability is just `sigmoid(intercept)`. For Drama that's 0.391, which clears its 0.39 threshold, so the raw model answers "Drama" to an empty string. The app handles this (§11). The model itself is left alone, so it still matches Python exactly.

## 11. The Flutter app

**`main.dart`: loading the model.**
- **Reading the file:** `rootBundle.loadString('assets/model.json', cache: false)` reads the bundled file, which is listed in `pubspec.yaml`. `cache: false` stops the asset bundle from keeping a second copy of the 3.5 MB string in memory after it has been parsed.
- **Parsing off the main thread:** `compute(_parseModel, text)` decodes the JSON on a **background isolate** (Dart's version of a thread, with separate memory). Decoding took about 60 ms on a laptop (a one-off measurement), and a phone is several times slower. On the main isolate, the loading spinner would freeze for that long.
- **Creating the future once:** `late final Future<GenreModel> _model = widget.loadModel();` runs a single time. If it were created inside `build()`, every rebuild (whenever a parent rebuilds, on every hot reload) would read and parse the 3.5 MB model again, which is a classic FutureBuilder mistake.
- **Dependency injection:** `GenreApp({this.loadModel = loadBundledModel})` lets the widget tests pass in a model they've already loaded, so they don't depend on the asset bundle.

**`home_page.dart`: the screen.**
- **State:** a `StatefulWidget` holds the text controller and the latest `Prediction`, and disposes the controller in `dispose()`.
- **Predict** calls `model.predict(text)` inside `setState`. It took about 0.8 ms for an average plot on a laptop (a one-off measurement), and a phone is a few times slower but still far below one frame, so it runs on the UI thread.
- **Guarding against no input:** an empty box shows a SnackBar asking for a plot. Text with *no* known words shows a message and no genres, because otherwise the intercepts alone would produce a meaningless "Drama" (§10). A code review found this bug, and two widget tests now pin the fix.
- **Few known words warning:** with 1–7 known words, a warning says the results are shaky, which is honest about garbage in, garbage out.
- **Genre cards** show the probability, a bar with a tick at the threshold (`LayoutBuilder` supplies the width for positioning the tick), and the top words as chips.
- **"Other genres"** is an `ExpansionTile`. `key: ObjectKey(prediction)` gives it a fresh state for each new prediction, so `initiallyExpanded` (open when nothing passed its threshold) takes effect again. Its percentages are rounded *down*. Otherwise 61.7% against a 62% threshold would read "62% (needs 62%)", which looks like it should have passed.
- **`ExcludeSemantics` on the bar** fixes a real bug the widget test caught. The first version gave the progress bar a text value ("100%, threshold 65%"), and Flutter's accessibility layer rejects that because it expects a number. The card already states both numbers in text, so the bar is hidden from screen readers.

**`about_page.dart`** shows how the model works, its test scores and the dataset credit. The scores come from `model.json`'s `test_metrics`, so they always describe the model that is actually shipped. There's no second copy that could drift out of date.

**`example_plots.dart`** has seven plots written for the app that don't describe real films, so there are no copyright questions.

**Offline for real.** The Android release manifest (`android/app/src/main/AndroidManifest.xml`) has no `INTERNET` permission, so a release build *cannot* use the network. Flutter adds that permission only to debug and profile builds, for hot reload. `flutter run` builds debug by default, so this applies to the release build.

**Checked on Android.** The release APK was installed on a Pixel 7 emulator, and `dumpsys package` confirmed it requests no internet permission. With airplane mode on and the network confirmed down:
- the western example predicted Western (100%), Action (86%) and Adventure (82%);
- the About page rendered correctly at phone width.

The screenshots in `docs/` are from that run.

## 12. Tests

Run `pytest training` (31 tests) and `cd app && flutter test` (26 tests).

| Test file | What it proves |
|---|---|
| `training/tests/test_tokenizer.py` | Python passes the shared tokenizer cases |
| `training/tests/test_genres.py` | label mapping: sub-genres map to every parent, format labels are ignored, no duplicates |
| `training/tests/test_reference.py` | TF-IDF and probability match hand calculations, the sigmoid can't overflow, and `reference.py` reproduces scikit-learn on a tiny corpus through the real export function |
| `training/tests/test_model.py` | threshold tuning finds the gap between classes, and the always-yes F1 equals 2p / (1 + p) |
| `app/test/tokenizer_test.dart` | Dart passes the same shared cases |
| `app/test/parity_test.dart` | Dart matches Python on 48 texts: tokens, all 720 probabilities (within 1e-6 of scikit-learn), yes/no decisions, top words |
| `app/test/app_test.dart` | the real widgets (below) |

The widget tests in `app_test.dart` check that:
- typing a western shows a Western card with "cattle";
- a one-word input shows the warning;
- an empty box asks for a plot;
- unknown words show no genres (not even "Drama");
- the About page shows scores and credits.

**Does the parity test actually catch bugs?** A test that can't fail proves nothing, so it was checked by planting one. Changing Dart's `(1 + math.log(count))` to `count` (forgetting sublinear tf) made it fail straight away: probability 0.928 against Python's 0.959. Try it yourself (§15).

**Why the 1e-6 tolerance?** Python's side comes from scikit-learn's full-precision weights, and Dart uses the 7-decimal rounded ones. The measured gap is at most 4e-8, so 1e-6 leaves plenty of margin. Typical bugs, like a wrong formula, a missing intercept or misaligned weights, show up as differences of 1e-2 or more. It wouldn't catch a change *smaller* than 1e-6, though. For example, storing the weights as 32-bit floats would pass. That's fine, because a change that small can't flip a decision in practice: no fixture probability is within 0.001 of its threshold.

## 13. What the model learned, and its biases

`python training/top_words.py` prints each genre's 10 words with the largest positive weight in `model.json`:

| Genre | Top 10 words |
|---|---|
| Action | fight, agent, kill, action, kills, martial, master, battle, rescue, revenge |
| Adventure | rescue, adventure, jungle, captain, yakuza, king, ranch, adventures, captured, named |
| Comedy | comedy, stooges, accidentally, wedding, win, date, girlfriend, eccentric, boyfriend, advice |
| Crime | police, murder, crime, gangster, detective, investigation, kill, robbery, murders, drug |
| Drama | life, relationship, returns, affair, dies, tragedy, cancer, young, emotional, prostitute |
| Family | dog, woody, bugs, children, kids, animals, help, christmas, cartoon, daffy |
| Fantasy | powers, magical, world, adventure, magic, earth, fairy, queen, king, stone |
| Horror | vampire, blood, creature, zombie, horror, body, ghost, strange, killer, werewolf |
| Musical | musical, song, music, singing, band, dance, show, sing, singer, concert |
| Mystery | murder, detective, killer, case, murdered, investigation, mystery, police, investigate, mysterious |
| Romance | love, relationship, she, romance, marry, romantic, feelings, meets, married, date |
| Science Fiction | alien, earth, scientist, planet, future, aliens, robot, dr, brain, machine |
| Thriller | murder, murdered, killing, thriller, detective, suspect, kill, police, killer, killed |
| War | war, german, soldiers, army, nazi, soldier, camp, japanese, captain, battle |
| Western | ranch, outlaw, west, cattle, cowboy, town, horse, saloon, gold, texas |

Mostly sensible, but some entries are **artefacts of the dataset**. The same script counts, across all 38,397 movies, how many plots contain a word and what share of those have the genre:

| Word | Its rank for the genre | Plots containing it | Share with that genre | Share of all movies |
|---|---|--:|--:|--:|
| woody | #2 for Family | 99 | 66% | 9% |
| bugs | #3 for Family | 223 | 65% | 9% |
| daffy | #10 for Family | 107 | 85% | 9% |
| porky | #11 for Family | 102 | 82% | 9% |
| stooges | #2 for Comedy | 202 | 99% | 34% |
| mumbai | #13 for Musical | 336 | 31% | 6% |
| she | #3 for Romance | 18,212 | 27% | 19% |
| comedy | #1 for Comedy | 535 | 79% | 34% |

What those rows show:
- **Cartoon names → Family.** Looney Tunes and Woody Woodpecker shorts are tagged Family, so their characters became "family words". The model learned *which franchise*, not *what kind of story*. It's also a leakage problem (§8), because those series appear in both train and test.
- **stooges → Comedy.** The Three Stooges shorts, the same effect.
- **mumbai → Musical.** Bollywood films are often tagged Musical, so a *place* predicts a *genre*. A plot set in Mumbai with no songs would be nudged toward Musical. It costs real accuracy: Freebase tags 69% of Bollywood films as musicals but only 7% of other Indian films, and 115 of the model's 255 Musical false positives on the test set are Indian films that aren't labelled Bollywood.
- **she → Romance.** "she" is in almost half of all plots, yet it still shifts the odds. Romance plots on Wikipedia centre female characters more often. The model reflects how the summaries are written, so a romance with two male leads would get less of a boost. That's a real bias you can point to.
- **Genre names themselves.** action, adventure, comedy, crime, horror, musical, mystery, romance, thriller and war are all in their own genre's top 10. That's mild label leakage: some summaries say "In this comedy…", so the text sometimes states the answer.

The fixes would be:
- more careful label curation;
- removing character names;
- a group-aware split;
- checking predictions across subgroups.

The general point is that a model learns whatever correlates with the labels, whether or not it's the thing you meant.

## 14. Limitations and next steps

**Limitations**
- **Data:** the labels are noisy and incomplete (Freebase), and the data is from 2012.
- **Leakage:** film series appear in both train and test, so some scores (Family, Comedy) are a bit optimistic (§8).
- **Writing style:** it's trained on English Wikipedia's style. A one-line pitch or a plot written in another language will work poorly, and the app warns when few words are known.
- **Tokenizer:** the ASCII-only tokenizer splits accented words.
- **Independence:** genres are predicted independently, so the model can't use the fact that Romance and Comedy often co-occur.
- **Calibration:** nobody checked it. "70%" isn't guaranteed to mean right 70% of the time, and balanced weighting deliberately distorts probabilities.
- **Word order:** bag of words ignores it, so "the dog bit the man" and "the man bit the dog" look identical.

**What you could do next**
- A **group-aware split** (keep each film series in one set) and near-duplicate removal, for a stricter test score.
- A **calibration curve** per genre, then Platt scaling or isotonic regression if needed.
- **Classifier chains**, which feed earlier genre predictions into later models, to use co-occurrence.
- **Per-genre C**, or an L1 penalty for sparse weights, which would also shrink `model.json`.
- **A smaller model file:** store weights as 16-bit floats, or keep only the top-k weights per genre.
- **Sentence embeddings** (for example a small transformer run with TensorFlow Lite). Probably more accurate, but larger, slower, harder to explain and harder to keep in parity.
- **Error analysis:** read 50 false positives and sort them into "model wrong" and "label missing".

## 15. Exercises

Doing these is the fastest way to own the code.

1. **Break parity on purpose.** In `app/lib/src/genre_model.dart`, change `(1 + math.log(count))` to `count`. Run `flutter test test/parity_test.dart` and read the failure. Undo it.
2. **Break the tokenizer.** Change `{2,}` to `{1,}` in `tokenizer.py` only. Which tests fail, in which language, and why? What changes once you retrain?
3. **Change a setting and retrain.** Set `MIN_DF = 20` in `model.py` and run `python training/train.py`. What happens to the vocabulary size, the file size and macro-F1? (Use `git diff training/results/` to compare.) Then revert and retrain.
4. **Add a genre.** Add "Sports" (672 movies carry the Freebase "Sports" label) to `genres.py` and retrain. How does it score, and what are its top words (`top_words.py`)? Expect two side effects:
   - `test_genres.py` checks for exactly 15 genres, so update that test.
   - Movies that were dropped for having no genre may now be kept. That reshuffles the split, so *every* genre's scores move a little, not just Sports.
5. **Explain one prediction by hand.** Pick a short plot, run it through `reference.py` in a Python shell, and check that `intercept + sum of contributions` gives the logit behind the probability.
6. **Find a failure.** Type a plot the app gets wrong. Look at the top words. Is the model wrong, or is the plot ambiguous?
