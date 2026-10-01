import 'dart:math' as math;
import 'dart:typed_data';

import 'tokenizer.dart';

/// How many words to show behind each genre.
const topWordCount = 5;

/// The trained model, running on the phone. This is a line-by-line
/// translation of training/reference.py.
///
/// To predict one plot:
///  1. Split it into tokens and count each word the model knows.
///  2. TF-IDF: each word gets (1 + ln count) * idf, and the vector is then
///     scaled to length 1.
///  3. For each genre, score = intercept + sum of weight[word] * value[word].
///  4. probability = sigmoid(score). The genre is predicted if the
///     probability reaches that genre's threshold.
class GenreModel {
  GenreModel._(this._vocabulary, this._idf, this.genres, this.testMetrics)
      : _index = {for (final (i, word) in _vocabulary.indexed) word: i};

  factory GenreModel.fromJson(Map<String, dynamic> json) {
    if (json['format_version'] != 1) {
      throw FormatException('Unsupported model format: ${json['format_version']}');
    }
    return GenreModel._(
      List<String>.from(json['vocabulary'] as List),
      _toFloat64List(json['idf'] as List),
      [
        for (final genre in json['genres'] as List)
          Genre.fromJson(genre as Map<String, dynamic>),
      ],
      TestMetrics.fromJson(json['test_metrics'] as Map<String, dynamic>),
    );
  }

  final List<String> _vocabulary;
  final Float64List _idf;

  /// word -> column number, e.g. {"aboard": 12, ...}
  final Map<String, int> _index;

  final List<Genre> genres;

  /// Scores measured on the held-out test set, shown on the About screen.
  final TestMetrics testMetrics;

  /// TF-IDF vector as {column: value}. Words the model doesn't know are skipped.
  Map<int, double> vectorize(String text) {
    final counts = <int, int>{};
    for (final token in tokenize(text)) {
      final i = _index[token];
      if (i != null) counts[i] = (counts[i] ?? 0) + 1;
    }

    final vector = {
      for (final MapEntry(key: i, value: count) in counts.entries)
        i: (1 + math.log(count)) * _idf[i],
    };

    // Scale to length 1, so long and short plots are comparable.
    var squared = 0.0;
    for (final value in vector.values) {
      squared += value * value;
    }
    final norm = math.sqrt(squared);
    if (norm > 0) vector.updateAll((i, value) => value / norm);
    return vector;
  }

  Prediction predict(String text) {
    final vector = vectorize(text);
    final results = <GenrePrediction>[];
    for (final genre in genres) {
      // A linear model's score is a sum of one term per word, so each word's
      // share of the score is exact, not an approximation.
      final contributions = {
        for (final MapEntry(key: i, value: value) in vector.entries)
          i: genre.weights[i] * value,
      };
      var logit = genre.intercept;
      for (final contribution in contributions.values) {
        logit += contribution;
      }
      results.add(GenrePrediction(
        genre: genre.name,
        probability: sigmoid(logit),
        threshold: genre.threshold,
        topWords: _topWords(contributions),
      ));
    }
    return Prediction(genres: results, knownWordCount: vector.length);
  }

  /// The words that pushed hardest toward this genre (positive contributions only).
  List<WordContribution> _topWords(Map<int, double> contributions) {
    final positive = [
      for (final MapEntry(key: i, value: c) in contributions.entries)
        if (c > 0) WordContribution(_vocabulary[i], c),
    ];
    // Largest contribution first; ties broken alphabetically, like Python.
    positive.sort((a, b) {
      final byContribution = b.contribution.compareTo(a.contribution);
      return byContribution != 0 ? byContribution : a.word.compareTo(b.word);
    });
    return positive.take(topWordCount).toList();
  }
}

/// 1 / (1 + e^-z), written so that e^x never overflows for large |z|.
double sigmoid(double z) {
  if (z >= 0) return 1 / (1 + math.exp(-z));
  final e = math.exp(z);
  return e / (1 + e);
}

/// One genre's logistic regression: a weight per vocabulary word, an
/// intercept, and the decision threshold tuned on validation data.
class Genre {
  Genre.fromJson(Map<String, dynamic> json)
      : name = json['name'] as String,
        threshold = (json['threshold'] as num).toDouble(),
        intercept = (json['intercept'] as num).toDouble(),
        weights = _toFloat64List(json['weights'] as List);

  final String name;
  final double threshold;
  final double intercept;

  /// A packed array of doubles: 20,000 numbers take 160 KB, rather than
  /// 20,000 separate objects in a `List<double>`.
  final Float64List weights;
}

class Prediction {
  const Prediction({required this.genres, required this.knownWordCount});

  /// One entry per genre, in the model's (alphabetical) order.
  final List<GenrePrediction> genres;

  /// How many different words in the text the model knows. Very few known
  /// words means the prediction is little more than a guess.
  final int knownWordCount;

  /// The predicted genres, most likely first.
  List<GenrePrediction> get predicted =>
      genres.where((g) => g.isPredicted).toList()
        ..sort((a, b) => b.probability.compareTo(a.probability));

  /// The genres that didn't reach their threshold, most likely first.
  List<GenrePrediction> get notPredicted =>
      genres.where((g) => !g.isPredicted).toList()
        ..sort((a, b) => b.probability.compareTo(a.probability));
}

class GenrePrediction {
  const GenrePrediction({
    required this.genre,
    required this.probability,
    required this.threshold,
    required this.topWords,
  });

  final String genre;
  final double probability;
  final double threshold;
  final List<WordContribution> topWords;

  bool get isPredicted => probability >= threshold;
}

class WordContribution {
  const WordContribution(this.word, this.contribution);

  final String word;

  /// How much this word added to the genre's score (in log-odds).
  final double contribution;
}

/// The test-set scores that training/train.py stores in model.json.
class TestMetrics {
  TestMetrics.fromJson(Map<String, dynamic> json)
      : testMovies = json['test_movies'] as int,
        microF1 = (json['micro_f1'] as num).toDouble(),
        macroF1 = (json['macro_f1'] as num).toDouble(),
        macroF1Interval = (
          ((json['macro_f1_95ci'] as List)[0] as num).toDouble(),
          ((json['macro_f1_95ci'] as List)[1] as num).toDouble(),
        ),
        perGenre = {
          for (final MapEntry(:key, :value) in (json['per_genre'] as Map<String, dynamic>).entries)
            key: GenreMetrics.fromJson(value as Map<String, dynamic>),
        };

  final int testMovies;
  final double microF1;
  final double macroF1;

  /// 95% bootstrap confidence interval for macro-F1.
  final (double, double) macroF1Interval;
  final Map<String, GenreMetrics> perGenre;
}

class GenreMetrics {
  GenreMetrics.fromJson(Map<String, dynamic> json)
      : precision = (json['precision'] as num).toDouble(),
        recall = (json['recall'] as num).toDouble(),
        f1 = (json['f1'] as num).toDouble(),
        support = json['support'] as int,
        alwaysYesF1 = (json['always_yes_f1'] as num).toDouble();

  final double precision;
  final double recall;
  final double f1;

  /// How many test movies have this genre.
  final int support;

  /// F1 of the "always say yes" baseline, for comparison.
  final double alwaysYesF1;
}

Float64List _toFloat64List(List<dynamic> values) =>
    Float64List.fromList([for (final v in values) (v as num).toDouble()]);
