import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:movie_genre_predictor/src/genre_model.dart';
import 'package:movie_genre_predictor/src/tokenizer.dart';

/// Checks that the app predicts exactly what the Python model predicts.
///
/// training/train.py writes test/fixtures/parity_cases.json: 42 real test-set
/// plots (enough that every genre is predicted at least once) plus 6 odd
/// inputs. For each one it stores the tokens, the
/// probability scikit-learn gives each genre, the yes/no decision, and the
/// top words. Here the same texts go through the Dart code and must give the
/// same answers.
void main() {
  late GenreModel model;
  late List<dynamic> cases;

  setUpAll(() {
    model = GenreModel.fromJson(
        jsonDecode(File('assets/model.json').readAsStringSync()) as Map<String, dynamic>);
    cases = jsonDecode(File('test/fixtures/parity_cases.json').readAsStringSync())['cases'] as List;
  });

  test('the fixture has cases where each genre is predicted', () {
    // Guards against a fixture so easy that it tests nothing.
    final predictedGenres = {
      for (final c in cases)
        for (final g in c['genres'] as List)
          if (g['predicted'] == true) g['name'],
    };
    expect(predictedGenres, hasLength(model.genres.length));
  });

  test('tokens, probabilities, decisions and top words match Python', () {
    for (final (i, c) in cases.indexed) {
      final text = c['text'] as String;
      final reason = 'case $i: "${text.length > 60 ? '${text.substring(0, 60)}...' : text}"';

      expect(tokenize(text), c['tokens'], reason: reason);

      final prediction = model.predict(text);
      for (final (g, expected) in (c['genres'] as List).indexed) {
        final actual = prediction.genres[g];
        final where = '$reason, ${expected['name']}';
        expect(actual.genre, expected['name'], reason: where);
        // model.json stores weights to 7 decimal places, so allow a tiny difference.
        expect(actual.probability, closeTo(expected['probability'] as double, 1e-6), reason: where);
        expect(actual.isPredicted, expected['predicted'], reason: where);

        final expectedWords = expected['top_words'] as List;
        expect([for (final w in actual.topWords) w.word], [for (final w in expectedWords) w[0]],
            reason: where);
        for (final (k, word) in actual.topWords.indexed) {
          expect(word.contribution, closeTo(expectedWords[k][1] as double, 1e-9), reason: where);
        }
      }
    }
  });
}
