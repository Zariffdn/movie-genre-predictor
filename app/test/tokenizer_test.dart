import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:movie_genre_predictor/src/tokenizer.dart';

/// The same cases are checked by training/tests/test_tokenizer.py, so if both
/// pass, the two tokenizers agree on every one of them.
void main() {
  final shared = jsonDecode(File('../shared/tokenizer_cases.json').readAsStringSync());

  for (final testCase in shared['cases'] as List) {
    test(testCase['note'], () {
      expect(tokenize(testCase['text'] as String), testCase['tokens']);
    });
  }
}
