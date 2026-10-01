/// The tokenizer. An exact copy of training/tokenizer.py; that file explains
/// why it is so strict.
///
/// A token is a run of two or more ASCII letters (A-Z, a-z), lowercased.
/// Everything else separates tokens.
///
///     tokenize("Spider-Man's 2nd fight!")  // [spider, man, nd, fight]
///
/// A match only ever contains ASCII letters, so toLowerCase() can't do
/// anything surprising, and Dart's UTF-16 strings behave the same as
/// Python's code points: a non-ASCII character is never part of a match
/// either way.
final _token = RegExp(r'[A-Za-z]{2,}');

List<String> tokenize(String text) =>
    [for (final match in _token.allMatches(text)) match[0]!.toLowerCase()];
