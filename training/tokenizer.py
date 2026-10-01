"""The tokenizer. The Flutter app has an exact copy in app/lib/src/tokenizer.dart.

A token is a run of two or more ASCII letters (A-Z, a-z), lowercased.
Everything else separates tokens: spaces, punctuation, digits, apostrophes,
accented letters and emoji.

    "Spider-Man's 2nd fight!"  ->  ["spider", "man", "nd", "fight"]

Why so strict? The model is trained here and runs in Dart on the phone. If
the two languages split text differently, the phone would look up
different words than the model learned, and its predictions would quietly
drift from the ones we measured. General Unicode rules are where Python and
Dart are most likely to disagree:

  * In Python, "\N{KELVIN SIGN}".lower() gives an ASCII "k", and
    "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}".lower() gives two
    characters. Dart has its own case tables.
  * The regex shortcut for "word character" means different things in
    Python's re and Dart's RegExp.

Matching only [A-Za-z] avoids all of that. A match contains only ASCII
letters, and lowercasing ASCII works the same in every language. The cost is
that accented words get split: "café" -> ["caf"]. That is rare in
English-language Wikipedia plots and hardly affects the model.

shared/tokenizer_cases.json lists inputs with the expected tokens, and both
languages' tests check against it.
"""

import re

_TOKEN = re.compile(r"[A-Za-z]{2,}")


def tokenize(text: str) -> list[str]:
    return [match.lower() for match in _TOKEN.findall(text)]
