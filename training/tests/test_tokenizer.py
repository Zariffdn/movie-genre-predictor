import json
from pathlib import Path

import pytest

from tokenizer import tokenize

# The same file is checked by app/test/tokenizer_test.dart.
SHARED_CASES = Path(__file__).resolve().parents[2] / "shared" / "tokenizer_cases.json"
CASES = json.loads(SHARED_CASES.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[case["note"] for case in CASES])
def test_shared_cases(case):
    assert tokenize(case["text"]) == case["tokens"]
