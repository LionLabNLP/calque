
from __future__ import annotations

import unicodedata
from typing import Sequence


def _levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def sentence_edit_distance(a: str, b: str) -> int:
    a = unicodedata.normalize("NFC", a or "")
    b = unicodedata.normalize("NFC", b or "")
    return _levenshtein(a, b)


def token_sequence_edit_distance(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    return _levenshtein(list(tokens_a), list(tokens_b))
