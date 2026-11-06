
from __future__ import annotations

import re

CLASSIFICATION_LEARNER_SENTENCE_FIELD = "classification_learner_sentence"
CLASSIFICATION_RECONSTRUCTED_SENTENCE_FIELD = "classification_reconstructed_sentence"

_OPEN_TO_CLOSE = {"(": ")", "（": "）"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}

_WHITESPACE_RE = re.compile(r"[ \t]+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def remove_parentheticals(text: str) -> str:
    if not text:
        return text

    stacks: dict[str, list[int]] = {"(": [], "（": []}
    remove_spans: list[tuple[int, int]] = []

    for i, ch in enumerate(text):
        if ch in _OPEN_TO_CLOSE:
            stacks[ch].append(i)
        elif ch in _CLOSE_TO_OPEN:
            stack = stacks[_CLOSE_TO_OPEN[ch]]
            if stack:
                start = stack.pop()
                if not stack:
                    remove_spans.append((start, i + 1))

    if not remove_spans:
        return _normalize_whitespace(text)

    remove_spans.sort()
    kept: list[str] = []
    cursor = 0
    for start, end in remove_spans:
        if start < cursor:
            continue
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    return _normalize_whitespace("".join(kept))


def apply_classification_preprocessing(row: dict) -> dict:
    row = dict(row)
    row[CLASSIFICATION_LEARNER_SENTENCE_FIELD] = remove_parentheticals(row.get("learner_sentence") or "")
    row[CLASSIFICATION_RECONSTRUCTED_SENTENCE_FIELD] = remove_parentheticals(
        row.get("reconstructed_sentence") or ""
    )
    return row
