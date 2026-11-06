from __future__ import annotations

import re
import unicodedata
from typing import List

NORMALIZATION_VERSION = "norm-v1"

SENTENCE_FINAL_STOPS = (".", "。")

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_core(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_sentence(text: str) -> str:
    text = normalize_core(text)
    if text.endswith(SENTENCE_FINAL_STOPS):
        text = text[:-1]
    return text.strip()


DIVERGENCE_TOKENIZER_VERSION = "tok-v1"
_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def divergence_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize_sentence(text))
