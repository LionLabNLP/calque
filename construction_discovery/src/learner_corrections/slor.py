
import json
import math
from collections import Counter
from pathlib import Path

FLORES_LANGS = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "sv": "swe_Latn",
    "se": "swe_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
}

DEFAULT_ALPHA = 1.0
DEFAULT_SPLIT = "dev"

_UNIGRAM_CACHE = {}


def load_flores_sentences(lang_code, split=DEFAULT_SPLIT):
    path = Path(__file__).resolve().parents[2] / "data" / "flores200_dataset" / split / f"{lang_code}.{split}"
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _disk_cache_path(cache_key):
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "slor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in cache_key)
    return cache_dir / f"{safe_key}.json"


def _tokenizer_cache_key(tokenizer):
    name = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    return str(name)


def get_unigram_token_counts(tokenizer, lang_code, split=DEFAULT_SPLIT, use_disk_cache=True):
    if lang_code not in FLORES_LANGS:
        raise ValueError(f"No FLORES language mapping for lang code: {lang_code!r}")

    tokenizer_key = _tokenizer_cache_key(tokenizer)
    flores_code = FLORES_LANGS[lang_code]
    cache_key = f"{tokenizer_key}__{flores_code}__{split}"

    if cache_key in _UNIGRAM_CACHE:
        return _UNIGRAM_CACHE[cache_key]

    disk_path = _disk_cache_path(cache_key)
    if use_disk_cache and disk_path.exists():
        with open(disk_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        result = {
            "token_counts": Counter({int(k): v for k, v in cached["token_counts"].items()}),
            "total_tokens": cached["total_tokens"],
            "n_sentences": cached["n_sentences"],
            "tokenizer_key": tokenizer_key,
            "flores_code": flores_code,
            "split": split,
        }
        _UNIGRAM_CACHE[cache_key] = result
        return result

    sentences = load_flores_sentences(flores_code, split=split)

    token_counts = Counter()
    total_tokens = 0
    for sentence in sentences:
        token_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
        token_counts.update(token_ids)
        total_tokens += len(token_ids)

    result = {
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "n_sentences": len(sentences),
        "tokenizer_key": tokenizer_key,
        "flores_code": flores_code,
        "split": split,
    }
    _UNIGRAM_CACHE[cache_key] = result

    if use_disk_cache:
        with open(disk_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "token_counts": {str(k): v for k, v in token_counts.items()},
                    "total_tokens": total_tokens,
                    "n_sentences": len(sentences),
                    "tokenizer_key": tokenizer_key,
                    "flores_code": flores_code,
                    "split": split,
                },
                f,
            )

    return result


def smoothed_unigram_logprob(token_id, token_counts, total_tokens, vocab_size, alpha=DEFAULT_ALPHA):
    count = token_counts.get(token_id, 0)
    prob = (count + alpha) / (total_tokens + alpha * vocab_size)
    return math.log(prob)


def sentence_unigram_logprob(token_ids, token_counts, total_tokens, vocab_size, alpha=DEFAULT_ALPHA):
    return sum(
        smoothed_unigram_logprob(tid, token_counts, total_tokens, vocab_size, alpha=alpha)
        for tid in token_ids
    )


def compute_slor(sentence_logprob, token_ids, token_counts, total_tokens, vocab_size, alpha=DEFAULT_ALPHA):
    n = len(token_ids)
    if n == 0:
        return float("nan")
    unigram_logprob = sentence_unigram_logprob(token_ids, token_counts, total_tokens, vocab_size, alpha=alpha)
    return (sentence_logprob - unigram_logprob) / n
