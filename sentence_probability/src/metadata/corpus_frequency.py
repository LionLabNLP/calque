from __future__ import annotations

import itertools
import logging
import math
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from data_loading import canonical_language_code

from .cache import STATUS_OK, FeatureCache, canonical_json
from .normalize import NORMALIZATION_VERSION, normalize_core, normalize_sentence

log = logging.getLogger(__name__)

NAMESPACE = "fineweb2_freq"
CORPUS_NAME = "FineWeb2"
BOUNDARY_VERSION = "boundary-v1"
GIB = 1024 ** 3
TARGET_BYTES = 5 * 1024 ** 3

FINEWEB2_DATASET = "HuggingFaceFW/fineweb-2"
LANGS = {
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "sv": "swe_Latn",
    "zh": "cmn_Hani",
    "ja": "jpn_Jpan",
}
ENGLISH_SOURCE = ("HuggingFaceFW/fineweb", "sample-10BT")
SPLIT = "train"
TEXT_FIELD = "text"

TERMINATORS = set(".!?。！？…")
QUOTES_BRACKETS = set("\"'“”‘’«»「」『』()[]{}（）【】《》〈〉")
CLOSING = set("\"'”’»」』)]}）】》〉")
MAX_LINE_CHARS = 1 << 22


@dataclass(frozen=True)
class FineWeb2Settings:
    budget_bytes: int = TARGET_BYTES
    boundary_mode: str = "sentence"
    split: str = SPLIT
    text_field: str = TEXT_FIELD


def corpus_source(language: str) -> Optional[Tuple[str, str]]:
    code = canonical_language_code(language)
    if code == "en":
        return ENGLISH_SOURCE
    if code in LANGS:
        return (FINEWEB2_DATASET, LANGS[code])
    return None


def frequency_config(s: FineWeb2Settings, language: str) -> Dict[str, object]:
    dataset, config = corpus_source(language)
    return {
        "corpus": {"name": CORPUS_NAME, "dataset": dataset, "config": config, "split": s.split,
                   "text_field": s.text_field, "streaming": True},
        "budget_bytes": s.budget_bytes,
        "budget_rule": "utf-8 bytes of row text; stop before a document that would exceed the budget",
        "normalization": NORMALIZATION_VERSION,
        "boundary": {"mode": s.boundary_mode, "version": BOUNDARY_VERSION},
        "line_unit": "newline-separated lines of each document; matches never span lines",
    }


class _PyAhoCorasick:

    def __init__(self):
        self.goto: List[Dict[str, int]] = [{}]
        self.fail: List[int] = [0]
        self.out: List[List[object]] = [[]]

    def add_word(self, word: str, value):
        node = 0
        for ch in word:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                nxt = len(self.goto)
                self.goto[node][ch] = nxt
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
            node = nxt
        self.out[node].append(value)

    def make_automaton(self):
        from collections import deque

        q = deque(self.goto[0].values())
        while q:
            r = q.popleft()
            for ch, s in self.goto[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[s] = self.goto[f].get(ch, 0) if self.goto[f].get(ch, 0) != s else 0
                self.out[s] = self.out[s] + self.out[self.fail[s]]

    def iter(self, text: str):
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for v in self.out[node]:
                yield i, v


def build_automaton(patterns: Sequence[str], force_python: bool = False):
    try:
        if force_python:
            raise ImportError
        import ahocorasick

        A = ahocorasick.Automaton()
    except ImportError:
        if not force_python:
            log.warning("pyahocorasick not installed; using the slow pure-Python matcher")
        A = _PyAhoCorasick()
    for idx, p in enumerate(patterns):
        A.add_word(p, (idx, len(p)))
    A.make_automaton()
    return A


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_" or unicodedata.category(ch).startswith("M")


def sentence_boundary_ok(text: str, start: int, end: int) -> bool:
    k = start - 1
    while k >= 0 and text[k] == " ":
        k -= 1
    if k >= 0 and text[k] not in TERMINATORS and text[k] not in QUOTES_BRACKETS:
        return False
    n = len(text)
    if end == n:
        return True
    nxt = text[end]
    if nxt == "。":
        return True
    if nxt == ".":
        return end + 1 == n or text[end + 1] == " " or text[end + 1] in CLOSING
    return nxt in CLOSING


def token_boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and _is_word_char(text[start - 1]) and _is_word_char(text[start]):
        return False
    if end < len(text) and _is_word_char(text[end]) and _is_word_char(text[end - 1]):
        return False
    return True


BOUNDARY_RULES: Dict[str, Callable[[str, int, int], bool]] = {
    "sentence": sentence_boundary_ok,
    "token": token_boundary_ok,
}


class LineMatcher:
    def __init__(self, patterns: Sequence[str], boundary_mode: str = "sentence", force_python: bool = False):
        if boundary_mode not in BOUNDARY_RULES:
            raise ValueError(f"unknown boundary mode {boundary_mode!r}")
        if any(not p for p in patterns):
            raise ValueError("empty pattern")
        self.patterns = list(patterns)
        self.max_len = max((len(p) for p in patterns), default=0)
        self.counts = [0] * len(self.patterns)
        self.boundary = BOUNDARY_RULES[boundary_mode]
        self.automaton = build_automaton(self.patterns, force_python=force_python) if patterns else None

    def feed_line(self, raw_line: str):
        if self.automaton is None:
            return
        text = normalize_core(raw_line)
        if not text:
            return
        if len(text) <= MAX_LINE_CHARS:
            self._scan(text, 0, len(text))
            return
        step = MAX_LINE_CHARS
        for lo in range(0, len(text), step):
            self._scan(text, lo, min(lo + step, len(text)))

    def _scan(self, text: str, lo: int, hi: int):
        w_lo = max(0, lo - 1)
        w_hi = min(len(text), hi + self.max_len + 1)
        window = text[w_lo:w_hi]
        for end_idx, (pid, plen) in self.automaton.iter(window):
            start = w_lo + end_idx - plen + 1
            if not (lo <= start < hi):
                continue
            if self.boundary(text, start, start + plen):
                self.counts[pid] += 1


def open_stream(dataset: str, config: str, split: str = SPLIT):
    from datasets import load_dataset

    return load_dataset(dataset, name=config, split=split, streaming=True)


def _skip(rows, n: int):
    if not n:
        return rows
    if hasattr(rows, "skip"):
        return rows.skip(n)
    return itertools.islice(rows, n, None)


@dataclass
class ScanResult:
    language: str
    dataset: str
    config: str
    counts: Dict[str, int]
    text_bytes: int
    docs: int
    budget_bytes: int
    budget_reached: bool
    seconds: float
    resumed: bool = False


def scan_language(
    settings: FineWeb2Settings,
    language: str,
    patterns: Sequence[str],
    cache: Optional[FeatureCache] = None,
    progress_key: Optional[str] = None,
    checkpoint_bytes: int = 512 * 1024 ** 2,
    force_python_matcher: bool = False,
    rows: Optional[Iterable[dict]] = None,
) -> ScanResult:
    t0 = time.time()
    dataset, config = corpus_source(language)
    matcher = LineMatcher(patterns, settings.boundary_mode, force_python=force_python_matcher)

    docs, total, resumed = 0, 0, False
    if cache is not None and progress_key:
        state = cache.load_progress(NAMESPACE, progress_key)
        if (state and state.get("dataset") == dataset and state.get("config") == config
                and state.get("patterns_n") == len(patterns)):
            docs, total = int(state["docs"]), int(state["text_bytes"])
            matcher.counts = list(state["counts"])
            resumed = True
            log.info("[%s] resuming %s scan after %d documents (%.2f GiB)", language, CORPUS_NAME, docs, total / GIB)

    stream = rows if rows is not None else open_stream(dataset, config, settings.split)
    budget_reached = False
    next_ckpt = total + checkpoint_bytes
    for row in _skip(stream, docs):
        text = row[settings.text_field] or ""
        size = len(text.encode("utf-8"))
        if total + size > settings.budget_bytes:
            budget_reached = True
            break
        for line in text.split("\n"):
            matcher.feed_line(line)
        total += size
        docs += 1
        if total >= next_ckpt and cache is not None and progress_key:
            cache.save_progress(NAMESPACE, progress_key, {
                "dataset": dataset, "config": config, "patterns_n": len(patterns),
                "docs": docs, "text_bytes": total, "counts": matcher.counts,
            })
            next_ckpt = total + checkpoint_bytes
            log.info("[%s] %.2f GiB scanned", language, total / GIB)

    return ScanResult(
        language=language, dataset=dataset, config=config,
        counts={pat: c for pat, c in zip(matcher.patterns, matcher.counts)},
        text_bytes=total, docs=docs, budget_bytes=settings.budget_bytes, budget_reached=budget_reached,
        seconds=time.time() - t0, resumed=resumed,
    )


FREQ_COLUMNS = ["freq_corpus", "freq_dataset", "freq_dataset_config", "normalized_sentence", "freq_raw",
                "freq_status", "freq_text_bytes_scanned", "freq_budget_bytes", "freq_budget_reached",
                "freq_config_id"]


def run_frequency_stage(
    table: pd.DataFrame,
    cache: FeatureCache,
    settings: FineWeb2Settings,
    run: bool,
    languages: Optional[Sequence[str]] = None,
    force_python_matcher: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report: Dict[str, object] = {"corpus": CORPUS_NAME, "executed": run, "budget_bytes": settings.budget_bytes,
                                 "languages": {}}
    pairs = table[["language", "sentence"]].drop_duplicates()
    by_lang: Dict[str, List[str]] = {}
    for lang, sent in pairs.itertuples(index=False):
        by_lang.setdefault(lang, []).append(sent)

    cids: Dict[str, str] = {}
    cached_all: Dict[Tuple[str, str], object] = {}
    for lang, sents in sorted(by_lang.items()):
        source = corpus_source(lang)
        if source is None:
            continue
        cid = cache.register_config(NAMESPACE, frequency_config(settings, lang))
        cids[lang] = cid
        cached = cache.get_many(NAMESPACE, cid, [(lang, s) for s in sents])
        missing = [s for s in sents if not ((lang, s) in cached and cached[(lang, s)].ok)]
        lang_report = {"dataset": source[0], "config": source[1], "config_id": cid, "sentences": len(sents),
                       "cached": len(sents) - len(missing), "missing": len(missing)}

        if run and (languages is None or lang in languages) and missing:
            norm = {s: normalize_sentence(s) for s in missing}
            empty = [s for s, n in norm.items() if not n]
            patterns = sorted({n for n in norm.values() if n})
            progress_key = f"{cid}:{_patterns_hash(patterns)}"
            log.info("[%s] streaming %s %s for %d patterns, budget %.2f GiB",
                     lang, source[0], source[1], len(patterns), settings.budget_bytes / GIB)
            try:
                res = scan_language(settings, lang, patterns, cache, progress_key,
                                    force_python_matcher=force_python_matcher)
            except Exception as e:
                log.exception("[%s] %s scan failed", lang, CORPUS_NAME)
                lang_report["error"] = f"{type(e).__name__}: {e}"
                report["languages"][lang] = lang_report
                cached_all.update(cached)
                continue
            scan_meta = {"text_bytes_scanned": res.text_bytes, "docs_scanned": res.docs,
                         "budget_bytes": res.budget_bytes, "budget_reached": res.budget_reached,
                         "dataset": res.dataset, "config": res.config}
            items = [((lang, s), STATUS_OK, dict(scan_meta, count=res.counts[norm[s]], normalized=norm[s]), None)
                     for s in missing if norm[s]]
            items += [((lang, s), "failed", None, "empty normalized sentence") for s in empty]
            cache.put_many(NAMESPACE, cid, items)
            cache.clear_progress(NAMESPACE, progress_key)
            cached = cache.get_many(NAMESPACE, cid, [(lang, s) for s in sents], count_stats=False)
            lang_report.update(scan_meta, seconds=round(res.seconds, 1), resumed=res.resumed)
        report["languages"][lang] = lang_report
        cached_all.update(cached)

    cols: Dict[str, List[object]] = {c: [] for c in FREQ_COLUMNS}
    for lang, sent in table[["language", "sentence"]].itertuples(index=False):
        source = corpus_source(lang)
        rec = cached_all.get((lang, sent))
        cols["freq_corpus"].append(CORPUS_NAME if source else None)
        cols["freq_dataset"].append(source[0] if source else None)
        cols["freq_dataset_config"].append(source[1] if source else None)
        cols["normalized_sentence"].append(normalize_sentence(sent))
        cols["freq_config_id"].append(cids.get(lang))
        cols["freq_budget_bytes"].append(settings.budget_bytes)
        if source is None:
            status, v = "unsupported_language", None
        elif rec is None:
            status, v = "not_computed", None
        elif not rec.ok:
            status, v = "failed", None
        else:
            status, v = "ok", rec.value
        cols["freq_status"].append(status)
        cols["freq_raw"].append(v["count"] if v else None)
        cols["freq_text_bytes_scanned"].append(v["text_bytes_scanned"] if v else None)
        cols["freq_budget_reached"].append(v["budget_reached"] if v else None)

    out = table.copy()
    for c in FREQ_COLUMNS:
        if c in ("freq_raw", "freq_text_bytes_scanned", "freq_budget_bytes"):
            out[c] = pd.array(cols[c], dtype="Int64")
        else:
            out[c] = cols[c]
    return add_frequency_contrasts(out), report


CONTRAST_COLUMNS = ["freq_contrast", "freq_contrast_status", "tl_freq_contrast", "en_freq_contrast"]


def frequency_contrast(freq_proper, freq_transferred) -> Optional[float]:
    if freq_proper is None or freq_transferred is None or pd.isna(freq_proper) or pd.isna(freq_transferred):
        return None
    return math.log1p(float(freq_proper)) - math.log1p(float(freq_transferred))


def add_frequency_contrasts(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    ok = out["freq_status"] == "ok"
    freq = out["freq_raw"].where(ok)
    by_contrast: Dict[str, Dict[str, object]] = {}
    for cid, strategy, f in zip(out["contrast_id"], out["strategy"], freq):
        by_contrast.setdefault(cid, {})[strategy] = None if pd.isna(f) else int(f)
    values: Dict[str, Tuple[Optional[float], str]] = {}
    for cid, d in by_contrast.items():
        if "proper" not in d or "transferred" not in d:
            values[cid] = (None, "missing_pair_sentence")
            continue
        F = frequency_contrast(d["proper"], d["transferred"])
        values[cid] = (F, "ok" if F is not None else "frequency_unavailable")
    out["freq_contrast"] = pd.array([values[c][0] for c in out["contrast_id"]], dtype="Float64")
    out["freq_contrast_status"] = [values[c][1] for c in out["contrast_id"]]
    for side in ("tl", "en"):
        side_val = {}
        for sid, cid in zip(out["source_row_id"], out["contrast_id"]):
            if cid.endswith(f":{side}"):
                side_val[sid] = values[cid][0]
        out[f"{side}_freq_contrast"] = pd.array([side_val.get(sid) for sid in out["source_row_id"]], dtype="Float64")
    return out


def _patterns_hash(patterns: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(list(patterns)).encode("utf-8")).hexdigest()[:16]
