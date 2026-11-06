from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .cache import STATUS_FAILED, STATUS_OK, FeatureCache

log = logging.getLogger(__name__)

NAMESPACE = "gloss"
PROMPT_VERSION = "gloss-prompt-v1"
VALIDATOR_VERSION = "gloss-validator-v1"

PROMPT_INSTRUCTIONS = """Please annotate sentences based on the above annotation scheme.

The return format has to be exactly the following:

Sentence: {Sentence}
Object language line: {Object language line}
Interlinear gloss: {Interlinear gloss}

Repeat this format for every sentence.

Do not omit any sentence.
Do not change, correct, normalize, or paraphrase the sentence.
Preserve the exact sentence text and punctuation.
Return the sentences in the same order as the input.
Do not add explanations, comments, markdown, or any other text.

Sentences:
"""


@dataclass(frozen=True)
class GlossSettings:
    model: str = "gpt-5.1"
    batch_size: int = 20
    max_attempts: int = 3
    temperature: Optional[float] = None
    api_retries: int = 4


def load_scheme(path) -> Tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def gloss_config(settings: GlossSettings, scheme_sha256: str) -> Dict[str, object]:
    return {
        "scheme_sha256": scheme_sha256,
        "model": settings.model,
        "temperature": settings.temperature,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
    }


def build_prompt(scheme_text: str, sentences: Sequence[str]) -> str:
    return scheme_text.rstrip("\n") + "\n\n" + PROMPT_INSTRUCTIONS + "\n".join(sentences)


_FIELD_RE = re.compile(r"^\s*(Sentence|Object language line|Interlinear gloss)\s*:\s?(.*?)\s*$")


def parse_response(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    cur: Optional[Dict[str, str]] = None
    for line in text.splitlines():
        m = _FIELD_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key == "Sentence":
            cur = {"Sentence": value}
            records.append(cur)
        elif cur is not None and key not in cur:
            cur[key] = value
    return records


_BOUNDARIES = "-=~"
_INFIX_RE = re.compile(r"<[^<>\s]+>")


def _strip_bracketed(unit: str) -> str:
    return re.sub(r"\[[^\]]*\]|\([^)]*\)", "", unit)


def unit_structure(unit: str, is_gloss: bool) -> Tuple[Optional[Tuple[str, ...]], Optional[int], Optional[str]]:
    u = _strip_bracketed(unit) if is_gloss else unit
    n_infix = len(_INFIX_RE.findall(u))
    rest = _INFIX_RE.sub("", u)
    if "<" in rest or ">" in rest:
        return None, None, f"malformed infix brackets in {unit!r}"
    return tuple(ch for ch in rest if ch in _BOUNDARIES), n_infix, None


def _skeleton(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch for ch in text
        if not ch.isspace() and ch not in "-=~<>" and not unicodedata.category(ch).startswith("P")
    )


def validate_annotation(sentence: str, echoed: Optional[str], oll: Optional[str], gloss: Optional[str]) -> List[str]:
    errors: List[str] = []
    if echoed is None or echoed.strip() != sentence.strip():
        errors.append(f"echoed sentence {echoed!r} != input")
    if not oll or not gloss:
        errors.append("missing object language line or gloss")
        return errors
    if "\n" in oll or "\n" in gloss:
        errors.append("multi-line field")
    ou, gu = oll.split(), gloss.split()
    if len(ou) != len(gu):
        errors.append(f"unit count mismatch: {len(ou)} object-language vs {len(gu)} gloss units")
    else:
        for i, (o, g) in enumerate(zip(ou, gu)):
            ob, oi, oe = unit_structure(o, is_gloss=False)
            gb, gi, ge = unit_structure(g, is_gloss=True)
            if oe or ge:
                errors.append(oe or ge)
            elif ob != gb:
                errors.append(f"unit {i} boundary mismatch: {o!r} {''.join(ob)!r} vs {g!r} {''.join(gb)!r}")
            elif oi != gi:
                errors.append(f"unit {i} infix count mismatch: {o!r} vs {g!r}")
    if _skeleton(oll) != _skeleton(sentence):
        errors.append("object language line does not reproduce the sentence text")
    return errors


Requester = Callable[[str], str]


def openai_requester(settings: GlossSettings) -> Requester:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set; required for --gloss")
    client = OpenAI()

    def request(prompt: str) -> str:
        kwargs = {"model": settings.model, "messages": [{"role": "user", "content": prompt}]}
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        delay = 2.0
        for attempt in range(settings.api_retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception:
                if attempt == settings.api_retries:
                    raise
                log.warning("OpenAI request failed (attempt %d); retrying in %.0fs", attempt + 1, delay, exc_info=True)
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    return request


def generate(
    language: str,
    sentences: Sequence[str],
    scheme_text: str,
    settings: GlossSettings,
    requester: Requester,
    on_batch_done: Callable[[List[Tuple[Tuple[str, str], str, Optional[dict], Optional[str]]]], None],
) -> Dict[str, int]:
    attempts = {s: 0 for s in sentences}
    last_error: Dict[str, str] = {}
    pending = [s for s in sentences if "\n" not in s]
    multi = [s for s in sentences if "\n" in s]
    if multi:
        on_batch_done([((language, s), STATUS_FAILED, None, "sentence contains a newline") for s in multi])
    stats = {"requests": 0, "accepted": 0, "failed": len(multi)}

    while pending:
        batch, pending = pending[: settings.batch_size], pending[settings.batch_size :]
        for s in batch:
            attempts[s] += 1
        stats["requests"] += 1
        try:
            parsed = parse_response(requester(build_prompt(scheme_text, batch)))
            err_all = None
        except Exception as e:
            parsed, err_all = [], f"request error: {type(e).__name__}: {e}"
        by_sentence: Dict[str, Dict[str, str]] = {}
        for rec in parsed:
            by_sentence.setdefault(rec.get("Sentence", "").strip(), rec)

        done, retry = [], []
        for s in batch:
            rec = by_sentence.get(s.strip())
            if rec is None:
                errors = [err_all or "sentence missing from response"]
            else:
                errors = validate_annotation(s, rec.get("Sentence"), rec.get("Object language line"),
                                             rec.get("Interlinear gloss"))
            if not errors:
                done.append(((language, s), STATUS_OK, {
                    "object_language_line": rec["Object language line"],
                    "interlinear_gloss": rec["Interlinear gloss"],
                    "attempts": attempts[s],
                }, None))
                stats["accepted"] += 1
            elif attempts[s] < settings.max_attempts:
                last_error[s] = "; ".join(errors)
                retry.append(s)
            else:
                done.append(((language, s), STATUS_FAILED, {"attempts": attempts[s]}, "; ".join(errors)))
                stats["failed"] += 1
        if done:
            on_batch_done(done)
        pending.extend(retry)
    return stats


GLOSS_COLUMNS = ["Object language line", "Interlinear gloss", "gloss_status", "gloss_error",
                 "gloss_attempts", "gloss_model", "gloss_scheme_sha256", "gloss_config_id"]


def annotate(
    table: pd.DataFrame,
    cache: FeatureCache,
    scheme_path,
    settings: GlossSettings,
    languages: Optional[Sequence[str]],
    requester: Optional[Requester] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    report: Dict[str, object] = {"requested_languages": list(languages or []), "settings": asdict(settings)}
    scheme_path = Path(scheme_path)
    if not scheme_path.exists():
        if languages:
            raise FileNotFoundError(f"gloss annotation scheme not found: {scheme_path}")
        report["note"] = f"scheme {scheme_path} not found; gloss columns left empty"
        out = table.copy()
        for c in GLOSS_COLUMNS:
            out[c] = None
        out["gloss_status"] = "not_computed"
        return out, report

    scheme_text, scheme_sha = load_scheme(scheme_path)
    cid = cache.register_config(NAMESPACE, gloss_config(settings, scheme_sha))
    report.update(config_id=cid, scheme_sha256=scheme_sha)
    keys = list(dict.fromkeys(zip(table["language"], table["sentence"])))
    cached = cache.get_many(NAMESPACE, cid, keys)

    if languages:
        unknown = sorted(set(languages) - set(table["language"]))
        if unknown:
            report["unknown_requested_languages"] = unknown
            log.warning("--gloss languages not present in the data: %s", unknown)
        todo: Dict[str, List[str]] = {}
        for lang, s in keys:
            if lang in languages and not ((lang, s) in cached and cached[(lang, s)].ok):
                todo.setdefault(lang, []).append(s)
        report["to_generate"] = {k: len(v) for k, v in todo.items()}
        if todo:
            requester = requester or openai_requester(settings)
            for lang, sents in sorted(todo.items()):
                log.info("glossing %d %s sentences with %s", len(sents), lang, settings.model)
                report.setdefault("stats", {})[lang] = generate(
                    lang, sents, scheme_text, settings, requester,
                    lambda items: cache.put_many(NAMESPACE, cid, items),
                )
            cached = cache.get_many(NAMESPACE, cid, keys, count_stats=False)

    cols: Dict[str, List[object]] = {c: [] for c in GLOSS_COLUMNS}
    for key in zip(table["language"], table["sentence"]):
        rec = cached.get(key)
        v = rec.value if rec is not None and rec.value else {}
        status = "not_computed" if rec is None else ("ok" if rec.ok else "failed")
        cols["Object language line"].append(v.get("object_language_line") if status == "ok" else None)
        cols["Interlinear gloss"].append(v.get("interlinear_gloss") if status == "ok" else None)
        cols["gloss_status"].append(status)
        cols["gloss_error"].append(rec.error if rec is not None and not rec.ok else None)
        cols["gloss_attempts"].append(v.get("attempts"))
        cols["gloss_model"].append(settings.model if rec is not None else None)
        cols["gloss_scheme_sha256"].append(scheme_sha if rec is not None else None)
        cols["gloss_config_id"].append(cid)
    out = table.copy()
    for c in GLOSS_COLUMNS:
        out[c] = cols[c]
    out["gloss_attempts"] = pd.array(cols["gloss_attempts"], dtype="Int64")
    return out, report
