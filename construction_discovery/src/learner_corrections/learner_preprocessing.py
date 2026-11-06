
from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import ALLOWED_PAIRS, LANGUAGE_ABBR, is_allowed_pair, language_abbr

logger = logging.getLogger(__name__)

TAG_NAMES = ("f-red", "f-blue", "sline")
_TAG_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(t) for t in TAG_NAMES) + r")\](.*?)\[/\1\]",
    re.DOTALL,
)




def extract_tag_contents(text: str) -> list[str]:
    return [m.group(2) for m in _TAG_PATTERN.finditer(text)]


def _is_punct_or_symbol_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("P", "S")


def is_punct_or_symbol_only(s: str) -> bool:
    return all(ch.isspace() or _is_punct_or_symbol_char(ch) for ch in s)


def is_single_ascii_capital(s: str) -> bool:
    stripped = s.strip()
    return len(stripped) == 1 and "A" <= stripped <= "Z"


def strip_punct_and_symbols(s: str) -> str:
    return "".join(ch for ch in s if not _is_punct_or_symbol_char(ch))


def classify_value_level_filter(learner_sentence: str, correction: Any) -> str | None:
    if not isinstance(correction, str):
        return "not_a_string"
    if correction == "":
        return "empty"
    if correction == learner_sentence:
        return "identical_to_learner"
    if correction.lower() == learner_sentence.lower():
        return "case_insensitive_identical_to_learner"

    tag_contents = extract_tag_contents(correction)
    if tag_contents and all(is_punct_or_symbol_only(c) for c in tag_contents):
        return "tag_content_punctuation_or_symbol_only"
    if tag_contents and all(is_single_ascii_capital(c) for c in tag_contents):
        return "tag_content_single_capital_letters_only"

    if strip_punct_and_symbols(learner_sentence) == strip_punct_and_symbols(correction):
        return "identical_after_removing_punctuation"

    return None




@dataclass
class MalformedRecord:
    source_file: str
    line_number: int
    reason: str
    raw_record: Any = None
    sentence_index: int | None = None

    def to_dict(self) -> dict:
        raw = self.raw_record
        try:
            json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = repr(raw)
        return {
            "source_file": self.source_file,
            "line_number": self.line_number,
            "reason": self.reason,
            "sentence_index": self.sentence_index,
            "raw_record": raw,
        }


@dataclass
class ParsedRow:
    source_row_id: str
    source_file: str
    line_number: int
    learning_language: str
    native_language: str
    learner_sentences: list[str]
    corrections_list: list[list[Any]]


def _validate_top_level(
    obj: Any, source_file: str, line_number: int
) -> tuple[ParsedRow | None, MalformedRecord | None]:
    if not isinstance(obj, list):
        return None, MalformedRecord(source_file, line_number, "record_not_a_list", obj)
    if len(obj) < 6:
        return None, MalformedRecord(source_file, line_number, "too_few_fields", obj)

    learning_language, native_language = obj[2], obj[3]
    learner_sentences, corrections_list = obj[4], obj[5]

    if not isinstance(learning_language, str) or not learning_language:
        return None, MalformedRecord(source_file, line_number, "invalid_learning_language", obj)
    if not isinstance(native_language, str) or not native_language:
        return None, MalformedRecord(source_file, line_number, "invalid_native_language", obj)
    if not isinstance(learner_sentences, list):
        return None, MalformedRecord(source_file, line_number, "learner_sentences_not_list", obj)
    if not isinstance(corrections_list, list):
        return None, MalformedRecord(source_file, line_number, "corrections_list_not_list", obj)
    if len(learner_sentences) != len(corrections_list):
        return None, MalformedRecord(
            source_file, line_number, "sentence_correction_length_mismatch", obj
        )

    source_row_id = f"{Path(source_file).stem}:{line_number}"
    return (
        ParsedRow(
            source_row_id=source_row_id,
            source_file=source_file,
            line_number=line_number,
            learning_language=learning_language,
            native_language=native_language,
            learner_sentences=learner_sentences,
            corrections_list=corrections_list,
        ),
        None,
    )


def iter_source_rows(
    input_path: Path,
) -> Iterator[tuple[ParsedRow | None, MalformedRecord | None]]:
    source_file = str(input_path)
    with open(input_path, "r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                yield None, MalformedRecord(
                    source_file, line_number, f"json_decode_error: {exc}", raw_line[:2000]
                )
                continue
            yield _validate_top_level(obj, source_file, line_number)




@dataclass
class PreprocessResult:
    sentence_rows: list[dict] = field(default_factory=list)
    retained_corrections: list[dict] = field(default_factory=list)
    removed_corrections: list[dict] = field(default_factory=list)
    malformed_rows: list[dict] = field(default_factory=list)
    pair_counts_seen: Counter = field(default_factory=Counter)
    pair_counts_allowed_rows: Counter = field(default_factory=Counter)


def preprocess_learner_corpus(
    input_path: Path,
    allowed_pairs: frozenset[tuple[str, str]] = ALLOWED_PAIRS,
) -> PreprocessResult:
    result = PreprocessResult()
    input_path = Path(input_path)

    for parsed, malformed in iter_source_rows(input_path):
        if malformed is not None:
            result.malformed_rows.append(malformed.to_dict())
            continue
        assert parsed is not None

        result.pair_counts_seen[(parsed.learning_language, parsed.native_language)] += 1
        if not is_allowed_pair(parsed.learning_language, parsed.native_language):
            continue

        learning_code = language_abbr(parsed.learning_language)
        native_code = language_abbr(parsed.native_language)
        row_had_retained_sentence = False

        for sent_idx, learner_sentence in enumerate(parsed.learner_sentences):
            sentence_id = f"{parsed.source_row_id}:sent:{sent_idx}"
            corrections_for_sentence = parsed.corrections_list[sent_idx]

            if not isinstance(learner_sentence, str) or not learner_sentence.strip():
                result.malformed_rows.append(
                    MalformedRecord(
                        parsed.source_file,
                        parsed.line_number,
                        "learner_sentence_not_nonempty_string",
                        learner_sentence,
                        sentence_index=sent_idx,
                    ).to_dict()
                )
                continue
            if not isinstance(corrections_for_sentence, list):
                result.malformed_rows.append(
                    MalformedRecord(
                        parsed.source_file,
                        parsed.line_number,
                        "corrections_for_sentence_not_list",
                        corrections_for_sentence,
                        sentence_index=sent_idx,
                    ).to_dict()
                )
                continue

            seen_raw_corrections: dict[str, str] = {}
            retained_for_this_sentence: list[dict] = []

            for corr_idx, raw_correction in enumerate(corrections_for_sentence):
                correction_id = f"{sentence_id}:corr:{corr_idx}"
                reason = classify_value_level_filter(learner_sentence, raw_correction)
                if reason is None and raw_correction in seen_raw_corrections:
                    reason = "duplicate_correction"

                base_row = {
                    "correction_id": correction_id,
                    "sentence_id": sentence_id,
                    "source_row_id": parsed.source_row_id,
                    "learning_language": parsed.learning_language,
                    "native_language": parsed.native_language,
                    "learning_code": learning_code,
                    "native_code": native_code,
                    "learner_sentence": learner_sentence,
                    "original_tagged_correction": raw_correction,
                    "source_line_number": parsed.line_number,
                    "source_sentence_index": sent_idx,
                    "source_correction_index": corr_idx,
                }

                if reason is not None:
                    result.removed_corrections.append({**base_row, "filter_reason": reason})
                    continue

                seen_raw_corrections[raw_correction] = correction_id
                retained_for_this_sentence.append(base_row)

            if retained_for_this_sentence:
                result.retained_corrections.extend(retained_for_this_sentence)
                result.sentence_rows.append(
                    {
                        "sentence_id": sentence_id,
                        "source_row_id": parsed.source_row_id,
                        "learning_language": parsed.learning_language,
                        "native_language": parsed.native_language,
                        "learning_code": learning_code,
                        "native_code": native_code,
                        "learner_sentence": learner_sentence,
                        "source_line_number": parsed.line_number,
                        "source_sentence_index": sent_idx,
                        "num_retained_corrections": len(retained_for_this_sentence),
                    }
                )
                row_had_retained_sentence = True

        if row_had_retained_sentence:
            result.pair_counts_allowed_rows[(parsed.learning_language, parsed.native_language)] += 1

    return result
