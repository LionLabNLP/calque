from __future__ import annotations

import hashlib
import logging
from typing import List, Tuple

import pandas as pd

from data_loading import canonical_language_code

from .schema import (
    JAPANESE_CONTROL_COLUMNS,
    PREFERENCE_COLUMN_BY_SIDE,
    SEMANTIC_KEY_COLUMN,
    SENTENCE_COLUMNS,
    TARGET_LANGUAGE_COLUMN,
    language_info,
    sentence_language,
)

log = logging.getLogger(__name__)

EXCEL_HEADER_ROWS = 1

ID_COLUMNS = [
    "sentence_row_id",
    "source_row_id",
    "semantic_content_id",
    "contrast_id",
    "excel_row",
    "sentence",
    "language",
    "language_iso",
    "target_language",
    "target_language_source",
    "experiment",
    "source_column",
    "sentence_role",
    "side",
    "strategy",
    "side_preference_annotation",
]


def short_hash(*parts: str, n: int = 12) -> str:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def source_row_key(target_language_source, proper_english) -> str:
    return "sr_" + short_hash(str(target_language_source).strip(), str(proper_english))


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def build_sentence_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    source_columns = list(df.columns)
    clash = set(ID_COLUMNS) & set(source_columns)
    if clash:
        raise ValueError(f"source columns clash with derived metadata columns: {sorted(clash)}")
    seen_keys = {}
    records = []
    skipped_blank = 0
    ja_usable = ja_skipped = 0
    has_ja_columns = all(c in df.columns for c in JAPANESE_CONTROL_COLUMNS)

    for idx, row in df.iterrows():
        target_source = row[TARGET_LANGUAGE_COLUMN]
        target = str(target_source).strip()
        target_canonical = canonical_language_code(target)
        semantic = row[SEMANTIC_KEY_COLUMN]
        if _is_missing(semantic):
            warnings.append(f"Excel row {idx + EXCEL_HEADER_ROWS + 1}: empty {SEMANTIC_KEY_COLUMN!r}; row skipped")
            continue
        semantic = str(semantic)
        base_id = source_row_key(target, semantic)
        seen_keys[base_id] = seen_keys.get(base_id, 0) + 1
        source_row_id = base_id if seen_keys[base_id] == 1 else f"{base_id}#{seen_keys[base_id]}"
        if seen_keys[base_id] == 2:
            warnings.append(
                f"duplicate (Target language, Proper English strategy) = ({target!r}, {semantic!r}); "
                "later occurrences get '#n' suffixed IDs"
            )
        semantic_id = "sc_" + short_hash(semantic)

        columns = dict(SENTENCE_COLUMNS)
        if has_ja_columns:
            if all(not _is_missing(row[c]) for c in JAPANESE_CONTROL_COLUMNS):
                columns.update(JAPANESE_CONTROL_COLUMNS)
                ja_usable += 1
            else:
                ja_skipped += 1

        for column, spec in columns.items():
            value = row[column]
            if _is_missing(value):
                skipped_blank += 1
                continue
            lang = sentence_language(column, target_canonical)
            info = language_info(lang)
            pref_col = PREFERENCE_COLUMN_BY_SIDE[spec["side"]]
            rec = {
                "sentence_row_id": f"{source_row_id}:{spec['role']}",
                "source_row_id": source_row_id,
                "semantic_content_id": semantic_id,
                "contrast_id": f"{source_row_id}:{spec['side']}",
                "excel_row": int(idx) + EXCEL_HEADER_ROWS + 1,
                "sentence": value if isinstance(value, str) else str(value),
                "language": lang,
                "language_iso": info.iso if info else lang,
                "target_language": target_canonical,
                "target_language_source": target_source,
                "experiment": spec["experiment"],
                "source_column": column,
                "sentence_role": spec["role"],
                "side": spec["side"],
                "strategy": spec["strategy"],
                "side_preference_annotation": row[pref_col] if pref_col in df.columns else None,
            }
            for c in source_columns:
                rec[c] = row[c]
            records.append(rec)

    if skipped_blank:
        warnings.append(f"{skipped_blank} blank sentence cell(s) produced no metadata row")
    if has_ja_columns:
        log.info("expand: Japanese-control pairs usable=%d skipped=%d (both sentences required)", ja_usable, ja_skipped)
    for w in warnings:
        log.warning("expand: %s", w)
    table = pd.DataFrame.from_records(records, columns=ID_COLUMNS + source_columns)
    return table, warnings
