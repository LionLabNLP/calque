from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data_loading import (
    EXPERIMENT_JAPANESE_CONTROL,
    EXPERIMENT_STANDARD,
    JAPANESE_CONTROL_ANNOTATION_COLUMN,
    JAPANESE_CONTROL_COMPARISON,
    SENTENCE_COLUMN_LANGUAGE_MAP,
    canonical_language_code,
)

log = logging.getLogger(__name__)

TARGET_LANGUAGE_COLUMN = "Target language"
SEMANTIC_KEY_COLUMN = "Proper English strategy"
ENGLISH_PREFERENCE_COLUMN = "English preference annotation"
TARGET_PREFERENCE_COLUMN = "Target language preference annotation"

COL_EN_PROPER = "Proper English strategy"
COL_EN_TRANSFERRED = "Literal English translation of target language strategy"
COL_TL_PROPER = "Proper strategy in target language"
COL_TL_TRANSFERRED = "Literal translation of Proper English into target language"
COL_JA_PROPER, COL_JA_TRANSFERRED = JAPANESE_CONTROL_COMPARISON

SENTENCE_COLUMNS: Dict[str, Dict[str, str]] = {
    COL_EN_PROPER: {"role": "en_proper", "side": "en", "strategy": "proper", "experiment": EXPERIMENT_STANDARD},
    COL_EN_TRANSFERRED: {"role": "en_transferred", "side": "en", "strategy": "transferred", "experiment": EXPERIMENT_STANDARD},
    COL_TL_PROPER: {"role": "tl_proper", "side": "tl", "strategy": "proper", "experiment": EXPERIMENT_STANDARD},
    COL_TL_TRANSFERRED: {"role": "tl_transferred", "side": "tl", "strategy": "transferred", "experiment": EXPERIMENT_STANDARD},
}
JAPANESE_CONTROL_COLUMNS: Dict[str, Dict[str, str]] = {
    COL_JA_PROPER: {"role": "ja_control_proper", "side": "ja_control", "strategy": "proper",
                    "experiment": EXPERIMENT_JAPANESE_CONTROL},
    COL_JA_TRANSFERRED: {"role": "ja_control_transferred", "side": "ja_control", "strategy": "transferred",
                         "experiment": EXPERIMENT_JAPANESE_CONTROL},
}
ALL_SENTENCE_COLUMNS: Dict[str, Dict[str, str]] = {**SENTENCE_COLUMNS, **JAPANESE_CONTROL_COLUMNS}
PREFERENCE_COLUMN_BY_SIDE = {
    "en": "English preference annotation",
    "tl": "Target language preference annotation",
    "ja_control": JAPANESE_CONTROL_ANNOTATION_COLUMN,
}

REQUIRED_COLUMNS = [
    TARGET_LANGUAGE_COLUMN,
    *SENTENCE_COLUMNS.keys(),
    "Group",
    "Subgroup",
    "Temporal interpretation",
]
EXPECTED_COLUMNS = [
    ENGLISH_PREFERENCE_COLUMN, TARGET_PREFERENCE_COLUMN, "Subject",
    COL_JA_PROPER, COL_JA_TRANSFERRED, JAPANESE_CONTROL_ANNOTATION_COLUMN,
]


@dataclass(frozen=True)
class LanguageInfo:
    name: str
    iso: str
    stanza: Optional[str]


LANGUAGES: Dict[str, LanguageInfo] = {
    "en": LanguageInfo("English", "en", "en"),
    "de": LanguageInfo("German", "de", "de"),
    "fr": LanguageInfo("French", "fr", "fr"),
    "pt": LanguageInfo("Portuguese", "pt", "pt"),
    "zh": LanguageInfo("Chinese", "zh", "zh-hans"),
    "ja": LanguageInfo("Japanese", "ja", "ja"),
    "sv": LanguageInfo("Swedish", "sv", "sv"),
}


def language_info(code: str) -> Optional[LanguageInfo]:
    return LANGUAGES.get(canonical_language_code(code))


def sentence_language(column: str, target_language: str) -> str:
    spec = SENTENCE_COLUMN_LANGUAGE_MAP[column]["language"]
    return canonical_language_code(target_language) if spec == "target" else spec


@dataclass
class SourceData:
    path: Path
    sheet: str
    df: pd.DataFrame
    file_sha256: str
    warnings: List[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_sheet(path: Path, sheet) -> pd.DataFrame:
    return pd.read_excel(
        path, sheet_name=sheet, dtype=object, keep_default_na=False, na_values=[""]
    )


def discover_sheet(path: Path, sheet: Optional[str] = None) -> str:
    xls = pd.ExcelFile(path)
    if sheet is not None:
        if sheet not in xls.sheet_names:
            raise ValueError(f"Sheet {sheet!r} not found in {path}; available: {xls.sheet_names}")
        return sheet
    matches = []
    for name in xls.sheet_names:
        cols = set(pd.read_excel(xls, sheet_name=name, nrows=0).columns)
        if all(c in cols for c in REQUIRED_COLUMNS):
            matches.append(name)
    if not matches:
        raise ValueError(
            f"No worksheet in {path} contains the required columns {REQUIRED_COLUMNS}. "
            f"Sheets: {xls.sheet_names}"
        )
    if len(matches) > 1:
        raise ValueError(f"Several worksheets match the schema ({matches}); pass --sheet explicitly.")
    return matches[0]


def load_source(path, sheet: Optional[str] = None) -> SourceData:
    path = Path(path)
    sheet_name = discover_sheet(path, sheet)
    df = _read_sheet(path, sheet_name)
    warnings: List[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            warnings.append(f"expected column {c!r} is absent")
    dup_cols = [c for c in df.columns if str(c).endswith(".1")]
    if dup_cols:
        warnings.append(f"possible duplicated header names (pandas-suffixed): {dup_cols}")

    empty = df.isna().all(axis=1)
    if empty.any():
        warnings.append(f"ignored {int(empty.sum())} fully empty row(s)")
    df = df.loc[~empty]

    langs = df[TARGET_LANGUAGE_COLUMN]
    if langs.isna().any():
        raise ValueError(f"{int(langs.isna().sum())} row(s) have no {TARGET_LANGUAGE_COLUMN!r}")
    unknown = sorted({str(v) for v in langs if canonical_language_code(v) not in LANGUAGES})
    if unknown:
        warnings.append(
            f"unknown target-language code(s) {unknown}: rows are kept, but Stanza/FineWeb2 "
            "stages will mark them unsupported"
        )

    for w in warnings:
        log.warning("source schema: %s", w)
    return SourceData(path=path, sheet=sheet_name, df=df, file_sha256=file_sha256(path), warnings=warnings)
