from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .expand import EXCEL_HEADER_ROWS, _is_missing
from data_loading import canonical_language_code

from .schema import JAPANESE_CONTROL_COLUMNS, SEMANTIC_KEY_COLUMN, SENTENCE_COLUMNS, sentence_language

PROVENANCE_COLUMNS = ["sentence_row_id", "source_row_id", "semantic_content_id", "excel_row",
                      "sentence", "language", "target_language", "source_column", "sentence_role",
                      "Proper English strategy", "Group", "Temporal interpretation"]

RANGE_COLUMNS = {
    "divergence_score": (0.0, 1.0),
    "divergence_en_realization_score": (0.0, 1.0),
    "divergence_tl_realization_score": (0.0, 1.0),
    "other_language_strategy_divergence_score": (0.0, 1.0),
    "other_language_strategy_divergence_en_realization_score": (0.0, 1.0),
    "other_language_strategy_divergence_tl_realization_score": (0.0, 1.0),
    "wo_score": (0.0, 1.0),
    "en_non11_prop": (0.0, 1.0),
    "tl_non11_prop": (0.0, 1.0),
    "non11_prop_self": (0.0, 1.0),
}

STATUS_VALUE = {
    "freq_status": ("freq_raw", {"ok"}),
    "freq_contrast_status": ("freq_contrast", {"ok"}),
    "divergence_status": ("divergence_score", {"ok"}),
    "other_language_strategy_divergence_status": ("other_language_strategy_divergence_score", {"ok"}),
    "align_status": ("align_n_links", {"ok", "ok_insufficient_links"}),
    "gloss_status": ("Interlinear gloss", {"ok"}),
}


class ValidationError(RuntimeError):
    pass


def validate(table: pd.DataFrame, source: pd.DataFrame) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    def check(name: str, bad: int, detail: str = ""):
        results.append({"check": name, "ok": bad == 0, "n_violations": int(bad), "detail": detail})

    check("unique sentence_row_id", int(table["sentence_row_id"].duplicated().sum()))
    missing_prov = table[PROVENANCE_COLUMNS].isna().any(axis=1) if set(PROVENANCE_COLUMNS) <= set(table.columns) else None
    check("provenance present", int(missing_prov.sum()) if missing_prov is not None else len(table),
          "required: " + ", ".join(PROVENANCE_COLUMNS))

    wrong_lang = sum(
        lang != sentence_language(col, tl)
        for lang, col, tl in table[["language", "source_column", "target_language"]].itertuples(index=False)
    )
    check("sentence language matches source column", wrong_lang)
    en_rows = table[table["language"] == "en"]
    check("target_language retained on English rows", int(en_rows["target_language"].isna().sum()))
    if "target_language_source" in table.columns:
        check(
            "target_language is the canonical form of the source Target language",
            int((table["target_language_source"].map(canonical_language_code) != table["target_language"]).sum()),
        )

    src_cells = {}
    for idx, row in source.iterrows():
        if _is_missing(row[SEMANTIC_KEY_COLUMN]):
            continue
        excel_row = int(idx) + EXCEL_HEADER_ROWS + 1
        columns = list(SENTENCE_COLUMNS)
        if all(c in source.columns for c in JAPANESE_CONTROL_COLUMNS) and all(
            not _is_missing(row[c]) for c in JAPANESE_CONTROL_COLUMNS
        ):
            columns += list(JAPANESE_CONTROL_COLUMNS)
        for col in columns:
            v = row[col]
            if not _is_missing(v):
                src_cells[(excel_row, col)] = v if isinstance(v, str) else str(v)
    meta_cells = {}
    dup = 0
    for excel_row, col, sent in table[["excel_row", "source_column", "sentence"]].itertuples(index=False):
        k = (int(excel_row), col)
        dup += k in meta_cells
        meta_cells[k] = sent
    stale = sum(1 for k, v in meta_cells.items() if src_cells.get(k) != v)
    missing = sum(1 for k in src_cells if k not in meta_cells)
    check("no stale/altered rows (metadata sentence == current source cell)", stale)
    check("every source sentence cell represented", missing)
    check("no duplicated source cell", dup)

    bad_groups = 0
    for _, g in table.groupby("source_row_id"):
        if g["excel_row"].nunique() != 1 or g["sentence_role"].duplicated().any():
            bad_groups += 1
    check("source rows reconstructable from source_row_id", bad_groups)

    for col, (lo, hi) in RANGE_COLUMNS.items():
        if col in table.columns:
            v = pd.to_numeric(table[col], errors="coerce")
            check(f"{col} within [{lo}, {hi}]", int(((v < lo) | (v > hi)).sum()))

    if "freq_raw" in table.columns:
        f = table.loc[table["freq_status"] == "ok", "freq_raw"]
        bad = sum(1 for x in f if pd.isna(x) or int(x) != x or x < 0)
        check("frequency is a non-negative integer when computed", bad)

    for status_col, (value_col, present) in STATUS_VALUE.items():
        if status_col in table.columns and value_col in table.columns:
            has = table[value_col].notna()
            st = table[status_col].isin(present)
            check(f"{status_col}: value present iff computed (NA != 0)", int((has != st).sum()))

    failed = [r for r in results if not r["ok"]]
    if failed:
        lines = "\n".join(f"  - {r['check']}: {r['n_violations']} violation(s)" for r in failed)
        raise ValidationError(f"metadata validation failed:\n{lines}")
    return results
