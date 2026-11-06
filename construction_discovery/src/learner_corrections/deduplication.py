
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", s.strip()))


def _match_stage(a: str, b: str) -> str:
    if a == b:
        return "exact"
    if a.strip() == b.strip():
        return "whitespace_stripped"
    if unicodedata.normalize("NFC", a.strip()) == unicodedata.normalize("NFC", b.strip()):
        return "nfc"
    return "whitespace_normalized"


def _keep_priority_key(row: dict) -> tuple:
    is_suggestion = 0 if row["candidate_type"] == "primary" else 1
    suggestion_index = row["suggestion_index"] if row["suggestion_index"] is not None else -1
    return (
        is_suggestion,
        row["source_correction_index"],
        suggestion_index,
        row["candidate_id"],
    )


def deduplicate_candidates(candidate_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidate_rows:
        key = (row["sentence_id"], normalize_for_dedup(row["reconstructed_sentence"]))
        groups[key].append(row)

    kept_rows: list[dict] = []
    dedup_map_rows: list[dict] = []

    for (sentence_id, _norm_key), rows in groups.items():
        if len(rows) == 1:
            kept_rows.append(rows[0])
            continue

        rows_sorted = sorted(rows, key=_keep_priority_key)
        keeper = rows_sorted[0]
        kept_rows.append(keeper)
        for merged in rows_sorted[1:]:
            dedup_map_rows.append(
                {
                    "kept_candidate_id": keeper["candidate_id"],
                    "merged_candidate_id": merged["candidate_id"],
                    "sentence_id": sentence_id,
                    "match_stage": _match_stage(
                        keeper["reconstructed_sentence"], merged["reconstructed_sentence"]
                    ),
                    "kept_candidate_type": keeper["candidate_type"],
                    "merged_candidate_type": merged["candidate_type"],
                    "kept_correction_id": keeper["correction_id"],
                    "merged_correction_id": merged["correction_id"],
                }
            )

    return kept_rows, dedup_map_rows
