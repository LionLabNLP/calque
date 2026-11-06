#!/usr/bin/env python3
"""Component C CLI: export annotated candidates from the SQLite Filtering database.

Example:
    python scripts/export_reviews.py \\
        --input data/processed/toy/classified/all_classified.csv \\
        --database data/processed/toy/review/filtering.sqlite \\
        --output-dir data/processed/toy/review/export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from learner_corrections.io import atomic_write_dataframe, setup_logging
from learner_corrections.review_app import database as db

EXPORT_COLUMNS = [
    "candidate_id",
    "sentence_id",
    "source_row_id",
    "correction_id",
    "learning_language",
    "native_language",
    "learning_code",
    "native_code",
    "candidate_type",
    "suggestion_index",
    "learner_sentence",
    "reconstructed_sentence",
    "original_tagged_correction",
    "suggestion_evidence",
    "api_model",
    "classification",
    "state",
    "keep",
    "low_confidence",
    "functions",
    "revision",
    "created_at",
    "updated_at",
    "construction_group_id",
    "construction_group_rank",
    "construction_group_size",
    "construction_group_type",
    "construction_operation_pattern",
    "construction_group_signature",
    "construction_group_key",
    "construction_edit_count",
]

GROUP_SUMMARY_EXPORT_COLUMNS = [
    "construction_group_id",
    "construction_group_rank",
    "construction_group_type",
    "construction_operation_pattern",
    "construction_group_signature",
    "construction_group_size",
    "reviewed_count",
    "unreviewed_count",
    "kept_count",
    "rejected_count",
    "low_confidence_count",
    "learning_languages",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="all_classified.csv (used to (re)import candidates if not already in the database)")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-new-input", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _build_export_rows(conn) -> list[dict]:
    rows = db.list_all(conn)
    export_rows = []
    for row in rows:
        functions = db.get_functions(conn, row["candidate_id"])
        functions_json = json.dumps(
            [
                {
                    "correct_construction": f["correct_construction"],
                    "learner_construction": f["learner_construction"],
                }
                for f in functions
            ],
            ensure_ascii=False,
        )
        export_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "sentence_id": row["sentence_id"],
                "source_row_id": row["source_row_id"],
                "correction_id": row["correction_id"],
                "learning_language": row["learning_language"],
                "native_language": row["native_language"],
                "learning_code": row["learning_code"],
                "native_code": row["native_code"],
                "candidate_type": row["candidate_type"],
                "suggestion_index": row["suggestion_index"],
                "learner_sentence": row["learner_sentence"],
                "reconstructed_sentence": row["reconstructed_sentence"],
                "original_tagged_correction": row["original_tagged_correction"],
                "suggestion_evidence": row["suggestion_evidence"],
                "api_model": row["api_model"],
                "classification": row["classification"],
                "state": row["state"],
                "keep": bool(row["keep"]),
                "low_confidence": bool(row["low_confidence"]),
                "functions": functions_json,
                "revision": row["revision"] or 0,
                "created_at": row["annotation_created_at"] or "",
                "updated_at": row["annotation_updated_at"] or "",
                "construction_group_id": row["construction_group_id"] or "",
                "construction_group_rank": row["construction_group_rank"],
                "construction_group_size": row["construction_group_size"],
                "construction_group_type": row["construction_group_type"] or "",
                "construction_operation_pattern": row["construction_operation_pattern"] or "",
                "construction_group_signature": row["construction_group_signature"] or "",
                "construction_group_key": row["construction_group_key"] or "",
                "construction_edit_count": row["construction_edit_count"],
            }
        )
    return export_rows


def _build_group_summary_rows(rows: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        group_id = row["construction_group_id"]
        if not group_id:
            continue
        group = groups.setdefault(
            group_id,
            {
                "construction_group_id": group_id,
                "construction_group_rank": row["construction_group_rank"],
                "construction_group_type": row["construction_group_type"] or "",
                "construction_operation_pattern": row["construction_operation_pattern"] or "",
                "construction_group_signature": row["construction_group_signature"] or "",
                "construction_group_size": row["construction_group_size"],
                "reviewed_count": 0,
                "unreviewed_count": 0,
                "kept_count": 0,
                "rejected_count": 0,
                "low_confidence_count": 0,
                "languages": set(),
            },
        )
        state = row["state"]
        if state == "filtering":
            group["unreviewed_count"] += 1
        else:
            group["reviewed_count"] += 1
        if state == "kept":
            group["kept_count"] += 1
        elif state == "discarded":
            group["rejected_count"] += 1
        if row["low_confidence"]:
            group["low_confidence_count"] += 1
        language = row["learning_language"]
        if language:
            group["languages"].add(language)

    summary_rows = [
        {
            **{k: v for k, v in group.items() if k != "languages"},
            "learning_languages": ", ".join(sorted(group["languages"])),
        }
        for group in groups.values()
    ]
    summary_rows.sort(
        key=lambda g: g["construction_group_rank"] if g["construction_group_rank"] not in (None, "") else 10**9
    )
    return summary_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs" / "export_reviews.log")

    with db.get_connection(args.database) as conn:
        db.init_db(conn)
        df_in = db.load_candidates_csv(args.input)
        db.import_candidates(conn, df_in, args.input, allow_new_input=args.allow_new_input)

        export_rows = _build_export_rows(conn)
        all_candidate_rows = db.list_all(conn)

    df = pd.DataFrame(export_rows, columns=EXPORT_COLUMNS)

    all_df = df
    kept_df = df[df["state"] == "kept"]
    discarded_df = df[df["state"] == "discarded"]
    filtering_df = df[df["state"] == "filtering"]

    atomic_write_dataframe(all_df, output_dir, "all", overwrite=args.overwrite)
    atomic_write_dataframe(kept_df, output_dir, "kept", overwrite=args.overwrite)
    atomic_write_dataframe(discarded_df, output_dir, "discarded", overwrite=args.overwrite)
    atomic_write_dataframe(filtering_df, output_dir, "filtering", overwrite=args.overwrite)

    group_summary_rows = _build_group_summary_rows(all_candidate_rows)
    group_df = pd.DataFrame(group_summary_rows, columns=GROUP_SUMMARY_EXPORT_COLUMNS)
    atomic_write_dataframe(group_df, output_dir, "construction_groups_review_summary", overwrite=args.overwrite)

    print(
        f"Exported {len(all_df)} total ({len(kept_df)} kept, {len(discarded_df)} discarded, "
        f"{len(filtering_df)} still in filtering) to {output_dir}"
    )
    print(f"Exported {len(group_df)} construction group summary row(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
