#!/usr/bin/env python3
"""Component B CLI: Stanza-parse learner sentences and their reconstructed corrections, align
them, and classify each pair into one of: lexical, morphological, pure, constructional+others,
mixed, word_order, unchanged, invalid.

Example (full mode):
    python scripts/classify_errors.py \\
        --input data/processed/full/prepared_candidates.parquet \\
        --output-dir data/processed/full/classified \\
        --stanza-resources-dir ~/.cache/stanza_resources

Example (toy mode):
    python scripts/classify_errors.py \\
        --input data/processed/toy/prepared_candidates.parquet \\
        --output-dir data/processed/toy/classified
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from learner_corrections.config import ALLOWED_PAIRS, LANGUAGE_ABBR, load_config
from learner_corrections.construction_grouping import (
    assign_construction_groups,
    format_construction_group_report,
    summarize_construction_groups,
)
from learner_corrections.error_classification import (
    FUNCTION_WORD_UPOS,
    classify_candidates,
    load_classified_rows,
)
from learner_corrections.io import atomic_write_dataframe, guard_overwrite, setup_logging
from learner_corrections.parenthetical import (
    CLASSIFICATION_LEARNER_SENTENCE_FIELD,
    CLASSIFICATION_RECONSTRUCTED_SENTENCE_FIELD,
    remove_parentheticals,
)
from learner_corrections.reproducibility import RunSummary
from learner_corrections.stanza_parsing import StanzaPipelineManager

logger = logging.getLogger("classify_errors")

DEFAULT_STANZA_RESOURCES_DIR = Path("~/.cache/stanza_resources").expanduser()

JSON_LIST_FIELDS = (
    "function_word_changes",
    "dependency_arc_changes",
    "lemma_changes",
    "morphological_feature_changes",
    "orthographic_changes",
    "token_alignment",
)

CLASSIFIED_COLUMNS = [
    "candidate_id",
    "source_row_id",
    "sentence_id",
    "correction_id",
    "candidate_type",
    "suggestion_index",
    "learning_language",
    "native_language",
    "learning_code",
    "native_code",
    "learner_sentence",
    "original_tagged_correction",
    "reconstructed_sentence",
    CLASSIFICATION_LEARNER_SENTENCE_FIELD,
    CLASSIFICATION_RECONSTRUCTED_SENTENCE_FIELD,
    "suggestion_evidence",
    "source_line_number",
    "source_sentence_index",
    "source_correction_index",
    "api_model",
    "api_status",
    "api_raw_response_path",
    "is_toy",
    "classification",
    "contains_constructional",
    "contains_lexical",
    "contains_morphological",
    "function_word_changes",
    "dependency_arc_changes",
    "lemma_changes",
    "morphological_feature_changes",
    "orthographic_changes",
    "token_alignment",
    "learner_conllu",
    "correction_conllu",
    "parse_status",
    "classification_notes",
    "stanza_word_edit_distance",
    "construction_group_id",
    "construction_group_rank",
    "construction_group_size",
    "construction_group_type",
    "construction_operation_pattern",
    "construction_group_signature",
    "construction_group_key",
    "construction_edit_count",
]

GROUP_SUMMARY_COLUMNS = [
    "construction_group_id",
    "construction_group_rank",
    "construction_group_type",
    "construction_operation_pattern",
    "construction_group_signature",
    "construction_group_size",
    "construction_edit_count",
    "learning_languages",
    "language_count",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="prepared_candidates.parquet (or .csv) from Component A")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", action="append", default=[], type=Path)
    parser.add_argument("--stanza-resources-dir", type=Path, default=DEFAULT_STANZA_RESOURCES_DIR)
    parser.add_argument("--languages", default=None, help="Comma-separated learning-language codes to process (e.g. fr,de); default: all six allowed languages")
    parser.add_argument("--batch-size", type=int, default=50, help="Candidates classified between checkpoint flushes")
    parser.add_argument("--use-gpu", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_candidates(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False, na_values=[""])
    return pd.read_parquet(path)


def _write_conllu_corpus(rows: list[dict], field_name: str, out_path: Path, overwrite: bool) -> None:
    blocks = [r[field_name] for r in rows if r.get(field_name)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    guard_overwrite(out_path, overwrite)
    with open(out_path, "w", encoding="utf-8") as fh:
        for block in blocks:
            fh.write(block)
            if not block.endswith("\n"):
                fh.write("\n")
            fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = output_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "logs" / "classify_errors.log")

    overrides = {"stanza.use_gpu": args.use_gpu}
    config = load_config(extra_config_paths=args.config, overrides=overrides)
    seed = config["random_seed"]
    function_word_upos = frozenset(config["stanza"]["function_word_upos"]) or FUNCTION_WORD_UPOS
    use_gpu = bool(config["stanza"]["use_gpu"])

    summary = RunSummary(command="classify_errors", argv=sys.argv, config=config, seed=seed)
    summary.record_inputs({"prepared_candidates": args.input})

    logger.info("Reading candidates from %s", args.input)
    df = _read_candidates(args.input)
    logger.info("Loaded %d candidate rows", len(df))

    allowed_codes = {LANGUAGE_ABBR[pair[0]] for pair in ALLOWED_PAIRS}
    df = df[df["learning_code"].isin(allowed_codes)]

    if args.languages:
        requested = {c.strip() for c in args.languages.split(",") if c.strip()}
        unknown = requested - allowed_codes
        if unknown:
            raise SystemExit(f"--languages contains unsupported code(s): {sorted(unknown)}; allowed: {sorted(allowed_codes)}")
        df = df[df["learning_code"].isin(requested)]

    logger.info("Classifying %d candidate rows across languages: %s", len(df), sorted(df["learning_code"].unique()))

    candidates = df.to_dict(orient="records")

    checkpoint_path = interim_dir / "classification_checkpoint.jsonl"
    if not args.resume:
        stale = checkpoint_path.exists()
        if stale and not args.overwrite:
            raise SystemExit(
                f"A previous classification checkpoint exists at {checkpoint_path}. Pass "
                "--resume to continue it, or --overwrite to discard it and start fresh."
            )
        if stale and args.overwrite:
            checkpoint_path.unlink()

    final_overwrite = args.overwrite or args.resume

    pipeline_manager = StanzaPipelineManager(args.stanza_resources_dir, use_gpu=use_gpu)

    run_result = classify_candidates(
        candidates,
        pipeline_manager,
        function_word_upos=function_word_upos,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        fail_fast=args.fail_fast,
        batch_size=args.batch_size,
    )
    summary.record(
        "classification_run",
        {
            "n_processed": run_result.n_processed,
            "n_skipped_already_done": run_result.n_skipped_already_done,
            "n_invalid": run_result.n_invalid,
        },
    )

    all_rows = load_classified_rows(checkpoint_path)
    selected_ids = {c["candidate_id"] for c in candidates}
    all_rows = [r for r in all_rows if r["candidate_id"] in selected_ids]

    _write_conllu_corpus(all_rows, "learner_conllu", output_dir / "parses" / "learner.conllu", final_overwrite)
    _write_conllu_corpus(all_rows, "correction_conllu", output_dir / "parses" / "correction.conllu", final_overwrite)

    all_rows, group_summary_rows = assign_construction_groups(all_rows, function_word_upos=function_word_upos)
    group_stats = summarize_construction_groups(all_rows, group_summary_rows)
    group_report = format_construction_group_report(group_stats)
    logger.info("%s", group_report)
    print(group_report)
    summary.record("construction_grouping", group_stats)

    group_df = pd.DataFrame(group_summary_rows, columns=GROUP_SUMMARY_COLUMNS)
    atomic_write_dataframe(group_df, output_dir, "construction_groups", overwrite=final_overwrite)

    csv_rows = []
    for row in all_rows:
        csv_row = dict(row)
        for field_name in JSON_LIST_FIELDS:
            csv_row[field_name] = json.dumps(csv_row.get(field_name) or [], ensure_ascii=False)
        csv_rows.append(csv_row)

    all_df = pd.DataFrame(csv_rows)

    if not all_df.empty:
        all_df = all_df[CLASSIFIED_COLUMNS]

        integer_columns = [
            "construction_group_rank",
            "construction_group_size",
            "construction_edit_count",
            "stanza_word_edit_distance",
        ]

        for col in integer_columns:
            all_df[col] = pd.to_numeric(
                all_df[col].replace("", pd.NA),
                errors="coerce",
            ).astype("Int64")
    else:
        all_df = pd.DataFrame(columns=CLASSIFIED_COLUMNS)

    atomic_write_dataframe(all_df, output_dir, "all_classified", overwrite=final_overwrite)

    def _subset(mask) -> pd.DataFrame:
        return all_df[mask] if not all_df.empty else all_df

    lexical_only = _subset(all_df.get("classification") == "lexical")
    morphological_only = _subset(all_df.get("classification") == "morphological")
    pure = _subset(all_df.get("classification") == "pure")
    constructional_others = _subset(all_df.get("classification") == "constructional+others")
    mixed = _subset(all_df.get("classification") == "mixed")
    word_order = _subset(all_df.get("classification") == "word_order")
    unchanged = _subset(all_df.get("classification") == "unchanged")
    invalid = _subset(all_df.get("classification") == "invalid")

    atomic_write_dataframe(pure, output_dir, "pure", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(constructional_others, output_dir, "constructional_others", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(lexical_only, output_dir, "lexical_only", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(morphological_only, output_dir, "morphological_only", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(mixed, output_dir, "mixed", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(word_order, output_dir, "word_order", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(unchanged, output_dir, "unchanged", overwrite=final_overwrite, formats=("csv",))
    atomic_write_dataframe(invalid, output_dir, "invalid_or_parse_failed", overwrite=final_overwrite, formats=("csv",))

    classification_counts = all_df["classification"].value_counts().to_dict() if not all_df.empty else {}
    by_language = (
        all_df.groupby(["learning_language", "classification"]).size().unstack(fill_value=0).to_dict(orient="index")
        if not all_df.empty
        else {}
    )
    summary.record_counts(
        {
            "input_candidates": len(candidates),
            "classified_total": len(all_df),
            "pure": len(pure),
            "constructional_others": len(constructional_others),
            "lexical_only": len(lexical_only),
            "morphological_only": len(morphological_only),
            "mixed": len(mixed),
            "word_order": len(word_order),
            "unchanged": len(unchanged),
            "invalid_or_parse_failed": len(invalid),
        }
    )
    summary.record("classification_counts", classification_counts)
    summary.record("classification_counts_by_language", by_language)
    summary.write(output_dir / "classification_summary.json")

    logger.info("Done. Outputs written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
